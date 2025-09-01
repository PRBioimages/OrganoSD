import os
import torch
import math
from torch import nn
import numpy as np
from torch.nn import functional as F
from config import organ512 as config
from model.prior.prior_box import PriorBox
# from model.detection import Detect
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class L2Norm(nn.Module):
    '''
    conv4_3特征图大小38x38，网络层靠前，norm较大，需要加一个L2 Normalization,以保证和后面的检测层差异不是很大，具体可以参考：ParseNet。这个前面的推文里面有讲。
    '''
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        # 将一个不可训练的类型Tensor转换成可以训练的类型 parameter
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
    # 初始化参数
    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        # 计算x的2范数
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() # shape[b,1,38,38]
        x = x / norm   # shape[b,512,38,38]
        out = self.weight[None,...,None,None] * x
        return out

class Conv_Block_down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block_down, self).__init__()
        self.group = 8
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=self.group),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=self.group),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=self.group),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU())
        self.L2Norm = L2Norm(out_channel, 20)
        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    def forward(self,x):
        identity = self.shortcut(x)
        out = self.layer1(x)
        for i in range(1):
            out = self.layer2(out)
        out += identity
        out = self.L2Norm(out)
        return out


class Conv_Block_up(nn.Module):
    def __init__(self, en_channel, de_channel):
        super(Conv_Block_up, self).__init__()
        self.group = 8
        self.in_channel = int(en_channel + de_channel)
        self.out_channel = int(en_channel)
        self.layer = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, groups=self.group),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1, groups=self.group),
            nn.BatchNorm2d(self.out_channel),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1, groups=self.group),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1, groups=self.group),
            nn.BatchNorm2d(self.out_channel),
            nn.LeakyReLU())

    def forward(self, Fen, Fde):
        Fde = F.interpolate(Fde, scale_factor=2, mode='nearest')
        out = torch.cat((Fen, Fde), dim=1)
        out = self.layer(out) + Fen
        return out


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        return self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, num_layers=3):
        super(TransformerBlock, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, in_channels))
        encoder_layers = TransformerEncoderLayer(d_model=in_channels, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # 将特征图展平并调整维度以适应 Transformer 输入
        x += self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2).view(b, c, h, w)  # 恢复特征图的原始形状
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out
class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def cosine_position_encoding(x_flat, h, w):
    b, seq_len, c = x_flat.shape
    num_pos_feats = c // 2
    position_h = torch.arange(0, h, dtype=torch.float, device=x_flat.device).unsqueeze(1)
    position_w = torch.arange(0, w, dtype=torch.float, device=x_flat.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, num_pos_feats, 2, dtype=torch.float, device=x_flat.device) * (
                -math.log(10000.0) / num_pos_feats))
    pos_emb_h = torch.zeros(1, h, num_pos_feats, dtype=torch.float, device=x_flat.device)
    pos_emb_h[0, :, 0::2] = torch.sin(position_h * div_term)
    pos_emb_h[0, :, 1::2] = torch.cos(position_h * div_term)
    pos_emb_w = torch.zeros(1, w, num_pos_feats, dtype=torch.float, device=x_flat.device)
    pos_emb_w[0, :, 0::2] = torch.sin(position_w * div_term)
    pos_emb_w[0, :, 1::2] = torch.cos(position_w * div_term)
    pos_emb_h_expanded = pos_emb_h.unsqueeze(2).expand(-1, -1, w, -1).flatten(1, 2)
    pos_emb_w_expanded = pos_emb_w.unsqueeze(1).expand(-1, h, -1, -1).flatten(1, 2)
    pos_emb = torch.cat((pos_emb_h_expanded, pos_emb_w_expanded), dim=-1).expand(b, -1, -1)
    x_flat_with_pos = x_flat + pos_emb
    return x_flat_with_pos

class CrossAttentionModule(nn.Module):
    def __init__(self, in_channels_fe, in_channels_fd):
        super(CrossAttentionModule, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_f6d = nn.Conv2d(in_channels_fd, in_channels_fd // 2, kernel_size=1)
        self.conv_output = nn.Sequential(
                        nn.Conv2d(in_channels_fe + in_channels_fd // 2, in_channels_fe, kernel_size=1),
                        nn.BatchNorm2d(in_channels_fe),
                        nn.ReLU())
        self.upsample = nn.Sequential(
                        nn.ConvTranspose2d(in_channels_fe, in_channels_fe, kernel_size=2, stride=2),
                        nn.BatchNorm2d(in_channels_fe),
                        nn.ReLU())
        self.query_proj = nn.Linear(in_channels_fe, in_channels_fe)
        self.key_proj = nn.Linear(in_channels_fe, in_channels_fe)
        self.value_proj = nn.Linear(in_channels_fd // 2, in_channels_fd // 2)
        self.channel_attention_f5e = SELayer(in_channels_fe)
        self.spatial_attention_f5e = SpatialAttentionModule()
        self.channel_attention_f6d = SELayer(in_channels_fd)
        self.spatial_attention_f6d = SpatialAttentionModule()

    def forward(self, F5e, F6d):
        # 处理 F5e
        F5e = self.channel_attention_f5e(F5e)
        F5e = self.spatial_attention_f5e(F5e)
        F5e_pos = self.max_pool(F5e)
        b, c, h, w = F5e_pos.shape
        F5e_pos = F5e_pos.flatten(2).transpose(1, 2)
        F5e_pos = cosine_position_encoding(F5e_pos, h, w)
        Q = self.query_proj(F5e_pos)
        K = self.key_proj(F5e_pos)
        # 处理 F6d
        F6d = self.channel_attention_f6d(F6d)
        F6d = self.spatial_attention_f6d(F6d)
        F6d_conv = self.conv_f6d(F6d)
        F6d_pos = F6d_conv.flatten(2).transpose(1, 2)
        F6d_pos = cosine_position_encoding(F6d_pos, h, w)
        V = self.value_proj(F6d_pos)
        # 计算注意力分数
        attn_output = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn_output = torch.softmax(attn_output, dim=-1)
        # 计算注意力输出
        attn_output = torch.matmul(attn_output, V)
        attn_output = attn_output.transpose(1, 2).view(b, -1, h, w)
        attn_output = self.upsample(attn_output)
        # 与 F5e 对应元素相乘
        attn_output = attn_output * F5e
        # 处理 F6d 用于拼接
        F6d_conv = self.upsample(F6d_conv)
        # 通道拼接
        # F5d = torch.cat((attn_output, F6d_conv), dim=1)
        F5d = torch.cat((F5e, F6d_conv), dim=1)
        F5d = self.conv_output(F5d)
        return F5d

class UNet512(nn.Module):
    def __init__(self, in_channels, cfg, out_channels, transformer_num_layers=2, num_heads=4, cross_attention_positions=[]):
        super(UNet512, self).__init__()
        self.n = len(cfg)
        self.Conv_down = []
        self.Conv_up = list(range(self.n))
        for k, v in enumerate(cfg):
            self.Conv_down.insert(k, Conv_Block_down(in_channels, v))
            if k < self.n - 1:
                self.Conv_up[k] = Conv_Block_up(cfg[self.n - 2 - k], cfg[self.n - 1 - k])
            in_channels = v

        self.down = DownSample()
        self.transformer = TransformerBlock(cfg[-1], num_heads, transformer_num_layers)
        self.spatial_attention = SpatialAttentionModule()
        self.se_layer = SELayer(cfg[-1])
        self.cross_attention_positions = cross_attention_positions
        self.cross_attention_modules = nn.ModuleList([
            CrossAttentionModule(cfg[self.n - 2 - pos], cfg[self.n - 1 - pos]) for pos in cross_attention_positions
        ])
        self.out = nn.Sequential(
            nn.Conv2d(int(cfg[0]), 32, 3, 1, 1, bias=True),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, 1, 1, bias=True),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, out_channels, 3, 1, 1, bias=True),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        R = []
        O = []
        R.insert(0, self.Conv_down[0](x))
        for k in range(self.n - 1):
            R.insert(k + 1, self.down(self.Conv_down[k + 1](R[k])))

        # R[-1] = self.transformer(R[-1])
        R[-1] = self.spatial_attention(R[-1])
        R[-1] = self.se_layer(R[-1])
        O.insert(0, R[-1])

        for k in range(self.n - 1):
            if k in self.cross_attention_positions:
                attn_idx = self.cross_attention_positions.index(k)
                O.insert(k + 1, self.cross_attention_modules[attn_idx](R[self.n - 2 - k], O[k]))
            else:
                O.insert(k + 1, self.Conv_up[k](R[self.n - 2 - k], O[k]))

        out = self.out(O[-1])
        return O, out

class LocHead(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(LocHead, self).__init__()
        self.channel = 256
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, self.channel, 1),
            nn.Conv2d(self.channel, self.channel, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel, self.channel, 1),
            nn.Conv2d(self.channel, self.channel, 1),
            nn.BatchNorm2d(self.channel),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel, out_channel, 1),
            nn.LeakyReLU(),
            )
    def forward(self, x):
        return self.layer(x)

class ConfHead(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(ConfHead, self).__init__()
        self.channel = 256
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, self.channel, 1),
            nn.Conv2d(self.channel, self.channel, 1),
            # nn.BatchNorm2d(self.channel),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel, self.channel, 1),
            nn.Conv2d(self.channel, self.channel, 1),
            nn.BatchNorm2d(self.channel),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel, out_channel, 1),
            nn.Sigmoid()
            )
    def forward(self, x):
        return self.layer(x)

class ODSeg_U(nn.Module):
    def __init__(self, phase, in_channels, unet_out_channels, num_classes):
        super(ODSeg_U, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(config)
        with torch.no_grad():
            self.priors = torch.autograd.Variable(self.priorbox.forward())
        self.unet = UNet512(in_channels=in_channels, cfg=config['cfg_unet'], out_channels=unet_out_channels,
                            transformer_num_layers=3,
                            num_heads=4,
                            cross_attention_positions=[0, 1, 2])
        loc = list()
        conf = list()
        self.map_leyers = [config['cfg_unet'][k] for k in [-1, -2, -3, -4]]
        for k, v in enumerate(self.map_leyers):
            loc.append(LocHead(v, config['prior_num'][len(self.map_leyers) - 1 - k]*4))
            conf.append(ConfHead(v, config['prior_num'][len(self.map_leyers) - 1 - k]*self.num_classes))

        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect()

    def forward(self, x):
        loc = list()
        conf = list()
        feturemaps, out_seg = self.unet(x)

        for (x, l, c) in zip(feturemaps, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # ********************************************************************

        if self.phase == "test":
            out_detect = self.detect.apply(2, 0, 200, 0.01, 0.45,
                                       loc.view(loc.size(0), -1, 4),  # loc preds
                                       self.softmax(conf.view(-1,
                                                              2)),  # conf preds
                                       self.priors.type(type(x.data))  # default boxes
                                       )
        else:
            out_detect = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return out_detect, out_seg

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



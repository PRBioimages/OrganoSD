import os
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from config import organ512
from model.prior.prior_box import PriorBox
from model.detection import Detect

base = {
    '512': [4,  8, 16, 32, 64, 128, 256, 512, 1024],
    }

class Conv_Block_down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block_down, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='zeros', bias=True),
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU())
    def forward(self,x):
        out = self.layer1(x)
        for i in range(3):
            out = self.layer2(out)
        return out

class Conv_Block_up0(nn.Module):
    def __init__(self,in_channel):
        super(Conv_Block_up0, self).__init__()
        self.out_channel = int(in_channel / 2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, self.out_channel, 3, 1, 1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(self.out_channel),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(self.out_channel),
            nn.LeakyReLU())
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

class Conv_Block_up(nn.Module):
    def __init__(self,in_channel):
        super(Conv_Block_up, self).__init__()
        self.channel = int(in_channel / 2)
        self.out_channel = int(in_channel / 4)
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,self.channel,3,1,1,padding_mode='zeros', bias=True),
            nn.BatchNorm2d(self.channel),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel, self.out_channel, 3, 1, 1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(self.out_channel),
            nn.LeakyReLU())
    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer=nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        return self.layer(x)

def up(lower, higher):
    x = F.interpolate(lower, scale_factor=2, mode='nearest')
    return torch.cat((x, higher), dim=1)

class UPSample(nn.Module):
    def __init__(self, in_channel, en_ratio):
        super(UPSample, self).__init__()
        in_channel = int(in_channel)
        self.en_out_channel = int(np.ceil(2 * in_channel * en_ratio))
        self.de_out_channel = int(2 * in_channel - self.en_out_channel)
        self.layer_en = nn.Sequential(
            nn.Conv2d(in_channel,self.en_out_channel,3,1,1,padding_mode='zeros', bias=True),
            nn.BatchNorm2d(self.en_out_channel),
            nn.LeakyReLU()
        )
        self.layer_de = nn.Sequential(
            nn.Conv2d(in_channel, self.de_out_channel, 3, 1, 1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(self.de_out_channel),
            nn.LeakyReLU()
        )
    def forward(self, lower, higher):
        lower = F.interpolate(lower, scale_factor=2, mode='nearest')
        lower = self.layer_en(lower)
        higher = self.layer_de(higher)
        return torch.cat((lower, higher), dim=1)

cfg512 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
class UNet512(nn.Module):
    def __init__(self,in_channels, cfg, out_channels):
        super(UNet512, self).__init__()
        # self.en_ratio = [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.1]
        self.en_ratio = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.n = len(cfg)
        self.Conv_down = []
        self.Conv_up = list(range(self.n-1))
        self.Conv_up.insert(0, Conv_Block_up0(cfg[-1]))
        self.up = list(range(self.n-1))
        for k, v in enumerate(cfg):
            self.Conv_down.insert(k, Conv_Block_down(in_channels, v))
            if k != self.n and k != 0 and k != 1:
                self.Conv_up[self.n - k] = Conv_Block_up(v)
            if k != 0:
                self.up[self.n -1 - k] = UPSample(v/2, self.en_ratio[self.n -1 - k])
            in_channels = v

        self.down = DownSample()

        self.out = nn.Sequential(
                    nn.Conv2d(int(cfg[1]), 32, 3, 1, 1, bias=True),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 16, 3, 1, 1, bias=True),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(),
                    nn.Conv2d(16, 8, 3, 1, 1, bias=True),
                    nn.BatchNorm2d(8),
                    nn.LeakyReLU(),
                    nn.Conv2d(8, out_channels, 3, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                    nn.Sigmoid())

    def forward(self,x):
        R = []
        O = []
        R.insert(0, self.Conv_down[0](x))
        for k in range(self.n-1):
            R.insert(k+1, self.Conv_down[k+1](self.down(R[k])))

        O.insert(0, self.Conv_up[0](R[- 1]))
        for k in range(self.n - 2):
            O.insert(k+1, self.Conv_up[k+1](self.up[k](O[k],R[self.n - 2 - k])))
        out = self.out(self.up[-1](O[self.n - 2], R[0]))

        return O, out

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class LocConf(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(LocConf, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, 512, 3, 1, 1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, out_channel, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_channel),
            )
    def forward(self, x):
        return self.layer(x)


class ODSeg_U(nn.Module):
    def __init__(self, phase, in_channels, unet_out_channels, cfg_unet, num_classes):
        super(ODSeg_U, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(organ512)
        with torch.no_grad():
            self.priors = torch.autograd.Variable(self.priorbox.forward())

        self.unet = UNet512(in_channels, cfg_unet, unet_out_channels)

        loc = list()
        conf = list()
        self.map_leyers = cfg_unet[-2:-7:-1]
        for k, v in enumerate(self.map_leyers):
            i = len(organ512['prior_num'])-1-k
            loc.append(LocConf(v, organ512['prior_num'][i]*4))
            conf.append(LocConf(v, organ512['prior_num'][i]*self.num_classes))

        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect()

    def forward(self, x):
        loc = list()
        conf = list()

        feturemaps, out_seg = self.unet(x)

        # apply multibox head to source layers
        #********************************************************************

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


if __name__ == '__main__':
    x=torch.randn(2,1,512,512)
    # net = UNet512(1, base['512'], 1)
    net = ODSeg_U('train', 1, 1, cfg512, 1)
    x, y = net(x)
    print(x.shape)
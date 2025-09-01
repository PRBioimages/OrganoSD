import os

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from config import organ
from model.prior.prior_box import PriorBox
from model.detection import Detect

base = {
    '512': [4,  8, 16, 32, 64, 128, 256, 512, 1024],
    '732': [32, 64, 128, 256, 512],
    '732complex': [64, 128, 256, 512, 1024],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '732': [2, 2, 4, 4, 6, 2],
}

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '732': ['M', 256, 'M', 64, 32],
    '732complex': ['M', 512, 'M', 128, 64],
        }

extra_add = {'732': 512,
            '732complex': 1024,
             }

mbox_layer = {
        '732': [512, 256, 128, 64, 32, 16],
        '732complex': [1024, 512, 256, 128, 64, 32],
        }



class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,0,bias=True),
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 0, bias=True),
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class Trans_Conv(nn.Module):
    def __init__(self,in_channel):
        super(Trans_Conv, self).__init__()
        self.channel = int(in_channel / 2)
        self.out_channel = int(in_channel / 4)
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,self.channel,3,1,2,bias=True),
            nn.BatchNorm2d(self.channel),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel, self.out_channel, 3, 1, 2, bias=True),
            nn.BatchNorm2d(self.out_channel),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class Trans_Conv0(nn.Module):
    def __init__(self,in_channel):
        super(Trans_Conv0, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,1,2,bias=True),
            nn.BatchNorm2d(in_channel),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, int(in_channel/2), 3, 1, 2, bias=True),
            nn.BatchNorm2d(int(in_channel/2)),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )
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



def add_extras(cfg, in_channels):
    layers = []
    for k, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.Conv2d(in_channels, cfg[k+1], kernel_size=1, stride=1)]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            in_channels = cfg[k+1]
        else:
            layers += [nn.Conv2d(in_channels, int(v/2), kernel_size=1, stride=1)]
            layers += [nn.Conv2d(int(v/2), int(v/2), kernel_size=3, stride=2, padding=1, padding_mode='zeros')]
            in_channels = int(v/2)
    return layers

def multibox(cfg, num_classes, channels):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(channels):
        loc_layers += [nn.Conv2d(v, cfg[k]* 4, 3,1,1, padding_mode='zeros')]
        conf_layers += [nn.Conv2d(v, cfg[k]* num_classes, 3,1,1, padding_mode='zeros')]
    return (loc_layers, conf_layers)




class UNet(nn.Module):
    def __init__(self,in_channels, cfg, out_channels):
        super(UNet, self).__init__()
        self.n = len(cfg)
        self.Conv_down = []
        self.Conv_up = list(range(self.n))
        self.Conv_up.insert(0, Trans_Conv0(cfg[-1]))
        for k, v in enumerate(cfg):
            self.Conv_down.insert(k, Conv_Block(in_channels, v))
            if k != 0:
                self.Conv_up[self.n - k] = Trans_Conv(v)
            in_channels = v

        self.down = DownSample()

        self.out = nn.Sequential(
                    nn.Conv2d(int(cfg[0]/2), out_channels, 3, 1, 1, padding_mode='reflect', bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.Sigmoid())

    def forward(self,x):
        R = []
        O = []
        R.insert(0, self.Conv_down[0](x))
        for k in range(self.n-1):
            R.insert(k+1, self.Conv_down[k+1](self.down(R[k])))

        O.insert(0, self.Conv_up[0](R[self.n - 1]))
        for k in range(self.n - 1):
            O.insert(k+1, self.Conv_up[k+1](up(O[k],R[self.n - 2 - k])))

        return R[self.n - 1], self.out(O[self.n-1])

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


class ODSeg(nn.Module):
    def __init__(self, phase, in_channels, unet_out_channels, cfg_unet, extras, head, num_classes):
        super(ODSeg, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(organ)
        with torch.no_grad():
            self.priors = torch.autograd.Variable(self.priorbox.forward())

        self.unet = UNet(in_channels, cfg_unet, unet_out_channels)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect()

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        x, out_seg = self.unet(x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.leaky_relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(F.leaky_relu(l(x)).permute(0, 2, 3, 1).contiguous())
            conf.append(torch.sigmoid(c(x)).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
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



def build_odseg(phase, size=732, num_classes=1, unetcfg = '732', extracfg = '732', add = '732', mbox_ = '732'):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 732:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    extra_layers = add_extras(extras[extracfg], extra_add[add])
    head = multibox(mbox[str(size)], 1, mbox_layer[mbox_])

    return ODSeg(phase, 1, num_classes, base[unetcfg], extra_layers, head, 1)


if __name__ == '__main__':
    x=torch.randn(2,1,732,732)
    net = build_odseg('train',732,1)
    x, y = net(x)
    print(net(x).shape)
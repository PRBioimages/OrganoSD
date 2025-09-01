import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
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


class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)



base = {
    '512': [4,  8, 16, 32, 64, 128, 256, 512, 1024,
            ],
    '300': [],
}

class UNet(nn.Module):
    def __init__(self,in_channels, cfg):
        super(UNet, self).__init__()
        self.cd = []
        self.cu = []
        self.u = []
        self.n = len(cfg)
        for k, v in enumerate(cfg):
            self.cd[k] = Conv_Block(in_channels, v)
            if k != 0:
                self.cu[self.n - k - 1] = Conv_Block(v, in_channels)
                self.u[self.n - k - 1] = UpSample(v)
            in_channels = v
        self.d = DownSample()

        self.out = nn.Sequential(
                    nn.Conv2d(cfg[0], 32, 3, 1, 1, padding_mode='reflect', bias=False),
                    nn.BatchNorm2d(1),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 1, 3, 1, 1, padding_mode='reflect', bias=False),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid())

    def forward(self,x):
        R = []
        O = []
        R[0]=self.cd[0](x)
        for k in range(self.n-1):
            R[k+1] = self.cd[k+1](self.d(R[k]))

        O[0]=self.cu[0](self.u[0](R[self.n-1],R[self.n-2]))
        for k in range(self.n - 2):
            O[k+1]=self.cu[k+1](self.u[k+1](O[k],R[self.n - 3 - k]))

        return self.out(O[self.n - 2])

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)



if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=UNet()
    print(net(x).shape)
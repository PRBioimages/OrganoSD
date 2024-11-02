from torch import nn
import numpy as np
import torch
from .spectral_norm import spectral_norm, remove_spectral_norm
import os

# An autoencoder for 2D images using modified residual layers
# This is effectively a 2D simplified version of vaegan3D_cgan_target2.py

def get_activation(activation):
    if activation is None or activation.lower() == "none":
        return torch.nn.Sequential()
    elif activation.lower() == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation.lower() == "prelu":
        return torch.nn.PReLU()
    elif activation.lower() == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation.lower() == "leakyrelu":
        return torch.nn.LeakyReLU(0.2, inplace=True)
    elif activation.lower() == "softplus":
        return torch.nn.Softplus()

class PadLayer(nn.Module):
    def __init__(self, pad_dims):
        super(PadLayer, self).__init__()
        self.pad_dims = pad_dims

    def forward(self, x):
        if np.sum(self.pad_dims) == 0:
            return x
        else:
            return nn.functional.pad(x, [0, self.pad_dims[2], 0, self.pad_dims[1], 0, self.pad_dims[0]],
                "constant", 0,)

class DownLayerResidual(nn.Module):
    def __init__(self, ch_in, ch_out, activation="sigmoid", activation_last=None):
        super(DownLayerResidual, self).__init__()
        if activation_last is None:
            activation_last = activation
        self.bypass = nn.Sequential(nn.AvgPool2d(2, stride=2, padding=0),
            spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=True)),)
        self.resid = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_in, ch_in, 4, 2, padding=1, bias=True)),
            nn.BatchNorm2d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv2d(ch_in, ch_out, 3, 1, padding=1, bias=True)),
            nn.BatchNorm2d(ch_out),)
        self.activation = get_activation(activation_last)

    def forward(self, x):
        x = self.bypass(x) + self.resid(x)
        x = self.activation(x)
        return x

class UpLayerResidual(nn.Module):
    def __init__(self, ch_in, ch_out, activation="sigmoid", output_padding=0, activation_last=None):
        super(UpLayerResidual, self).__init__()
        if activation_last is None:
            activation_last = activation
        self.bypass = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
            nn.Upsample(scale_factor=2),
            PadLayer(output_padding),)
        self.resid = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ch_in, ch_in, 4, 2, padding=1, bias=True,)),
            nn.BatchNorm2d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv2d(ch_in, ch_out, 3, 1, padding=1, bias=True)),
            nn.BatchNorm2d(ch_out),
            PadLayer(output_padding),)
        self.activation = get_activation(activation_last)

    def forward(self, x):
        x = self.bypass(x) + self.resid(x)
        x = self.activation(x)
        return x

class Enc(nn.Module):
    def __init__(self, n_latent_dim=512, n_ch=1, conv_channels_list=[x * 2 for x in [2, 4, 8, 16, 32]], imsize_compressed=[3, 3], mode=None):
        super(Enc, self).__init__()
        self.n_latent_dim = n_latent_dim
        self.target_path = nn.ModuleList([DownLayerResidual(n_ch, conv_channels_list[0])])
        self.mode = mode
        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(DownLayerResidual(ch_in, ch_out))
            ch_in = ch_out
        self.latent_out = spectral_norm(
            nn.Linear(ch_in * int(np.prod(imsize_compressed)), self.n_latent_dim, bias=True))
        if self.mode == 'VAE':
            self.logvar_out = spectral_norm(
                nn.Linear(ch_in * int(np.prod(imsize_compressed)), self.n_latent_dim, bias=True))

    def forward(self, x_target):
        for target_path in self.target_path:
            x_target = target_path(x_target)
        x_target = x_target.view(x_target.size()[0], -1)
        z = self.latent_out(x_target)
        if self.mode == 'VAE':
            logvar = self.logvar_out(x_target)
            return z, logvar
        else:
            return z

class Dec(nn.Module):
    def __init__(self, n_latent_dim=512, padding_latent=[0, 1, 1], imsize_compressed=[3, 3],
                 n_ch=1, conv_channels_list=[x * 2 for x in [32, 16, 8, 4, 2]], activation_last="sigmoid",):
        super(Dec, self).__init__()
        self.padding_latent = padding_latent
        self.imsize_compressed = imsize_compressed
        self.ch_first = conv_channels_list[0]
        self.n_latent_dim = n_latent_dim
        self.n_channels = n_ch
        self.target_fc = spectral_norm(nn.Linear(self.n_latent_dim,
                conv_channels_list[0] * int(np.prod(self.imsize_compressed)), bias=True,))

        self.target_bn_relu = nn.Sequential(nn.BatchNorm2d(conv_channels_list[0]), nn.ReLU(inplace=True))
        self.target_path = nn.ModuleList([])
        l_sizes = conv_channels_list
        for i in range(len(l_sizes) - 1):
            if i == 2:
                padding = padding_latent
            else:
                padding = 0
            self.target_path.append(UpLayerResidual(l_sizes[i], l_sizes[i + 1], output_padding=padding))
        self.target_path.append(UpLayerResidual(l_sizes[i + 1], n_ch, activation_last=activation_last))

    def forward(self, z_target):
        x_target = self.target_fc(z_target).view(z_target.size()[0],self.ch_first,
                    self.imsize_compressed[0],self.imsize_compressed[1],)
        x_target = self.target_bn_relu(x_target)
        for target_path in self.target_path:
            x_target = target_path(x_target)
        return x_target

class ResAE(nn.Module):
    def __init__(self, encoded_space_dim=100):
        super().__init__()
        n_latent_dim = encoded_space_dim
        self.encoder = Enc(n_latent_dim=n_latent_dim)
        self.decoder = Dec(n_latent_dim=n_latent_dim)

    def forward(self, x):
        latent_code = self.encoder(x)
        x = self.decoder(latent_code)
        return x, latent_code

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


class ResVAE(nn.Module):
    def __init__(self, encoded_space_dim=100):
        super().__init__()
        n_latent_dim = encoded_space_dim
        self.encoder = Enc(n_latent_dim=n_latent_dim, mode='VAE')
        self.decoder = Dec(n_latent_dim=n_latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent_code = self.reparameterize(mu, logvar)
        pred = self.decoder(latent_code)
        return pred, latent_code, mu, logvar

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
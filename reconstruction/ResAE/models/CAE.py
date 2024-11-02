import torch
from torch import nn
from torch.nn import functional as F
from .spectral_norm import spectral_norm
import os

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(25 * 25 * 32, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, encoded_space_dim),
            nn.ReLU(True),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(100 * 100 * 1, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, encoded_space_dim)
        )
    def forward(self, x):
        x1 = self.encoder_cnn(x)
        x1 = self.flatten(x1)
        x1 = self.encoder_lin(x1)
        x2 = self.flatten(x)
        x2 = self.lin2(x2)
        out = x1 + x2
        return out


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 25 * 25 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 25, 25))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            )
        self.unflatten2 = nn.Unflatten(dim=1, unflattened_size=(1, 100, 100))
        self.lin2 = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 100 * 100 * 1)
        )
    def forward(self, x):
        x1 = self.decoder_lin(x)
        x1 = self.unflatten(x1)
        x1 = self.decoder_conv(x1)
        x2 = self.lin2(x)
        x2 = self.unflatten2(x2)
        x = torch.sigmoid(x1 + x2)
        return x

class AEadd(nn.Module):
    def __init__(self, encoded_space_dim=100):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim)
        self.decoder = Decoder(encoded_space_dim)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class AE(nn.Module):
    def __init__(self, encoded_space_dim=100):
        super().__init__()
        self.encoder = Encoder1(encoded_space_dim)
        self.decoder = Decoder1(encoded_space_dim)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class Encoder1(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(25 * 25 * 32, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, encoded_space_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder1(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 25 * 25 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 25, 25))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VAE(nn.Module):
    def __init__(self, dim = 100):
        super().__init__()
        self.dim = dim
        #encode
        self.fc_en1 = spectral_norm(nn.Linear(100 * 100, 2048))
        self.fc_en2 = spectral_norm(nn.Linear(2048, 512))
        self.fc_enmu = spectral_norm(nn.Linear(512, self.dim))
        self.fc_envar = spectral_norm(nn.Linear(512, self.dim))
        #decode
        self.fc_de1 = spectral_norm(nn.Linear(self.dim, 512))
        self.fc_de2 = spectral_norm(nn.Linear(512, 2048))
        self.fc_de3 = spectral_norm(nn.Linear(2048, 100*100))
        #flatten
        self.flatten = nn.Flatten(start_dim=1)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 100, 100))

    def encode(self, x):
        x = self.flatten(x)
        h = F.leaky_relu(self.fc_en1(x))
        h = F.leaky_relu(self.fc_en2(h))
        return torch.sigmoid(self.fc_enmu(h)), torch.sigmoid(self.fc_envar(h))

    def reparameterize(self, mu, logvar):
        x = mu.detach().numpy()
        y = logvar.detach().numpy()
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        o = F.leaky_relu(self.fc_de1(z))
        o = F.leaky_relu(self.fc_de2(o))
        out = torch.sigmoid(self.fc_de3(o))
        return self.unflatten(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = z.detach().numpy()
        return self.decode(z), z, mu, logvar

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
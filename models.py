import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

# DCGAN-like Generator & Discriminator for 28x28 MNIST

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, 3, 1, 0, bias=False),  # 1x1 -> 3x3
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), # 3x3 -> 6x6
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),  # 6x6 -> 12x12
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 2, bias=False),     # 12x12 -> 28x28
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=1, use_sigmoid=False, spectral=False):
        super().__init__()
        def conv(in_c, out_c, k, s, p, spectral=False):
            layer = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
            return SN(layer) if spectral else layer

        self.main = nn.Sequential(
            conv(nc, ndf, 4, 2, 1, spectral),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ndf, ndf*2, 4, 2, 1, spectral),
            nn.BatchNorm2d(ndf*2) if not spectral else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ndf*2, ndf*4, 3, 2, 1, spectral),
            nn.BatchNorm2d(ndf*4) if not spectral else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ndf*4, 1, 3, 1, 0, spectral)
        )
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out.squeeze(1)

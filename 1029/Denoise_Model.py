import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """通道注意力机制"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Denoise_Model(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Denoise_Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.middle = nn.Sequential(
            ResidualBlock(base_channels),
            SEBlock(base_channels),
            ResidualBlock(base_channels),
            SEBlock(base_channels),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1),
        )

    def forward(self, x):
        residual = x
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return residual + x  # 残差学习


def ResidualBlock(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, 1, 1),
        nn.InstanceNorm2d(channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(channels, channels, 3, 1, 1),
        nn.InstanceNorm2d(channels)
    )
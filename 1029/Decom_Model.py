import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, 4, 2, 1),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 256
        e2 = self.enc2(e1)  # 128
        e3 = self.enc3(e2)  # 64

        # Bottleneck
        b = self.bottleneck(e3)  # 64

        # Decoder
        d3 = self.dec3(b)  # 128
        d2 = self.dec2(torch.cat([d3, e2], 1))  # 256
        d1 = self.dec1(torch.cat([d2, e1], 1))  # 256

        return d1


class Decom_Model(nn.Module):
    def __init__(self):
        super(Decom_Model, self).__init__()
        self.R_net = UNet(3, 3)  # 反射层网络
        self.L_net = UNet(3, 1)  # 照明层网络

    def forward(self, x):
        R = self.R_net(x)
        L = self.L_net(x)

        # 确保照明层在合理范围
        L = torch.sigmoid(L)
        R = torch.sigmoid(R)

        return R, L
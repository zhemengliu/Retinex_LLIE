import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.transforms import functional as tvF
from PIL import Image, ImageFont, ImageDraw
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from sys import platform
from string import ascii_letters
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PSolver(nn.Module):
    """
    P求解器：min(P) = ||I - P*Q||² + γ||P - R||²
    解析解：P = (I*Q + γ*R) / (Q² + γ)
    """

    def __init__(self):
        super(PSolver, self).__init__()

    def forward(self, I, Q, R, gamma):
        """
        Args:
            I: 输入图像 [B, 3, H, W]
            Q: 光照分量 [B, 1, H, W]
            R: 恢复的反射分量 [B, 3, H, W]
            gamma: 正则化系数
        """
        epsilon = 1e-8

        # 扩展Q的维度以匹配I和R
        Q_expanded = Q.expand_as(I)

        # 计算分子和分母
        numerator = I * Q_expanded + gamma * R
        denominator = Q_expanded * Q_expanded + gamma + epsilon

        P = numerator / denominator

        return P


class QSolver(nn.Module):
    """
    Q求解器：min(Q) = ||I - P*Q||² + λ||Q - L||²
    解析解：Q = (Σ(I_r*P_r) + λ*L) / (Σ(P_r²) + λ)
    """

    def __init__(self):
        super(QSolver, self).__init__()

    def forward(self, I, P, L, lamda):
        """
        Args:
            I: 输入图像 [B, 3, H, W]
            P: 反射分量 [B, 3, H, W]
            L: 恢复的光照分量 [B, 1, H, W]
            lamda: 正则化系数
        """
        epsilon = 1e-8

        # 分离RGB通道
        I_r, I_g, I_b = I[:, 0:1, :, :], I[:, 1:2, :, :], I[:, 2:3, :, :]
        P_r, P_g, P_b = P[:, 0:1, :, :], P[:, 1:2, :, :], P[:, 2:3, :, :]

        # 计算分子和分母
        numerator = (I_r * P_r + I_g * P_g + I_b * P_b) + lamda * L
        denominator = (P_r * P_r + P_g * P_g + P_b * P_b) + lamda + epsilon

        Q = numerator / denominator

        return Q


class InitDecomNet(nn.Module):
    """初始分解网络 - 生成P0和L0"""

    def __init__(self, in_channels=3, channels=32):
        super(InitDecomNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 4, 3, 1, 1),  # 输出4通道: R(3) + L(1)
            nn.ReLU()  # 确保非负
        )

    def forward(self, x):
        output = self.net(x)
        P = output[:, 0:3, :, :]  # 反射分量
        L = output[:, 3:4, :, :]  # 光照分量

        P = torch.clamp(P, 0.0, 1.0)
        L = torch.clamp(L, 0.0, 1.0)

        return P, L


class R_Restore(nn.Module):
    """增强的反射层恢复网络"""

    def __init__(self, channels=64):
        super(R_Restore, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, P, Q):
        x = torch.cat([P, Q], dim=1)
        R_residual = self.net(x)
        R = P + 0.2 * R_residual  # 增强残差连接
        R = torch.sigmoid(R)
        return R


class I_Restore(nn.Module):
    """增强的光照层恢复网络"""

    def __init__(self, channels=64):
        super(I_Restore, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, Q):
        L_residual = self.net(Q)
        L = Q + 0.2 * L_residual  # 增强残差连接
        L = torch.sigmoid(L)
        return L


class DecomNet(nn.Module):
    """增强版URetinex分解网络"""

    def __init__(self, unfolding_rounds=5, init_channels=64, gamma=0.05, lamda=0.05,
                 gamma_offset=0.02, lamda_offset=0.02):
        super(DecomNet, self).__init__()

        self.unfolding_rounds = unfolding_rounds
        self.init_decom = InitDecomNet(in_channels=3, channels=init_channels)
        self.P_solver = PSolver()
        self.Q_solver = QSolver()
        self.R_restore = R_Restore(channels=init_channels)
        self.L_restore = I_Restore(channels=init_channels)

        # 调整正则化参数
        self.gamma = gamma
        self.lamda = lamda
        self.gamma_offset = gamma_offset
        self.lamda_offset = lamda_offset

    def forward(self, I):
        P, Q, R, L = None, None, None, None

        for t in range(self.unfolding_rounds):
            if t == 0:
                P, Q = self.init_decom(I)
                w_p = self.gamma
                w_q = self.lamda
            else:
                w_p = self.gamma + self.gamma_offset * t
                w_q = self.lamda + self.lamda_offset * t
                P = self.P_solver(I, Q, R, w_p)
                Q = self.Q_solver(I, P, L, w_q)

            R = self.R_restore(P, Q)
            L = self.L_restore(Q)

            # 更宽松的约束
            P = torch.clamp(P, 0.0, 1.0)
            Q = torch.clamp(Q, 0.0, 1.0)
            R = torch.clamp(R, 0.0, 1.0)
            L = torch.clamp(L, 0.0, 1.0)

        return R, L


def test_decom_net():
    """测试分解网络"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型 - 直接传递参数而不是args对象
    net = DecomNet(
        unfolding_rounds=5,
        init_channels=64,
        gamma=0.05,
        lamda=0.05,
        gamma_offset=0.02,
        lamda_offset=0.02
    ).cuda()

    x = torch.rand(1, 3, 128, 128).cuda()
    R, L = net(x)

    print(f"R范围: [{R.min():.4f}, {R.max():.4f}], L范围: [{L.min():.4f}, {L.max():.4f}]")
    print(f"R标准差: {R.std():.4f}, L标准差: {L.std():.4f}")
    print(f"R形状: {R.shape}, L形状: {L.shape}")

    # # 重建检查
    # recon = R * L
    # recon_error = F.l1_loss(recon, x)
    # print(f"重建误差: {recon_error:.6f}")

    print("✅ 分解网络测试成功！")


if __name__ == '__main__':
    # 不再使用argparse，直接调用测试函数
    test_decom_net()
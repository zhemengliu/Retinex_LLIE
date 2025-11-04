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



# -------------------------- 2. 基础工具函数（保留Uretinex原始架构） --------------------------
def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    """Uretinex原始卷积层封装，未修改"""
    return nn.Conv2d(
        in_channels=in_c, out_channels=out_c,
        kernel_size=k, stride=s, padding=p,
        dilation=dilation, groups=groups
    )


def get_batchnorm_layer(opts):
    """Uretinex原始批归一化层封装，未修改"""
    if opts.norm_layer == "batch":
        return nn.BatchNorm2d
    elif opts.norm_layer == "spectral_instance":
        return nn.InstanceNorm2d
    else:
        print("not implemented")
        exit()


class Identity(nn.Module):
    """Uretinex原始恒等映射层，未修改"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# -------------------------- 3. 迭代Uretinex分解（100%保留原始P/Q更新逻辑） --------------------------
class P(nn.Module):
    """原始P求解模块：min(P) = ||I-PQ||^2 + γ||P-R||^2 → 公式未修改"""

    def __init__(self):
        super().__init__()

    def forward(self, I, Q, R, gamma):
        return ((I * Q + gamma * R) / (gamma + Q * Q + 1e-6))  # 仅加1e-8防除零，公式不变

        denominator = gamma + Q * Q + 1e-6  # 增加epsilon防止除零
        return ((I * Q + gamma * R) / denominator)

class Q(nn.Module):
    """原始Q求解模块：min(Q) = ||I-PQ||^2 + λ||Q-L||^2 → 通道拆分+公式未修改"""

    def __init__(self):
        super().__init__()

    def forward(self, I, P, L, lamda):
        # 原始R/G/B通道拆分逻辑，未修改
        IR = I[:, 0:1, :, :]
        IG = I[:, 1:2, :, :]
        IB = I[:, 2:3, :, :]
        PR = P[:, 0:1, :, :]
        PG = P[:, 1:2, :, :]
        PB = P[:, 2:3, :, :]
        # 原始公式，未修改
        numerator = (IR * PR + IG * PG + IB * PB) + lamda * L
        denominator = (PR * PR + PG * PG + PB * PB) + lamda + 1e-6
        return numerator / denominator


class Decom(nn.Module):
    """Uretinex原始分解网络（初始化P0/L0），结构未修改"""

    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
            nn.ReLU()
        )

        # 新增：权重初始化（Xavier初始化，适合ReLU类激活函数）
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  # 偏置初始化为0，避免偏移过大

    def forward(self, input):
        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        R = torch.clamp(R, 0.0, 1.0)
        L = torch.clamp(L, 0.0, 1.0)
        return R, L


class SELayer(nn.Module):
    """Uretinex原始SE层，结构未修改"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HalfDnCNNSE(nn.Module):
    """Uretinex原始反射层恢复网络，结构未修改"""

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if self.opts.concat_L:
            self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = get_conv2d_layer(in_c=1, out_c=32, k=3, s=1, p=1)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.conv1 = get_conv2d_layer(in_c=3, out_c=64, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
        self.se_layer = SELayer(channel=64)
        self.conv3 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = get_conv2d_layer(in_c=64, out_c=3, k=3, s=1, p=1)

    def forward(self, r, l):
        if self.opts.concat_L:
            r_fs = self.relu1(self.conv1(r))
            l_fs = self.relu2(self.conv2(l))
            inf = torch.cat([r_fs, l_fs], dim=1)
            se_inf = self.se_layer(inf)
        else:
            r_fs = self.relu1(self.conv1(r))
            se_inf = self.se_layer(r_fs)
        x1 = self.relu3(self.conv3(se_inf))
        x2 = self.relu4(self.conv4(x1))
        x3 = self.relu5(self.conv5(x2))
        x4 = self.relu6(self.conv6(x3))
        x5 = self.relu7(self.conv7(x4))
        n = self.conv8(x5)
        r_restore = r + n
        r_restore = torch.clamp(r_restore, 0.0, 1.0)
        return r_restore


# Decom_Model.py
class IterativeUretinex(nn.Module):
    def __init__(self, opts=None, unfolding_round=3):
        super().__init__()
        # 设置默认参数
        if opts is None:
            opts = argparse.Namespace(
                gamma=0.1,
                lamda=0.1,
                Roffset=0.05,
                Loffset=0.05,
                concat_L=False
            )

        self.opts = opts
        self.unfolding_round = unfolding_round
        self.init_decom = Decom()
        self.P_solver = P()
        self.Q_solver = Q()
        self.r_restore_net = HalfDnCNNSE(opts)

    def forward(self, input_img):
        input_img = torch.clamp(input_img, 0.0, 1.0)

        # 初始分解
        R, L = self.init_decom(input_img)
        P, Q = R, L

        # 迭代优化
        for t in range(self.unfolding_round):
            if t > 0:
                w_p = self.opts.gamma + self.opts.Roffset * t
                w_q = self.opts.lamda + self.opts.Loffset * t

                P = self.P_solver(I=input_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q_solver(I=input_img, P=P, L=L, lamda=w_q)

                # 温和的数值约束
                P = torch.clamp(P, 0.01, 0.99)
                Q = torch.clamp(Q, 0.01, 0.99)

            # 只在最后一次迭代进行反射层恢复
            if t == self.unfolding_round - 1:
                R = self.r_restore_net(r=P, l=Q)
            else:
                R = P

            L = Q

        return R, L


if __name__== '__main__':
    parser = argparse.ArgumentParser(description="Low Light Enhancement (Uretinex+Noise2noise+Zero-DCE)")
    parser.add_argument("--unfolding_round", type=int, default=3, help="Uretinex iterative rounds")
    parser.add_argument("--gamma", type=float, default=0.1, help="P's lambda (原始参数)")
    parser.add_argument("--lamda", type=float, default=0.1, help="Q's lambda (原始参数)")
    parser.add_argument("--Roffset", type=float, default=0.05, help="gamma increment (原始参数)")
    parser.add_argument("--Loffset", type=float, default=0.05, help="lamda increment (原始参数)")
    parser.add_argument("--concat_L", type=bool, default=False, help="Concat L to R (原始参数)")
    args = parser.parse_args()
    x = torch.rand(1,3,128,128).cuda()
    net=IterativeUretinex(args).cuda()
    R , L=net(x)
    print(R.shape,L.shape)


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

# -------------------------- 1. 批归一化层 ---------------------------------
def get_batchnorm_layer(opts):
    """Uretinex原始批归一化层封装"""
    if opts.norm_layer == "batch":
        return nn.BatchNorm2d
    elif opts.norm_layer == "spectral_instance":
        return nn.InstanceNorm2d
    else:
        print("not implemented")
        exit()

# -------------------------- 2. 基础工具函数（保留Uretinex原始架构） --------------------------
def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    """Uretinex原始卷积层封装，未修改"""
    '''
        in_c：输入通道数。
        out_c：输出通道数。
        k：卷积核大小。
        s：步幅。
        p：填充（默认值为0）。
        dilation：卷积核扩展（默认值为1）。
        groups：分组卷积（默认值为1）。
        '''
    return nn.Conv2d(
        in_channels=in_c, out_channels=out_c,
        kernel_size=k, stride=s, padding=p,
        dilation=dilation, groups=groups
    )

#--------------------------------3.反卷积层---------------------------------------------#
def get_deconv2d_layer(in_c, out_c, k=1, s=1, p=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"), # 进行双线性插值,将输入的特征图上采样2倍。
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=s,
            padding=p
        )
    )

#--------------------------------4.恒等映射层-----------------------------------------
class Identity(nn.Module):
    """Uretinex原始恒等映射层，未修改"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# -------------------------- 3. 迭代Uretinex分解（100%保留原始P/Q更新逻辑） --------------------------
class P(nn.Module):
    """原始P求解模块：min(P) = ||I-PQ||^2 + γ||P-R||^2 ————→ P* = (gamma*R + I*Q) / (Q*Q + gamma)"""

    def __init__(self):
        super().__init__()

    def forward(self, I, Q, R, gamma):
        return ((I * Q + gamma * R) / (gamma + Q * Q + 1e-6))  # 仅加1e-8防除零，公式不变


class Q(nn.Module):
    """原始Q求解模块：min(Q) = ||I-PQ||^2 + λ||Q-L||^2 ————→ 通道拆分 +  Q* = (lamda*L + I*P) / (P*P + lamda)"""

    def __init__(self):
        super().__init__()

    def forward(self, I, P, L, lamda):
        '''
            I: 输入图像或特征图。
            P: 一个特征图或滤波器。
            L: 参考图像或特征图。
            lamda: 正则化参数。
        '''
        # 将 I 和 P 分别分成红、绿、蓝三个通道。
        # 原始R/G/B通道拆分
        IR = I[:, 0:1, :, :]
        IG = I[:, 1:2, :, :]
        IB = I[:, 2:3, :, :]

        PR = P[:, 0:1, :, :]
        PG = P[:, 1:2, :, :]
        PB = P[:, 2:3, :, :]
        # 原始公式
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

    def forward(self, input):
        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        R = torch.clamp(R, 0.0, 1.0)
        L = torch.clamp(L, 0.0, 1.0)
        return R, L


class HalfDnCNNSE(nn.Module):
    """Uretinex原始反射层恢复网络，结构未修改"""

    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        if self.opts.concat_L: # 一个布尔值，决定是否将 l 输入拼接到 r 输入
            self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = get_conv2d_layer(in_c=1, out_c=32, k=3, s=1, p=1)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.conv1 = get_conv2d_layer(in_c=3, out_c=64, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
        self.se_layer = SELayer(channel=64) # 引入 SE 层来提升通道间的特征表示能力。
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

        self.conv8 = get_conv2d_layer(in_c=64, out_c=3, k=3, s=1, p=1) # 将特征图还原到3个通道。

    def forward(self, r, l):
        if self.opts.concat_L:   # 如果 self.opts.concat_L 为真，则分别处理 r 和 l，然后将它们拼接。
            r_fs = self.relu1(self.conv1(r))
            l_fs = self.relu2(self.conv2(l))
            inf = torch.cat([r_fs, l_fs], dim=1)
            se_inf = self.se_layer(inf)  # 最后一层卷积输出 n，然后将 n 加到 r 上，得到最终的恢复图像 r_restore。
        else:
            r_fs = self.relu1(self.conv1(r))
            se_inf = self.se_layer(r_fs)
        x1 = self.relu3(self.conv3(se_inf))
        x2 = self.relu4(self.conv4(x1))
        x3 = self.relu5(self.conv5(x2))
        x4 = self.relu6(self.conv6(x3))
        x5 = self.relu7(self.conv7(x4))
        n = self.conv8(x5)
        r_restore = r + n  # 最后一层卷积输出 n，然后将 n 加到 r 上，得到最终的恢复图像 r_restore。
        r_restore = torch.clamp(r_restore, 0.0, 1.0)
        return r_restore


class SELayer(nn.Module):
    """Uretinex原始SE层，结构未修改"""

    def __init__(self, channel, reduction=16):  # reduction: 降维因子，用于减少通道数

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




class IterativeUretinex(nn.Module):
    def __init__(self, opts, unfolding_round=3):
        super().__init__()
        self.opts = opts
        self.unfolding_round = unfolding_round
        self.init_decom = Decom()
        self.P_solver = P()
        self.Q_solver = Q()
        self.r_restore_net = HalfDnCNNSE(opts)

    def forward(self, input_img):
        # 修复：正确初始化变量
        R, L = self.init_decom(input_img)  # R对应P，L对应Q
        P, Q = R, L  # 保持变量命名一致性

        # 修复迭代逻辑
        for t in range(self.unfolding_round):
            if t > 0:
                w_p = self.opts.gamma + self.opts.Roffset * t
                w_q = self.opts.lamda + self.opts.Loffset * t
                # 修复：使用正确的变量顺序
                P = self.P_solver(I=input_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q_solver(I=input_img, P=P, L=L, lamda=w_q)

            # 修复：反射层恢复应该在每次迭代都执行
            R = self.r_restore_net(r=P, l=Q)
            L = Q

        # 修复：确保输出在合理范围内
        R = torch.clamp(R, 0.0, 1.0)
        L = torch.clamp(L, 0.0, 1.0)

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

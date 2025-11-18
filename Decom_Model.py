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
import lpips
from sys import platform
from string import ascii_letters
import matplotlib.pyplot as plt

def get_device(input):
    """Derive device from input tensor"""
    return input.device

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
        denominator = gamma + Q * Q  # 增加epsilon防止除零
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
        denominator = (PR * PR + PG * PG + PB * PB) + lamda
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
        #读入init_  在评估里面

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


class Illumination_Alone(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.conv1 = get_conv2d_layer(in_c=1, out_c=32, k=5, s=1, p=2)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv5 = get_conv2d_layer(in_c=32, out_c=1, k=1, s=1, p=0)

        self.leaky_relu_1 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu_2 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu_3 = nn.LeakyReLU(0.2, inplace=True)
        self.leaky_relu_4 = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, l):
        x = l
        x1 = self.leaky_relu_1(self.conv1(x))
        x2 = self.leaky_relu_2(self.conv2(x1))
        x3 = self.leaky_relu_3(self.conv3(x2))
        x4 = self.leaky_relu_4(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        return x5

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
        return r_restore


class IterativeUretinex(nn.Module):
    def __init__(self, opts, unfolding_round=3):
        super().__init__()
        self.opts = opts
        self.unfolding_round = unfolding_round
        self.init_decom = Decom()
        self.P_solver = P()
        self.Q_solver = Q()
        self.r_restore_net = HalfDnCNNSE(opts)
        self.l_restore_net = Illumination_Alone(opts)


    def load_init_decom(self):
        checkpoint_Decom_low = torch.load('./init_low.pth')
        self.init_decom.load_state_dict(checkpoint_Decom_low['state_dict']['model_R'])
        # to freeze the params of Decomposition Model
        for param in self.init_decom.parameters():
            param.requires_grad = False
        return self.init_decom


    def forward(self, input_img):
        # 修复：正确初始化变量
        # 确保输入在合理范围内
        # input_img = torch.clamp(input_img, 0.0, 1.0)

        # 添加输入检查
        # print(f"分解模块输入 - input_img 范围: [{input_img.min():.4f}, {input_img.max():.4f}]")
         # R对应P，L对应Q
        # 修复迭代逻辑
        for t in range(self.unfolding_round):
            if t==0:
                P, Q = self.init_decom(input_img)
            else:
                w_p = self.opts.gamma + self.opts.Roffset * t
                w_q = self.opts.lamda + self.opts.Loffset * t
                # 修复：使用正确的变量顺序
                P = self.P_solver(I=input_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q_solver(I=input_img, P=P, L=L, lamda=w_q)
                # print(f"迭代{t}后 - P范围: {P.min():.4f}~{P.max():.4f}, Q范围: {Q.min():.4f}~{Q.max():.4f}")

            # # 主动约束，而非检测到NaN后替换
            # P = torch.clamp(P, 0.0, 1.0)  # 缩小范围，避免极端值
            # Q = torch.clamp(Q, 0.0, 1.0)
            # 修复：反射层恢复应该在每次迭代都执行
            R = self.r_restore_net(r=P, l=Q)
            L = self.l_restore_net(l=Q)

            # print(f"反射层恢复后 - R范围: {R.min():.4f}~{R.max():.4f}, L范围: {L.min():.4f}~{L.max():.4f}")
            # 修复：确保输出在合理范围内
            R = torch.clamp(R, 0.0, 1.0)
            L = torch.clamp(L, 0.0, 1.0)

        return R, L


class HyPaNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=5, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus())

    def forward(self, x):
        x = self.mlp(x)
        x = torch.sigmoid(x)
        return x*0.1 + 1


class HyParNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1):
        super(HyParNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_nc),
            nn.Sigmoid())

    def forward(self, x):
        x = self.mlp(x)
        x = torch.sigmoid(x)
        return x + 0.01

def grad(x):
    grd_x = torch.diff(x, dim=3)
    grd_x = F.pad(grd_x, (0, 1, 0, 0))
    grd_y = torch.diff(x, dim=2)
    grd_y = F.pad(grd_y, (0, 0, 0, 1))
    return grd_x, grd_y


def fftn(t, batch, channel, row, col, dim):
    y = torch.fft.fft(t, col, dim=dim)
    y = y.expand(batch, channel, col, row)
    return y


def fftnt(t, batch, channel, row, col, dim):
    y = torch.fft.fft(t, col, dim=dim)
    y = y.expand(batch, channel, row, col)
    return y


def Dive(x, y):
    x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
    x_e = (x[:, :, :, -1] - x[:, :, :, 0]).unsqueeze(3)
    x_diff = torch.cat((x_e, x_diff), 3)
    y_diff = y[:, :, :-1, :] - y[:, :, 1:, :]
    y_e = (y[:, :, -1, :] - y[:, :, 0, :]).unsqueeze(2)
    y_diff = torch.cat((y_e, y_diff), 2)
    return y_diff + x_diff


class DecomNet_RTV(nn.Module):
    def __init__(self, in_ch, k1=10):
        super(DecomNet_RTV, self).__init__()
        # Do not move submodules to device here; let the parent module's .to(device) handle it once
        self.hypar = HyPaNet(1, k1)
        self.par1 = HyParNet(in_ch)
        self.par2 = HyParNet(in_ch)
        self.par3 = HyParNet(in_ch)
        self.k1 = k1

    def forward(self, O):
        batch, ch, row, col = O.shape

        # Derive device and dtype from input tensor O
        dev = O.device
        dt = O.dtype

        I = O.clone()
        d1 = torch.zeros_like(I)
        d2 = torch.zeros_like(I)
        y1 = torch.zeros_like(I)
        y2 = torch.zeros_like(I)


        mu1 = torch.tensor(1.0, device=dev, dtype=dt).view(1, 1, 1, 1)
        mu = self.hypar(mu1)
        alpha = 0.001 * self.par1(O)
        px = self.par2(O)
        py = self.par3(O)

        eps = torch.tensor(0.001, device=dev, dtype=dt)
        Dx = ([1.0], [-1.0])
        Dy = ([1.0, -1.0])
        Dx = torch.tensor(Dx, device=dev, dtype=dt).unsqueeze(0).unsqueeze(0)
        Dy = torch.tensor(Dy, device=dev, dtype=dt).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        eigDtD = torch.pow(torch.abs(fftn(Dx, batch, ch, col, row, 2)), 2) + torch.pow(
            torch.abs(fftnt(Dy, batch, ch, row, col, 3)), 2)

        I_list = []

        for i in range(self.k1):
            rhs_I = O - mu[0, i, 0, 0] * Dive(d1 + y1, d2 + y2)
            lhs_I = 1 + mu[0, i, 0, 0] * eigDtD
            I = torch.real(torch.fft.ifftn(torch.fft.fftn(rhs_I) / lhs_I))
            DxI, DyI = grad(I)

            wtbx = torch.max(torch.pow(torch.abs(DxI), (2 - px)), eps) ** (-1)
            wtby = torch.max(torch.pow(torch.abs(DyI), (2 - py)), eps) ** (-1)
            wx = wtbx
            wy = wtby
            d1 = mu[0, i, 0, 0] * (DxI - y1) / (2 * alpha * wx + mu[0, i, 0, 0] + 1e-3)
            d2 = mu[0, i, 0, 0] * (DyI - y2) / (2 * alpha * wy + mu[0, i, 0, 0] + 1e-3)
            y1 = y1 + (d1 - DxI)
            y2 = y2 + (d2 - DyI)
            I_list.append(I)
            # wxx = torch.sqrt(alpha * wx)
            # wyy = torch.sqrt(alpha * wy)

        return I


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

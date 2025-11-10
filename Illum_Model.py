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
# 导入权重初始化模块
import torch.nn.init as init


class enhance_net_nopool(nn.Module):
    """Zero-DCE原始网络，结构+前向未修改"""

    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32  # 原始通道数，未修改
        # 原始7层卷积，未修改
        self.e_conv1 = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)

        # --- 关键层：这是输出曲线参数的最后一层 ---
        self.e_conv7 = nn.Conv2d(number_f * 2, 8, 3, 1, 1, bias=True)  # 输出8通道（8*1）

        # --- 新增：调用自定义的权重初始化方法 ---
        self._initialize_weights()

    # --- 新增：自定义权重初始化方法 ---
    def _initialize_weights(self):
        """
        对网络的权重进行自定义初始化。
        - 对非最后一层的卷积层使用标准的 Kaiming 初始化。
        - 对最后一层 (e_conv7) 使用零初始化，以确保训练从恒等变换开始。
        """
        print("--- Applying custom weight initialization ---")
        # 遍历网络的所有模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对非最后一层的卷积层使用标准的 Kaiming 初始化
                if m is not self.e_conv7:
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

        # --- 关键步骤：对最后一层 (e_conv7) 进行零初始化 ---
        print("Initializing the final layer (e_conv7) to all zeros.")
        init.constant_(self.e_conv7.weight, 0)
        init.constant_(self.e_conv7.bias, 0)

    def forward(self, x):
        """Zero-DCE原始8轮迭代增强逻辑，未修改"""
        # 注意：在您的Retinex框架中，输入的x (low_L) 已经是 [0,1] 范围，
        # 但为了安全，保留clamp操作。
        x = torch.clamp(x, 0.0, 1.0)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        # 经过零初始化的e_conv7后，x_r在训练开始时将为0
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # 原始8轮迭代增强，未修改
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 1, dim=1)

        # 当 r1, r2, ... r8 都为0时，下面的所有操作都相当于 x = x + 0，
        # 最终的 enhance_image 将等于输入的 x。
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)

        # 为了数值稳定性，最好在最后再做一次clamp
        enhance_image = torch.clamp(enhance_image, 0.0, 1.0)

        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)

        # 原始返回值：中间增强图+最终增强图+增强系数，未修改
        # 注意：在您的Retinex框架中，您可能只需要 enhance_image
        return enhance_image, r


if __name__ == '__main__':
    # 将设备设置为cuda如果可用，否则为cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建一个随机输入张量并移动到设备
    x = torch.rand(1, 1, 128, 128).to(device)

    # 实例化网络并移动到设备
    print("Creating the network...")
    net = enhance_net_nopool().to(device)
    print("\nNetwork created. Forward pass test:")

    # 前向传播
    enhance_image_1, enhance_image, r = net(x)

    # 打印输出形状
    print("\nOutput shapes:")
    print(f"Intermediate enhanced image shape: {enhance_image_1.shape}")
    print(f"Final enhanced image shape: {enhance_image.shape}")
    print(f"Curve parameters shape: {r.shape}")

    # --- 验证初始化是否成功 ---
    print("\nVerifying initialization:")
    # 检查最终增强图像是否与输入完全相同
    is_identity = torch.allclose(enhance_image, x)
    print(f"Is the output identical to the input? -> {is_identity}")
    # 检查曲线参数是否全为0
    is_r_zero = (r.abs().max().item() == 0)
    print(f"Are all curve parameters zero? -> {is_r_zero}")

    if is_identity and is_r_zero:
        print("\nSUCCESS: The network is correctly initialized for identity transformation.")
    else:
        print("\nWARNING: The network is NOT correctly initialized.")


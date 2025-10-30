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
# from Decom_Model import IterativeUretinex, DecomNet_RTV # 注释掉旧的
from model_Decom_fft import DecomNet_Fourier  # 假设正确的类名叫这个

from Illum_Model import *
# from Denoise_Model import *
import torch.fft


def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    # 对出界值的处理
    h = h % 1
    s = torch.clamp(s, 0, 1)
    v = torch.clamp(v, 0, 1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    hi0 = hi == 0
    hi1 = hi == 1
    hi2 = hi == 2
    hi3 = hi == 3
    hi4 = hi == 4
    hi5 = hi == 5

    r[hi0] = v[hi0]
    g[hi0] = t[hi0]
    b[hi0] = p[hi0]

    r[hi1] = q[hi1]
    g[hi1] = v[hi1]
    b[hi1] = p[hi1]

    r[hi2] = p[hi2]
    g[hi2] = v[hi2]
    b[hi2] = t[hi2]

    r[hi3] = p[hi3]
    g[hi3] = q[hi3]
    b[hi3] = v[hi3]

    r[hi4] = t[hi4]
    g[hi4] = p[hi4]
    b[hi4] = v[hi4]

    r[hi5] = v[hi5]
    g[hi5] = p[hi5]
    b[hi5] = q[hi5]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)

    return rgb


def rgb_to_hsv(img):
    eps = 1e-8
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]
    value = img.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value], dim=1)
    return hsv

# 在 Model.py 文件中，找到 class LLIE(nn.Module): 并用下面的代码块完整替换

class LLIE(nn.Module):
    def __init__(self, args=None): # 将args设为可选，方便调试
        super().__init__()
        # 确保这里导入的是正确的组件
        # DecomNet_Fourier 来自 model_Decom_fft.py
        # enhance_net_nopool 来自 Illum_Model.py
        self.decom_net = DecomNet_Fourier()
        self.enhance_net = enhance_net_nopool()

    def forward(self, x_low, decom_only=False, enhance_only=False, R_low=None, L_low=None):
        """
        模型的前向传播函数。
        """
        # 模式1: 只执行分解网络 (用于训练第二阶段的第一步)
        if decom_only:
            low_R, low_L, A, T, S, S_low_zy, T_low_zy = self.decom_net(x_low)
            return low_R, low_L, A, T, S, S_low_zy, T_low_zy
        # 模式2: 只执行增强网络 (用于训练第二阶段的第二步)
        if enhance_only:
            if L_low is None or R_low is None:
                raise ValueError("R_low and L_low must be provided when enhance_only is True.")

            # EnhanceNet 只返回增强后的L
            enhance_L = self.enhance_net(L_low)
            # 在这里完成图像重建
            enhance_img = R_low * enhance_L
            return enhance_L, enhance_img
        # 模式3: 完整流程 (用于测试/可视化)
        # 首先分解
        low_R, low_L, A, T, S, _, _ = self.decom_net(x_low)

        # 然后增强光照图
        enhance_L = self.enhance_net(low_L)
        # 最后重建图像
        enhance_img = low_R * enhance_L
        # 返回所有需要的结果
        return low_R, low_L, A, T, S, enhance_L, enhance_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Low Light Enhancement")


    # 为了让 __main__ 能运行，需要一个假的 args 对象
    class DummyArgs:
        pass


    args = DummyArgs()

    x = torch.rand(1, 3, 128, 128).cuda()
    net = LLIE(args).cuda()
    low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img = net(x)
    print(enhance_img.shape)


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
from Decom_Model import IterativeUretinex, DecomNet_RTV
from Illum_Model import *
# from Denoise_Model import *


def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    # 对出界值的处理
    h = h % 1
    s = torch.clamp(s, 1e-3, 1)
    v = torch.clamp(v, 1e-3, 1)

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
    eps = 1e-6
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


class LLIE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.decom_net = DecomNet_RTV(in_ch=1, k1=10)
        self.enhance_net = enhance_net_nopool()

    def gamma_correction(self, low_img):
        # 增加输入检查
        # low_img = torch.clamp(low_img, 0, 1.0)

        hsv_img = rgb_to_hsv(low_img)
        gamma = random.uniform(3.5, 3.7)

        V = hsv_img[:, 2] ** (1 / gamma)
        V = torch.clamp(V, 1e-3, 10)
        hsv_img[:, 2] = V

        x_gamma = hsv_to_rgb(hsv_img)
        x_gamma = torch.clamp(x_gamma, 1e-3, 20)

        return x_gamma

    def forward(self, low_img):
        # --- 1. 原始分解 (与您原来一致) ---
        low_L0 = torch.max(low_img, dim=1, keepdim=True)[0]
        low_L = self.decom_net(low_L0)
        low_L = torch.clamp(low_L, 0.1, 20.0)
        low_R = low_img / (low_L + 1e-3)
        low_R = torch.clamp(low_R, 0.1, 20.0)

        # --- 2. Gamma参考分支 (与您原来一致, 用于计算损失) ---
        x_gamma = self.gamma_correction(low_img)
        gamma_L0 = torch.max(x_gamma, dim=1, keepdim=True)[0]
        gamma_L = self.decom_net(gamma_L0)
        gamma_L = torch.clamp(gamma_L, 0.01, 10.0)
        gamma_R = x_gamma / (gamma_L + 1e-3)
        gamma_R = torch.clamp(gamma_R, 0.01, 10.0)

        # --- [核心修改 1: 为增强网络减负，提供更好的起点] ---
        # 我们将分解出的 low_L 进行一次Gamma校正，
        # 让 enhance_net 的任务从“大幅提亮”变为“精细调整”。
        # 这个操作是可微分的，不影响端到端训练。
        gamma_val_for_L = 2.2  # 这是一个常用的、效果很好的超参数
        low_L_gamma_input = torch.pow(low_L + 1e-6, 1.0 / gamma_val_for_L)

        # 将预处理过的 low_L_gamma_input 送入增强网络
        enhance_L, _ = self.enhance_net(low_L_gamma_input)

        # --- [核心修改 2: 强制模型学习提亮，提供驱动力] ---
        # 在最终重建时，我们对原始反射图 low_R 进行缩放。
        # 为了让最终的 enhance_img 在损失函数（如L1, SSIM）的监督下
        # 仍然能匹配目标亮度，模型将被迫学习输出一个更亮的 enhance_L 来补偿。
        reflectance_factor = 1

        # 使用增强后的 enhance_L 和 缩放后的 low_R 进行重建
        enhance_img = enhance_L * (low_R * reflectance_factor)
        enhance_img = torch.clamp(enhance_img, 0.01, 10.0)  # 最终输出建议clamp到[0,1]

        return low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img


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
    net=LLIE(args).cuda()
    low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img=net(x)
    #print(enhance_img.shape)





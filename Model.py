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
from Decom_Model import IterativeUretinex
from Illum_Model import *
from Denoise_Model import *


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



class LLIE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.decom_net = IterativeUretinex(args)
        self.enhance_net = enhance_net_nopool()
        self.denoise_net = SRResnet(input_channels=3, output_channels=3)

    def gamma_correction(self, low_img):
        # 增加输入检查
        low_img = torch.clamp(low_img, 1e-4, 1.0)

        hsv_img = rgb_to_hsv(low_img)
        gamma = random.uniform(1.2, 1.8)

        V = hsv_img[:, 2] ** (1 / gamma)
        V = torch.clamp(V, 1e-4, 1)
        hsv_img[:, 2] = V

        x_gamma = hsv_to_rgb(hsv_img)
        x_gamma = torch.clamp(x_gamma, 0, 1)

        return x_gamma

    def forward(self, low_img, gt_img):
        low_R, low_L = self.decom_net(low_img)
        gt_R, gt_L = self.decom_net(gt_img)
        x_gamma = self.gamma_correction(low_img)
        gamma_R, gamma_L = self.decom_net(x_gamma)
        # print(f"low_L 通道数: {low_L.shape[1]}")  # 应输出 1
        # print(f"enhance_net 第一层输入通道数: {self.enhance_net.e_conv1.in_channels}")  # 应输出 1
        # L_3ch = low_L.repeat(1, 3, 1, 1)
        enhance_L1, enhance_L, _ = self.enhance_net(low_L)
        enhance_L1 = torch.clamp(enhance_L1, 0.0, 1.0)
        enhance_L = torch.clamp(enhance_L, 0.0, 1.0)
        denoise_R = self.denoise_net(low_R)
        enhance_img = enhance_L * denoise_R
        enhance_img = torch.clamp(enhance_img, 0.0, 1.0)###+约束

        return low_R, low_L, gamma_R, gamma_L, enhance_L1, enhance_L, denoise_R, enhance_img


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
    low_R, low_L, gt_R, gt_L, enhance_L1, enhance_L, denoise_R, enhance_img=net(x, x)
    print(enhance_img.shape)





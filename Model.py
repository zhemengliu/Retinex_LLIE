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
from Decom_Model import  IterativeUretinex,DecomNet_RTV
from Illum_Model import *
# from Denoise_Model import *


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
    def __init__(self):
        super().__init__()
        # self.decom_net = IterativeUretinex(args)
        self.decom_net = DecomNet_RTV(in_ch=1, k1=10)
    #     self.decom_net = DecomNet(
    #     unfolding_rounds=5,
    #     init_channels=64,
    #     gamma=0.05,
    #     lamda=0.05,
    #     gamma_offset=0.02,
    #     lamda_offset=0.02
    # ).cuda()
        self.enhance_net = enhance_net_nopool()

        # # 添加输出监控
        # self.forward_count = 0
        # self.denoise_net = SRResnet(input_channels=3, output_channels=3)

    # def gamma_correction(self, low_img):
    #     # 增加输入检查
    #     # low_img = torch.clamp(low_img, 0, 1.0)
    #
    #     hsv_img = rgb_to_hsv(low_img)
    #     gamma = random.uniform(2.0, 2.5)
    #
    #     V = hsv_img[:, 2] ** (1 / gamma)
    #     V = torch.clamp(V, 0, 1)
    #     hsv_img[:, 2] = V
    #
    #     x_gamma = hsv_to_rgb(hsv_img)
    #     x_gamma = torch.clamp(x_gamma, 0, 1)
    #
    #     return x_gamma
    def gamma_correction(self, low_img, gamma=2.5):
        """使用固定gamma值确保训练稳定性"""
        low_img = torch.clamp(low_img, 0.0, 1.0)
        x_gamma = torch.pow(low_img, 1 / gamma)
        x_gamma = torch.clamp(x_gamma, 0.0, 1.0)
        return x_gamma

    def forward(self, low_img):
        # 确保输入在合理范围内
        low_img = torch.clamp(low_img, 0.0, 1.0)
        # self.forward_count += 1

        low_L0 = torch.max(low_img, dim=1, keepdim=True)[0]  # 保留批次和通道维度
        low_L0 = torch.clamp(low_L0,0.0,1.0)
        # print(f"[前向{self.forward_count}] low_L0范围: [{low_L0.min():.4f}, {low_L0.max():.4f}]")
        low_L = self.decom_net(low_L0)
        low_L = torch.clamp(low_L, 0.0, 1.0)
        print(f"low_L范围: [{low_L.min():.4f}, {low_L.max():.4f}]")
        # if low_L.std() < 1e-6:
        #     print(f"警告: low_L几乎为常数: {low_L.mean().item():.6f} ± {low_L.std().item():.6f}")
        low_R = low_img / (low_L + 1e-3)
        low_R = torch.clamp(low_R, 0.0,1.0)

        x_gamma = self.gamma_correction(low_img)
        x_gamma = torch.clamp(x_gamma,0.0,1.0)
        gamma_L0 = torch.max(x_gamma, dim=1, keepdim=True)[0]  # 保留批次和通道维度
        gamma_L0 = torch.clamp(gamma_L0,0.0,1.0)
        gamma_L = self.decom_net(gamma_L0)
        gamma_L = torch.clamp(gamma_L,0.0,1.0)
        gamma_R = x_gamma / (gamma_L + 1e-3)
        gamma_R = torch.clamp(gamma_R,0.0,1.0)

        # low_img_hsv = rgb_to_hsv(low_img)
        # low_img_v = low_img_hsv[:, 2].unsqueeze(1)
        # x_gamma = self.gamma_correction(low_img)
        # x_gamma_hsv = rgb_to_hsv(x_gamma)
        # x_gamma_v = x_gamma_hsv[:, 2].unsqueeze(1)
        # # print(f"Gamma校正后 - x_gamma范围: [{x_gamma.min():.4f}, {x_gamma.max():.4f}]")

        # # # 检测x_gamma
        # # if torch.isnan(x_gamma).any():
        # #     print("警告：gamma校正后出现NaN")
        # #     x_gamma = torch.clamp(x_gamma, 0, 1.0)
        #
        # low_R, low_L = self.decom_net(low_img_v)
        # # print(f"低光分解 - low_R范围: [{low_R.min():.4f}, {low_R.max():.4f}], low_L范围: [{low_L.min():.4f}, {low_L.max():.4f}]")
        # gamma_R, gamma_L = self.decom_net(x_gamma_v)
        # # print(f"Gamma分解 - gamma_R范围: [{gamma_R.min():.4f}, {gamma_R.max():.4f}], gamma_L范围: [{gamma_L.min():.4f}, {gamma_L.max():.4f}]")
        #
        # low_R = torch.cat([low_img_hsv[:,0].unsqueeze(1), low_img_hsv[:,1].unsqueeze(1), low_R], dim=1)
        # gamma_R = torch.cat([x_gamma_hsv[:, 0].unsqueeze(1), x_gamma_hsv[:, 1].unsqueeze(1), gamma_R], dim=1)
        #
        # low_R = hsv_to_rgb(low_R)
        # gamma_R = hsv_to_rgb(gamma_R)

        # # 检测分解输出
        # for name, tensor in [("low_R", low_R), ("low_L", low_L), ("gamma_R", gamma_R), ("gamma_L", gamma_L)]:
        #     if torch.isnan(tensor).any():
        #         print(f"警告：{name} 出现NaN，已替换为默认值")
        #         tensor = torch.clamp(tensor, 0, 1.0)
        #
        # # 添加数值检查
        # if torch.isnan(low_R).any() or torch.isnan(low_L).any():
        #     print("警告：分解输出出现NaN")
        #     # 使用备用值
        #     low_R = torch.ones_like(low_R) * 0.5
        #     low_L = torch.ones_like(low_L) * 0.5
        # print(f"[调试] low_L 范围: {torch.min(low_L):.4f} ~ {torch.max(low_L):.4f}")
        # print(f"[调试] low_L 均值: {torch.mean(low_L):.4f}")
        # print(f"[调试] 增强前光照范围: {torch.min(low_L):.4f} ~ {torch.max(low_L):.4f}")

        # print(f"low_L 通道数: {low_L.shape[1]}")  # 应输出 1
        # print(f"enhance_net 第一层输入通道数: {self.enhance_net.e_conv1.in_channels}")  # 应输出 1
        # L_3ch = low_L.repeat(1, 3, 1, 1)

        # 照明增强
        # print(f"照明增强输入 low_L 形状: {low_L.shape}, 范围: [{low_L.min():.4f}, {low_L.max():.4f}]")
        enhance_L, _ = self.enhance_net(low_L)
        enhance_L = torch.clamp(enhance_L,0.0,1.0)
        # enhance_L = torch.clamp(enhance_L, 0.0, 1.0)
        # print(f"照明增强输出 - enhance_L范围: [{enhance_L.min():.4f}, {enhance_L.max():.4f}]")
        # print(f"[调试] 增强后光照范围: {torch.min(enhance_L):.4f} ~ {torch.max(enhance_L):.4f}")
        # print(f"[调试] 增强后光照均值: {torch.mean(enhance_L):.4f}")

        # 检测照明增强输出
        # if torch.isnan(enhance_L).any():
        #     print("警告：enhance_L 出现NaN")
        #     enhance_L = torch.clamp(enhance_L, 0, 1.0)

        # denoise_R = self.denoise_net(low_R)
        enhance_img = enhance_L * low_R
        enhance_img = torch.clamp(enhance_img, 0.0, 1.0)###+约束

        # print(f"最终增强图像 - enhance_img范围: [{enhance_img.min():.4f}, {enhance_img.max():.4f}]")
        # print("=" * 60)

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
    x = torch.rand(1,1,128,128).cuda()
    net=LLIE().cuda()
    low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img=net(x)
    print(enhance_img.shape)





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
import math


def hsv_to_rgb(hsv_image: torch.Tensor) -> torch.Tensor:
    """
    将一个批次的HSV图像张量转换为RGB。
    输入张量的值域应为 [0, 1]。

    参数:
        hsv_image (torch.Tensor): 输入的HSV图像，shape为 (b, 3, h, w)。
                                  通道顺序为 H, S, V。
    返回:
        torch.Tensor: 输出的RGB图像，shape为 (b, 3, h, w)。
    """
    h, s, v = hsv_image[:, 0, :, :], hsv_image[:, 1, :, :], hsv_image[:, 2, :, :]

    h_i = (h * 6).floor()
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    h_i = h_i % 6

    rgb = torch.zeros_like(hsv_image)

    # 根据h_i的值填充对应的RGB通道
    mask = (h_i == 0).unsqueeze(1)
    rgb += mask * torch.stack([v, t, p], dim=1)

    mask = (h_i == 1).unsqueeze(1)
    rgb += mask * torch.stack([q, v, p], dim=1)

    mask = (h_i == 2).unsqueeze(1)
    rgb += mask * torch.stack([p, v, t], dim=1)

    mask = (h_i == 3).unsqueeze(1)
    rgb += mask * torch.stack([p, q, v], dim=1)

    mask = (h_i == 4).unsqueeze(1)
    rgb += mask * torch.stack([t, p, v], dim=1)

    mask = (h_i == 5).unsqueeze(1)
    rgb += mask * torch.stack([v, p, q], dim=1)

    return rgb

class L_spa(nn.Module):#ok
    """原始空间一致性损失，未修改"""

    def __init__(self):
        super(L_spa, self).__init__()
        # 原始梯度核，未修改
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        """原始损失计算逻辑，未修改"""
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        # 原始权重计算，未修改

        # 原始梯度差计算，未修改
        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
        D_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)
        # 原始损失求和，未修改
        D_left = torch.abs(D_org_left - D_enhance_left)
        D_right = torch.abs(D_org_right - D_enhance_right)
        D_up = torch.abs(D_org_up - D_enhance_up)
        D_down = torch.abs(D_org_down - D_enhance_down)
        E = torch.mean(D_left + D_right + D_up + D_down)
        return E


class L_exp(nn.Module):
    """原始曝光控制损失，未修改"""

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        """原始损失计算逻辑，未修改"""
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.abs(mean - torch.FloatTensor([self.mean_val]).cuda()))
        d = 500 * d
        return d


class L_hue_mean_diff(nn.Module):
    """
    HSV空间下，衡量增强前后图像色相均值差异的损失函数。

    它通过以下步骤实现：
    1. 使用数学上正确的“向量平均法”计算原始图像和增强图像的平均色相。
    2. 计算两个平均色相之间的最短环形距离（circular distance）。
    3. 将这个距离作为损失，惩罚增强过程中的色偏漂移。
    """

    def __init__(self):
        super(L_hue_mean_diff, self).__init__()

    def _calculate_circular_mean_hue(self, hsv_img):
        """
        辅助函数：使用向量法计算一个batch图像的平均色相。
        """
        # 1. 提取 H (色相) 通道，假设 H 是通道 0
        # 输入的 H 通道值域为 [0, 1]
        hue_channel = hsv_img[:, 0, :, :]
        # 2. 将 H 值从 [0, 1] 转换为弧度 [0, 2*pi]
        hue_rad = hue_channel * 2 * math.pi
        # 3. 将每个像素的色相转换为单位圆上的 (x, y) 坐标
        x_coords = torch.cos(hue_rad)
        y_coords = torch.sin(hue_rad)
        # 4. 计算 x 和 y 坐标在空间维度上的平均值
        # 得到每个图像的平均向量 (mean_x, mean_y)
        mean_x = torch.mean(x_coords, dim=[1, 2])
        mean_y = torch.mean(y_coords, dim=[1, 2])
        # 5. 将平均向量 (mean_x, mean_y) 转换回弧度
        # atan2 能正确处理所有象限，返回 [-pi, pi]
        mean_hue_rad = torch.atan2(mean_y, mean_x)
        # 6. 将结果从 [-pi, pi] 转换回 [0, 1] 的范围，方便后续计算
        # (mean_hue_rad + math.pi) -> [0, 2*pi]
        # / (2 * math.pi) -> [0, 1]
        mean_hue_norm = (mean_hue_rad + math.pi) / (2 * math.pi)

        return mean_hue_norm

    def forward(self, original_hsv_img, enhanced_hsv_img):
        """
        计算原始图像和增强图像之间的平均色相差异损失。
        """
        # 分别计算原始图像和增强图像的平均色相
        mean_hue_orig = self._calculate_circular_mean_hue(original_hsv_img)
        mean_hue_enh = self._calculate_circular_mean_hue(enhanced_hsv_img)
        # 计算两个平均色相之间的差值
        # 结果在 [-1, 1] 之间
        diff = mean_hue_orig - mean_hue_enh

        # 计算最短环形距离 (circular distance)
        # 例如，0.9 和 0.1 的差是 0.8，但它们的最短距离是 0.2 (通过 1.0/0.0 连接)
        # 距离 = min(|diff|, 1 - |diff|)
        abs_diff = torch.abs(diff)
        circular_diff = torch.min(abs_diff, 1.0 - abs_diff)
        # 对 batch 中的所有样本取平均，得到最终的损失值
        loss = torch.mean(circular_diff)

        return loss


class L_TV(nn.Module):
    """原始照明平滑度损失，未修改"""

    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        """原始损失计算逻辑，未修改"""
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])).sum()
        w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1])).sum()
        LLL = self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w)
        LLL = 10 * LLL
        return LLL


class L_color_rgb(nn.Module):
    """
    RGB空间的色彩恒常性损失函数。
    它接收HSV图像，内部转换为RGB，然后惩罚R,G,B三通道均值之间的差异。
    目的是消除图像的整体色偏，使其色彩更中性。
    """

    def __init__(self):
        super(L_color_rgb, self).__init__()

    def forward(self, hsv_image):
        """
        计算色彩恒常性损失。
        参数:
            hsv_image (torch.Tensor): 输入的HSV图像，shape为 (b, 3, h, w)。
        """
        # 1. 将输入的HSV图像转换为RGB
        rgb_image = hsv_to_rgb(hsv_image)

        # 2. 计算每个通道在空间维度上的均值
        # 输入 shape: (b, 3, h, w) -> 输出 shape: (b, 3)
        channel_means = torch.mean(rgb_image, dim=[2, 3])

        # 3. 提取R, G, B的均值
        mean_r = channel_means[:, 0]
        mean_g = channel_means[:, 1]
        mean_b = channel_means[:, 2]

        # 4. 计算两两通道均值之间的L1距离（绝对值差）
        loss_rg = torch.abs(mean_r - mean_g)
        loss_gb = torch.abs(mean_g - mean_b)
        loss_br = torch.abs(mean_b - mean_r)

        # 5. 将三个距离相加，得到每个样本的色彩损失
        color_loss_per_sample = loss_rg + loss_gb + loss_br

        # 6. 对batch中的所有样本取平均，得到最终的标量损失值
        loss = torch.mean(color_loss_per_sample)

        return loss



def compute_total_loss(x_low, low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img):
    """总损失：Uretinex分解损失+Noise2noise去噪损失+Zero-DCE照明损失+重建损失"""
    # 添加数值稳定性
    # epsilon = 1e-8
    # 1. Uretinex分解损失（原始逻辑）
    # 分解保真损失：x_low ≈ R*L
    decom_recon_loss = nn.MSELoss()(low_R * low_L, x_low) + nn.MSELoss()(gamma_R * gamma_L, x_gamma)
    decom_recon_loss = 100 * decom_recon_loss
    r_gradient_loss  = L_spa()(low_R, x_low) + L_spa()(gamma_R, x_gamma)
    r_gradient_loss = torch.clamp(r_gradient_loss, 0.1, 10.0) # 防止梯度爆炸
    decom_consist_loss = nn.MSELoss()(low_R, gamma_R)
    decom_tv_loss = L_TV()(low_L)
    low_L0 = torch.max(x_low, dim=1, keepdim=True)[0]  # 保留批次和通道维度
    gamma_L0 = torch.max(x_gamma, dim=1, keepdim=True)[0]
    low_L0 = torch.clamp(low_L0, 0.1, 10.0)  # 约束范围
    gamma_L0 = torch.clamp(gamma_L0, 0.1, 10.0)

    L_con = nn.MSELoss()(low_L0, low_L) + nn.MSELoss()(gamma_L0, gamma_L)
    L_con = 1000 * L_con
    L_spa_ex = L_spa()(x_low,enhance_img)
    Color = torch.abs(L_hue_mean_diff()(enhance_img,x_low))
    AAA = nn.MSELoss()(enhance_L,gamma_L)
    #print(AAA)
    color_constancy_loss_func = L_color_rgb()
    # 对分解出的两个反射率图计算损失
    loss_color_constancy = color_constancy_loss_func(low_R) + color_constancy_loss_func(gamma_R)

    total_decom_loss = 2*decom_recon_loss + 10 * decom_consist_loss + 0.01 * decom_tv_loss \
                       + 0.1 * r_gradient_loss + 1 * L_con + 2 * L_spa_ex + 5 * Color \
                        + 3 * loss_color_constancy

    #print(f'decom_recon_loss:{decom_recon_loss:.4f}, decom_consist_loss: {1*decom_consist_loss:.4f}, decom_tv_loss:{0.01 * decom_tv_loss:.4f}, L_con:{0.1 * L_con:.4f}, r_gradient_loss:{r_gradient_loss:.4f}')


    exp_loss = L_exp(patch_size=16, mean_val=0.7)(enhance_L)


    total_illum_loss = exp_loss + 50 * AAA #+ 0.01 * tv_loss #+ exp_loss_L+ 0.3 * color_loss

    # 总损失（关键修改：确保所有损失项都是标量，并最终求和为标量）
    total_loss = total_decom_loss + 0.1*total_illum_loss

    # 最终数值检查
    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
        print("警告：检测到NaN/Inf损失，使用备用损失")
        # 使用简单的重建损失作为备用
        total_loss = nn.MSELoss()(enhance_img, x_low)
        total_decom_loss = total_loss * 0.5
        total_illum_loss = total_loss * 0.5

    # 强制确保 total_loss 是标量（添加均值操作，针对可能的维度残留）
    if total_loss.dim() > 0:
        total_loss = total_loss.mean()

    return total_loss, total_decom_loss, total_illum_loss


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


def rgb_to_hsv(rgb_image: torch.Tensor) -> torch.Tensor:
    """
    将一个批次的RGB图像张量转换为HSV。输入值域应为[0,1]。
    参数: rgb_image (b,3,h,w)
    返回: hsv_image (b,3,h,w) with H,S,V in [0,1]
    """
    r, g, b = rgb_image[:, 0, :, :], rgb_image[:, 1, :, :], rgb_image[:, 2, :, :]
    maxc, _ = torch.max(rgb_image, dim=1)
    minc, _ = torch.min(rgb_image, dim=1)
    v = maxc
    deltac = (maxc - minc).clamp(min=1e-8)
    s = (deltac / (v + 1e-8)).clamp(0.0, 1.0)

    # Hue calculation
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = torch.zeros_like(v)
    mask_r = (maxc == r)
    mask_g = (maxc == g)
    mask_b = (maxc == b)
    h = torch.where(mask_r, (bc - gc) % 6.0, h)
    h = torch.where(mask_g, (2.0 + rc - bc), h)
    h = torch.where(mask_b, (4.0 + gc - rc), h)
    h = (h / 6.0) % 1.0

    hsv = torch.stack([h, s, v], dim=1)
    return hsv

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
    RGB空间的色彩恒常性损失：惩罚R,G,B通道的整体均值差，缓解色偏。
    直接接收RGB图像。
    """

    def __init__(self):
        super(L_color_rgb, self).__init__()

    def forward(self, rgb_image):
        channel_means = torch.mean(rgb_image, dim=[2, 3])  # (b,3)
        mean_r = channel_means[:, 0]
        mean_g = channel_means[:, 1]
        mean_b = channel_means[:, 2]
        loss_rg = torch.abs(mean_r - mean_g)
        loss_gb = torch.abs(mean_g - mean_b)
        loss_br = torch.abs(mean_b - mean_r)
        color_loss_per_sample = loss_rg + loss_gb + loss_br
        loss = torch.mean(color_loss_per_sample)
        return loss



def compute_total_loss(x_low, low_R, low_L, low_k, gamma_R, gamma_L, gamma_k, x_gamma, enhance_L, enhance_img, x_normal=None, loss_cfg=None):
    cfg = loss_cfg or {}
    decom_recon_loss_weight = cfg.get('decom_recon_loss_weight', 100)
    decom_consist_loss_weight = cfg.get('decom_consist_loss_weight', 10)
    decom_tv_loss_weight = cfg.get('decom_tv_loss_weight', 0.01)
    r_gradient_loss_weight = cfg.get('r_gradient_loss_weight', 0.1)
    L_con_weight = cfg.get('L_con_weight', 1000)
    L_spa_ex_weight = cfg.get('L_spa_ex_weight', 2)
    loss_color_constancy_weight = cfg.get('loss_color_constancy_weight', 3.0)
    Color_weight = cfg.get('Color_weight', 1.0)
    chroma_tv_reflect_weight = cfg.get('chroma_tv_reflect_weight', 0.05)
    anti_green_weight = cfg.get('anti_green_weight', 3.0)
    global_green_weight = cfg.get('global_green_weight', 1.0)
    anti_green_mid_weight = cfg.get('anti_green_mid_weight', 1.5)
    chroma_green_pen_weight = cfg.get('chroma_green_pen_weight', 0.8)
    dir_consistency_weight = cfg.get('dir_consistency_weight', 3.0)
    exp_mean_enhL = cfg.get('exp_mean_enhL', 0.85)
    exp_patch = cfg.get('exp_patch', 16)
    exp_mean_R = cfg.get('exp_mean_R', 0.4)
    bin_low_thr = cfg.get('bin_low_thr', 0.05)
    bin_high_thr = cfg.get('bin_high_thr', 0.98)
    dark_uv_target_weight = cfg.get('dark_uv_target_weight', 0.3)
    dark_chroma_loss_weight = cfg.get('dark_chroma_loss_weight', 0.15)
    loss_uv_weight = cfg.get('loss_uv_weight', 3.5)
    hue_pair_weight = cfg.get('hue_pair_weight', 1.5)
    sat_pair_weight = cfg.get('sat_pair_weight', 1.0)
    loss_channel_mean_weight = cfg.get('loss_channel_mean_weight', 0.5)
    chroma_tv_enh_weight = cfg.get('chroma_tv_enh_weight', 0.02)
    rg_loss_weight = cfg.get('rg_loss_weight', 1.0)
    illum_loss_scale = cfg.get('illum_loss_scale', 1.0)

    """总损失：Uretinex分解损失+Noise2noise去噪损失+Zero-DCE照明损失+重建损失"""
    # 添加数值稳定性
    # epsilon = 1e-8
    # 1. Uretinex分解损失（原始逻辑）
    # 分解保真损失：x_low ≈ L*(R + k)
    decom_recon_loss = nn.MSELoss()(low_L * (low_R + low_k), x_low) + nn.MSELoss()(gamma_L * (gamma_R + gamma_k), x_gamma)
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
    # 在HSV空间上约束色相变化（防止色偏漂移）
    hsv_low = rgb_to_hsv(torch.clamp(x_low, 0.0, 1.0))
    hsv_enh = rgb_to_hsv(torch.clamp(enhance_img, 0.0, 1.0))
    Color = L_hue_mean_diff()(hsv_low, hsv_enh)
    # 饱和度保留与过度饱和抑制（避免通过降低饱和度来满足其他损失）
    s_low = hsv_low[:,1:2,:,:]
    s_enh = hsv_enh[:,1:2,:,:]
    sat_preserve = torch.mean(F.relu(s_low - s_enh))      # 不允许降低饱和度
    sat_over = torch.mean(F.relu(s_enh - 1.2 * s_low))    # 过度提升也轻惩罚
    # 亮度均值（用于暗度自适应）
    mean_v_all = torch.mean(hsv_low[:,2,:,:])
    AAA = nn.MSELoss()(enhance_L,gamma_L)
    #print(AAA)
    color_constancy_loss_func = L_color_rgb()
    # 对反射率图和最终增强图加入色彩恒常性约束（直接在RGB空间）
    loss_color_constancy = color_constancy_loss_func(low_R) + color_constancy_loss_func(gamma_R) 
    # 自适应：暗度越高权重越大；场景越多彩（饱和度高）越小，避免把自然偏色场景拉向中性
    with torch.no_grad():
        darkness = torch.clamp(1.0 - mean_v_all, 0.0, 1.0)
        sat_mean_low = torch.mean(hsv_low[:,1,:,:]).clamp(0.0, 1.0)
    # 门控增强：对“暗且不够彩”的图更强纠偏，彩色场景更弱
    sat_gate = (1.0 - sat_mean_low) ** 2  # 多彩场景时显著减小
    loss_color_constancy += (0.5 + 1.5 * darkness) * sat_gate * color_constancy_loss_func(enhance_img)

    # 反射色度平滑（抑制极暗场景的彩色噪声在R中被放大）
    def tv_mean(x):
        # x: (b,1,h,w)
        return (torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])) +
                torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])))

    def rgb_to_yuv_local(x):
        r, g, b = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
        y = 0.299*r + 0.587*g + 0.114*b
        u = -0.14713*r - 0.28886*g + 0.436*b
        v = 0.615*r - 0.51499*g - 0.10001*b
        return y, u, v

    def chroma_rg(x):
        # x: (b,3,h,w) in [0,1]
        eps = 1e-6
        r, g, b = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
        s = (r + g + b).clamp(min=eps)
        rg = torch.cat([r/s, g/s], dim=1)
        return rg

    _, u_lr, v_lr = rgb_to_yuv_local(torch.clamp(low_R,0.0,1.0))
    _, u_gr, v_gr = rgb_to_yuv_local(torch.clamp(gamma_R,0.0,1.0))
    chroma_tv_reflect = tv_mean(u_lr) + tv_mean(v_lr) + tv_mean(u_gr) + tv_mean(v_gr)
    chroma_tv_reflect = 0.05 * chroma_tv_reflect

    # 暗区反绿正则（更强）：更强地抑制暗处的全局偏绿，阈值更低
    dark_w_local = (1.0 - hsv_low[:,2:3,:,:]).clamp(0.0, 1.0)
    g_over_rb = enhance_img[:,1:2,:,:] - torch.max(enhance_img[:,0:1,:,:], enhance_img[:,2:3,:,:])
    anti_green = torch.mean((dark_w_local**2) * F.relu(g_over_rb - 0.01))

    # 新增：中等亮度区域的抗绿（避免仅在极暗处约束，V∈[0.2,0.7]）
    v_map = hsv_low[:,2:3,:,:]
    mid_mask = ((v_map > 0.2) & (v_map < 0.7)).float()
    anti_green_mid = torch.mean(mid_mask * F.relu(g_over_rb - 0.01))

    # 新增：全局绿色偏移惩罚（按暗度加权），防止整体G通道均值高于R/B
    ch_means = torch.mean(enhance_img, dim=[2,3])  # (b,3)
    mean_r, mean_g, mean_b = ch_means[:,0], ch_means[:,1], ch_means[:,2]
    green_over_rb_global = mean_g - torch.max(mean_r, mean_b)
    img_dark_weight = torch.mean(dark_w_local.view(dark_w_local.size(0), -1), dim=1)  # (b,)
    global_green = torch.mean(img_dark_weight * F.relu(green_over_rb_global - 0.01))

    # 新增：基于色度比例的绿色偏置惩罚（rg色度中g占比过高时惩罚），暗区和中亮度权重
    def chroma_rg(x):
        eps = 1e-6
        r, g, b = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
        s = (r + g + b).clamp(min=eps)
        return torch.cat([r/s, g/s], dim=1)
    rg_enh_all = chroma_rg(torch.clamp(enhance_img,0.0,1.0))  # (b,2,h,w) with [r_share, g_share]
    g_excess = (rg_enh_all[:,1:2,:,:] - rg_enh_all[:,0:1,:,:])  # g over r share
    chroma_green_pen = torch.mean((0.5*mid_mask + dark_w_local) * F.relu(g_excess - 0.02))

    # 颜色方向一致性（保持RGB比例/色相方向接近原图；仅在中亮度且不高饱和区域生效）
    def cosine_dir_loss(a, b, mask):
        eps = 1e-6
        a = torch.clamp(a, 0.0, 1.0)
        b = torch.clamp(b, 0.0, 1.0)
        dot = (a * b).sum(dim=1, keepdim=True)
        na = torch.sqrt((a * a).sum(dim=1, keepdim=True) + eps)
        nb = torch.sqrt((b * b).sum(dim=1, keepdim=True) + eps)
        cos = dot / (na * nb + eps)
        lmap = (1.0 - cos).clamp(0.0, 2.0)
        return torch.mean(mask * lmap)

    neutral_mask = (s_low < 0.6).float()
    dir_mask = mid_mask * neutral_mask
    dir_consistency = cosine_dir_loss(enhance_img, x_low, dir_mask)

    # 反射层颜色保持：在暗处约束 R 的颜色方向与输入一致，并匹配色度比例，避免 R 退化为灰/无色
    r_dir_w = float(cfg.get('r_chroma_dir_w', 2.0))
    r_rg_w = float(cfg.get('r_chroma_rg_w', 1.0))
    v_mask_R = (hsv_low[:,2:3,:,:] < float(cfg.get('r_dir_v_thr', 0.25))).float()
    dir_R_loss = cosine_dir_loss(torch.clamp(low_R,0.0,1.0), torch.clamp(x_low,0.0,1.0), v_mask_R)
    rg_lowR = chroma_rg(torch.clamp(low_R,0.0,1.0))
    rg_xlow = chroma_rg(torch.clamp(x_low,0.0,1.0))
    rg_R_loss = torch.mean(v_mask_R * torch.abs(rg_lowR - rg_xlow))

    # k 正则：一致性、平滑、推动更大 k（在较暗/中亮区域更有效）
    k_cons = F.mse_loss(low_k, gamma_k)
    k_tv_loss_func = L_TV()
    k_tv = 0.02 * (k_tv_loss_func(low_k) + k_tv_loss_func(gamma_k))
    v_map_for_k = hsv_low[:,2:3,:,:]
    s_map_for_k = hsv_low[:,1:2,:,:]
    k_mask = ((v_map_for_k < 0.7) & (s_map_for_k < 0.7)).float()
    k_push = -0.1 * torch.mean(k_mask * low_k)

    total_decom_loss = 2*decom_recon_loss + 10 * decom_consist_loss + 0.01 * decom_tv_loss \
                       + 0.1 * r_gradient_loss + 1 * L_con + 2 * L_spa_ex  \
                       + 3.0 * loss_color_constancy + 1.0 * Color + chroma_tv_reflect \
                       + 1.0 * sat_preserve + 0.2 * sat_over + 3.0 * anti_green + 1.0 * global_green \
                       + 1.5 * anti_green_mid + 0.8 * chroma_green_pen + 3.0 * dir_consistency \
                       + r_dir_w * dir_R_loss + r_rg_w * rg_R_loss \
                       + 0.3 * k_cons + k_tv + k_push

    # 暗区像素的色度约束：若处于“早期配对监督阶段”且提供了配对正常光，
    # 则在暗处“对齐到正常光UV”；否则使用自监督的弱色度抑制分支。
    v_low_map = hsv_low[:,2:3,:,:]  # (b,1,h,w)
    dark_weight_map = (1.0 - v_low_map).clamp(0.0, 1.0)
    epoch_idx = int(cfg.get('epoch', 0))
    paired_color_until = int(cfg.get('paired_color_until', 5))
    if (x_normal is not None) and (epoch_idx < paired_color_until):
        _, u_enh_all, v_enh_all = rgb_to_yuv_local(torch.clamp(enhance_img, 0.0, 1.0))
        _, u_norm_all, v_norm_all = rgb_to_yuv_local(torch.clamp(x_normal, 0.0, 1.0))
        dark_uv_target = torch.mean(dark_weight_map * ((u_enh_all - u_norm_all)**2 + (v_enh_all - v_norm_all)**2))
        total_decom_loss = total_decom_loss + 0.3 * dark_uv_target
    else:
        _, u_enh_all, v_enh_all = rgb_to_yuv_local(torch.clamp(enhance_img, 0.0, 1.0))
        dark_chroma_loss = torch.mean(dark_weight_map * (u_enh_all**2 + v_enh_all**2))
        total_decom_loss = total_decom_loss + 0.15 * dark_chroma_loss

    # 额外：使用配对的正常光图（若提供）做颜色约束，仅在“早期配对监督阶段”启用；
    # 之后保持自监督，避免过拟合导致后期退化。
    if (x_normal is not None) and (epoch_idx < paired_color_until):
        with torch.no_grad():
            # darkness 根据输入亮度确定 [0,1]，越暗越接近1
            v_low = rgb_to_hsv(torch.clamp(x_low, 0.0, 1.0))[:,2,:,:]
            mean_v = torch.mean(v_low)
            darkness = torch.clamp(1.0 - mean_v, 0.0, 1.0)

        # YUV色度匹配（抑制色偏），仅匹配UV，忽略亮度
        def rgb_to_yuv(x):
            r, g, b = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
            y = 0.299*r + 0.587*g + 0.114*b
            u = -0.14713*r - 0.28886*g + 0.436*b
            v = 0.615*r - 0.51499*g - 0.10001*b
            return y, u, v

        _, u_e, v_e = rgb_to_yuv(torch.clamp(enhance_img, 0.0, 1.0))
        _, u_n, v_n = rgb_to_yuv(torch.clamp(x_normal, 0.0, 1.0))
        loss_uv = F.mse_loss(u_e, u_n) + F.mse_loss(v_e, v_n)

        # 色相/饱和度接近正常光（避免整体色调漂移与偏色饱和）
        hsv_norm = rgb_to_hsv(torch.clamp(x_normal, 0.0, 1.0))
        hue_pair = L_hue_mean_diff()(hsv_norm, hsv_enh)
        sat_pair = F.mse_loss(hsv_enh[:,1:2,:,:], hsv_norm[:,1:2,:,:])

        # 通道均值与正常光对齐（弱约束）
        def channel_mean(x):
            return torch.mean(x, dim=[0,2,3])
        cm_enh = channel_mean(torch.clamp(enhance_img,0.0,1.0))
        cm_norm = channel_mean(torch.clamp(x_normal,0.0,1.0))
        loss_channel_mean = torch.mean(torch.abs(cm_enh - cm_norm))

        # 增强图的色度平滑，避免在高提升下显著偏色噪点
        _, u_e2, v_e2 = rgb_to_yuv_local(torch.clamp(enhance_img,0.0,1.0))
        chroma_tv_enh = 0.02 * (tv_mean(u_e2) + tv_mean(v_e2))

        # 暗度加权的色度比例(r,g)匹配，避免靠通道均值导致“发灰”
        rg_enh = chroma_rg(torch.clamp(enhance_img,0.0,1.0))
        rg_norm = chroma_rg(torch.clamp(x_normal,0.0,1.0))
        dark_w = (1.0 - v_low.unsqueeze(1)).clamp(0.0,1.0)  # (b,1,h,w)
        rg_loss = torch.mean(dark_w * torch.abs(rg_enh - rg_norm))

        paired_color_loss = 3.5*loss_uv + 1.5*hue_pair + 1.0*sat_pair + 0.5*loss_channel_mean + chroma_tv_enh + 1.0*rg_loss
        darkness_scale = 3.0 * darkness  # 更强地在暗场景对齐到正常光，抑制偏绿
        total_decom_loss = total_decom_loss + darkness_scale * paired_color_loss

    #print(f'decom_recon_loss:{decom_recon_loss:.4f}, decom_consist_loss: {1*decom_consist_loss:.4f}, decom_tv_loss:{0.01 * decom_tv_loss:.4f}, L_con:{0.1 * L_con:.4f}, r_gradient_loss:{r_gradient_loss:.4f}')


    # 曝光目标（可调）并在极暗样本上进一步下调曝光损失权重
    exp_mean_enhL = float(cfg.get('exp_mean_enhL', 0.85))
    patch_size = int(cfg.get('exp_patch', 16))
    exp_loss = L_exp(patch_size=patch_size, mean_val=exp_mean_enhL)(enhance_L)
    exp_loss = (0.4 + 0.6 * mean_v_all) * exp_loss
    exp_mean_R = float(cfg.get('exp_mean_R', 0.4))
    exp_R_loss = L_exp(patch_size=patch_size, mean_val=exp_mean_R)(low_R) + torch.abs(torch.mean(low_R) - exp_mean_R)

    # 防止增强照明近似二值化：对接近0或1的像素给惩罚（边界惩罚，可调阈值）
    bin_low_thr = float(cfg.get('bin_low_thr', 0.05))
    bin_high_thr = float(cfg.get('bin_high_thr', 0.98))
    bin_penalty = torch.mean(F.relu(bin_low_thr - enhance_L) + F.relu(enhance_L - bin_high_thr))
    # 照明平滑（小权重TV）
    tv_enh = (torch.mean(torch.abs(enhance_L[:,:,:,1:] - enhance_L[:,:,:,:-1])) +
              torch.mean(torch.abs(enhance_L[:,:,1:,:] - enhance_L[:,:,:-1,:])))
    # 与 gamma_L 对齐（小权重，可调）
    align_gamma = F.mse_loss(enhance_L, torch.clamp(gamma_L,0.0,1.0))
    # 新增：振幅约束与梯度对齐，限制增强照明与gamma照明差异不要过大且结构一致
    amp_map = enhance_L - torch.clamp(gamma_L, 0.0, 1.0)
    amp_loss = torch.mean(torch.abs(amp_map))
    dx_e = enhance_L[:,:,:,1:] - enhance_L[:,:,:,:-1]
    dy_e = enhance_L[:,:,1:,:] - enhance_L[:,:,:-1,:]
    dx_g = gamma_L[:,:,:,1:] - gamma_L[:,:,:,:-1]
    dy_g = gamma_L[:,:,1:,:] - gamma_L[:,:,:-1,:]
    grad_align = torch.mean(torch.abs(dx_e - dx_g)) + torch.mean(torch.abs(dy_e - dy_g))

    v_low_map = hsv_low[:,2:3,:,:]
    v_thr = float(cfg.get('dark_v_thr', 0.10))
    s_thr = float(cfg.get('dark_s_thr', 0.20))
    lift_thr = float(cfg.get('dark_lift_thr', 0.20))
    dark_mask = ((v_low_map < v_thr) & (s_low < s_thr)).float()
    dark_lift = torch.mean(dark_mask * F.relu(lift_thr - enhance_L))
    bin_w = float(cfg.get('bin_penalty_w', 3.0))
    tv_w = float(cfg.get('tv_enh_w', 0.05))
    align_w = float(cfg.get('align_gamma_w', 0.25))
    amp_w = float(cfg.get('amp_loss_w', 1.2))
    grad_w = float(cfg.get('grad_align_w', 0.1))
    dark_w = float(cfg.get('dark_lift_w', 2.0))
    total_illum_loss = exp_loss + 1.0 * exp_R_loss 
    total_illum_loss = total_illum_loss + bin_w * bin_penalty + tv_w * tv_enh + align_w * align_gamma \
                                       + amp_w * amp_loss + grad_w * grad_align + dark_w * dark_lift

    # 总损失（关键修改：确保所有损失项都是标量，并最终求和为标量）
    illum_scale = float(cfg.get('illum_loss_scale', 0.05))
    total_loss = total_decom_loss + illum_scale*total_illum_loss

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


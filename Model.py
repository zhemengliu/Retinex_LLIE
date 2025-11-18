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
        # select: 0 -> DecomNet_RTV (current), 1 -> IterativeUretinex
        self.select = getattr(args, 'select', 0)
        # 颜色校正强度（主要用于RTV路径）
        self.cc_gain = float(getattr(args, 'cc_gain', 1.0))
        self.wb_alpha_cap = float(getattr(args, 'wb_alpha_cap', 0.6))
        # 降噪：统一强度由 train.py 的 --denoise_strength 控制，实际强度按输入自适应
        self.use_denoise = int(getattr(args, 'denoise', 1))
        self.denoise_strength_base = float(getattr(args, 'denoise_strength', 0.6))
        self.denoise_stage = getattr(args, 'denoise_stage', 'both')
        self.denoise_dark_thresh = float(getattr(args, 'denoise_dark_thresh', 0.45))
        self.denoise_sat_thresh = float(getattr(args, 'denoise_sat_thresh', 0.35))
        self.denoise_edge_weight = float(getattr(args, 'denoise_edge_weight', 0.6))
        if self.select == 1:
            # Normalize UR hyperparams to stable defaults (do not affect RTV path)
            # These fields are used internally by IterativeUretinex
            if not hasattr(args, 'unfolding_round') or args.unfolding_round is None:
                args.unfolding_round = 3
            args.gamma = 0.1
            args.lamda = 0.1
            args.Roffset = 0.05
            args.Loffset = 0.05
            args.concat_L = False
            self.decom_iter = IterativeUretinex(args)
            self.decom_rtv = None
        else:
            self.decom_rtv = DecomNet_RTV(in_ch=1, k1=10)
            self.decom_iter = None
        self.enhance_net = enhance_net_nopool()

        self.k_max = float(getattr(args, 'k_max', 0.4))
        self.k_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        # 轻量颜色校正头：从输入全局特征预测 3x3 颜色矩阵与偏置
        # 初始化为单位矩阵 + 零偏置，起始不改变颜色，由损失引导逐步学习
        self.cc_gap = nn.AdaptiveAvgPool2d(1)
        self.cc_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True)
        )
        self.cc_fc = nn.Linear(16, 12)  # 输出 9(矩阵) + 3(偏置)
        with torch.no_grad():
            self.cc_fc.weight.zero_()
            # bias 排成 [1,0,0, 0,1,0, 0,0,1, 0,0,0]
            init_bias = torch.tensor([1,0,0, 0,1,0, 0,0,1, 0,0,0], dtype=torch.float)
            self.cc_fc.bias.copy_(init_bias)

        # 轻量降噪网络（预测噪声残差），最后一层零初始化，初始即为恒等
        if self.use_denoise:
            # 输出域降噪
            self.denoise_net = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1)
            )
            with torch.no_grad():
                for m in self.denoise_net.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)
                # 最后一层置零，起始不改变图像
                last = list(self.denoise_net.modules())[-1]
                if isinstance(last, nn.Conv2d):
                    nn.init.constant_(last.weight, 0.0)
                    if last.bias is not None:
                        nn.init.constant_(last.bias, 0.0)
            # R 域降噪（与输出域分开，避免权重耦合）
            self.r_denoise_net = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1)
            )
            with torch.no_grad():
                for m in self.r_denoise_net.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)
                last_r = list(self.r_denoise_net.modules())[-1]
                if isinstance(last_r, nn.Conv2d):
                    nn.init.constant_(last_r.weight, 0.0)
                    if last_r.bias is not None:
                        nn.init.constant_(last_r.bias, 0.0)

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
        # --- 1. 原始分解 (可切换) ---
        if self.select == 1 and self.decom_iter is not None:
            # 使用 IterativeUretinex，直接得到 R, L（保持[0,1]范围）
            low_R, low_L = self.decom_iter(low_img)
            low_R = torch.clamp(low_R, 0.0, 1.0)
            low_L = torch.clamp(low_L, 0.0, 1.0)
        else:
            # 使用 DecomNet_RTV：对亮度图做估计，再由除法得到反射率
            low_L0 = torch.max(low_img, dim=1, keepdim=True)[0]
            low_L = self.decom_rtv(low_L0)
            low_L = torch.clamp(low_L, 0.1, 20.0)
            low_R = low_img / (low_L + 1e-3)
            low_R = torch.clamp(low_R, 0.1, 20.0)

        # --- 2. Gamma参考分支 (与您原来一致, 用于计算损失) ---
        x_gamma = self.gamma_correction(low_img)
        if self.select == 1 and self.decom_iter is not None:
            gamma_R, gamma_L = self.decom_iter(x_gamma)
            gamma_L = torch.clamp(gamma_L, 0.0, 1.0)
            gamma_R = torch.clamp(gamma_R, 0.0, 1.0)
        else:
            gamma_L0 = torch.max(x_gamma, dim=1, keepdim=True)[0]
            gamma_L = self.decom_rtv(gamma_L0)
            gamma_L = torch.clamp(gamma_L, 0.01, 10.0)
            gamma_R = x_gamma / (gamma_L + 1e-3)
            gamma_R = torch.clamp(gamma_R, 0.01, 10.0)

        # --- [核心修改 1: 为增强网络减负，提供更好的起点] ---
        # 我们将分解出的 low_L 进行一次Gamma校正，
        # 让 enhance_net 的任务从“大幅提亮”变为“精细调整”。
        # UR分支(L在[0,1])与RTV分支(L在约[0.1,20])分布不同，
        # 为避免输入分布漂移导致增强网产生条纹/伪影，
        # 在select==1时先将L缩放到与RTV类似的动态范围再做gamma。
        # UR使用更温和的gamma，RTV保持原值
        gamma_val_for_L = 1.6 if (self.select == 1 and self.decom_iter is not None) else 2.2
        if self.select == 1 and self.decom_iter is not None:
            # UR分支：保持在[0,1]范围
            low_L_for_enh = torch.clamp(low_L, 0.0, 1.0)
        else:
            # RTV分支：维持原有动态范围
            low_L_for_enh = torch.clamp(low_L, 0.1, 20.0)
        low_L_gamma_input = torch.pow(low_L_for_enh + 1e-6, 1.0 / gamma_val_for_L)

        # 将预处理过的 low_L_gamma_input 送入增强网络
        enhance_L, _ = self.enhance_net(low_L_gamma_input)
        # 仅在UR路径：对增强后的L做轻量平滑并与原L保守融合，避免块状/条纹
        if self.select == 1 and self.decom_iter is not None:
            with torch.no_grad():
                k = torch.ones((1, 1, 3, 3), device=enhance_L.device, dtype=enhance_L.dtype) / 9.0
            enhance_L = F.conv2d(enhance_L, k, padding=1)
            # 保守融合：85%增强 + 15%原L
            enhance_L = 0.85 * enhance_L + 0.15 * torch.clamp(low_L, 0.0, 1.0)

        with torch.no_grad():
            low_L_k = torch.clamp(low_L, 0.0, 1.0) if (self.select == 1 and self.decom_iter is not None) else torch.clamp(low_L, 0.0, 1.0)
            gamma_L_k = torch.clamp(gamma_L, 0.0, 1.0)
        low_k = self.k_max * torch.sigmoid(self.k_net(low_L_k))
        gamma_k = self.k_max * torch.sigmoid(self.k_net(gamma_L_k))

        # --- [核心修改 2: 强制模型学习提亮，提供驱动力] ---
        # 在最终重建时，我们对原始反射图 low_R 进行缩放。
        # 为了让最终的 enhance_img 在损失函数（如L1, SSIM）的监督下
        # 仍然能匹配目标亮度，模型将被迫学习输出一个更亮的 enhance_L 来补偿。
        reflectance_factor = 1

        # R 域强效降噪（在重建前），仅当开启 denoise 且阶段包含 reflectance
        low_R_eff = low_R
        if self.use_denoise and (self.denoise_stage in ['reflectance', 'both']):
            # 暗区/低饱和掩码（来自输入低照图像的 HSV）
            with torch.no_grad():
                hsv_in = rgb_to_hsv(torch.clamp(low_img, 0.0, 1.0))
                V = hsv_in[:, 2:3, :, :]
                S = hsv_in[:, 1:2, :, :]
                m_dark = (V < self.denoise_dark_thresh).float()
                m_sat = (S < self.denoise_sat_thresh).float()
                mask = (m_dark * m_sat)
                # 边缘保护：基于 R 的亮度梯度削弱掩码
                r_gray = (0.299 * low_R[:,0:1] + 0.587 * low_R[:,1:2] + 0.114 * low_R[:,2:3])
                kx = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=low_R.dtype, device=low_R.device).view(1,1,3,3)
                ky = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=low_R.dtype, device=low_R.device).view(1,1,3,3)
                gx = F.conv2d(r_gray, kx, padding=1)
                gy = F.conv2d(r_gray, ky, padding=1)
                grad_mag = torch.sqrt(gx*gx + gy*gy)
                # 归一化到 [0,1]
                g_norm = grad_mag / (grad_mag.amax(dim=[2,3], keepdim=True) + 1e-6)
                mask = mask * (1.0 - self.denoise_edge_weight * g_norm).clamp(0.0, 1.0)

                # 基于暗低饱和区域的高频能量估计噪声强度（越嘈越大）
                k = torch.ones((3,3), device=low_img.device, dtype=low_img.dtype) / 9.0
                k = k.view(1,1,3,3)
                li_gray = (0.299*low_img[:,0:1]+0.587*low_img[:,1:2]+0.114*low_img[:,2:3])
                blur = F.conv2d(li_gray, k, padding=1)
                hf = torch.abs(li_gray - blur)
                denom = mask.sum(dim=[1,2,3], keepdim=True) + 1e-6
                score = (hf*mask).sum(dim=[1,2,3], keepdim=True) / denom  # ~[0,1]
                noise_factor = torch.clamp(score/0.08, 0.0, 1.0)  # 经验归一
                r_strength = self.denoise_strength_base * noise_factor
            # 规范化到 [0,1] 以稳定卷积残差
            R_in = torch.clamp(low_R, 0.0, 1.0)
            r_res = self.r_denoise_net(R_in)
            low_R_eff = torch.clamp(low_R - r_strength * mask * r_res,
                                     0.0,
                                     1.0 if (self.select == 1 and self.decom_iter is not None) else 20.0)

            # 反射域色度平滑（边缘保护，仅在暗且低饱和区域更强）
            def rgb_to_yuv_local(x):
                r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
                y = 0.299*r + 0.587*g + 0.114*b
                u = -0.14713*r - 0.28886*g + 0.436*b
                v = 0.615*r - 0.51499*g - 0.10001*b
                return y, u, v
            def yuv_to_rgb_local(y, u, v):
                r = y + 1.13983*v
                g = y - 0.39465*u - 0.58060*v
                b = y + 2.03211*u
                return torch.cat([r,g,b], dim=1)
            def gauss_blur(ch, sigma=1.2, ksize=5):
                half = ksize // 2
                with torch.no_grad():
                    coords = torch.arange(ksize, device=ch.device, dtype=ch.dtype) - half
                    g1d = torch.exp(-(coords*coords)/(2*sigma*sigma))
                    g1d = (g1d / g1d.sum()).view(1,1,ksize,1)
                ch = F.conv2d(ch, g1d, padding=(half,0), groups=ch.size(1))
                ch = F.conv2d(ch, g1d.transpose(2,3), padding=(0,half), groups=ch.size(1))
                return ch
            y,u,v = rgb_to_yuv_local(torch.clamp(low_R_eff,0.0,1.0))
            u_b = gauss_blur(u, 1.2, 5)
            v_b = gauss_blur(v, 1.2, 5)
            with torch.no_grad():
                w = mask * (1.0 - g_norm).clamp(0.0,1.0)
            u_s = u*(1.0-w) + u_b*w
            v_s = v*(1.0-w) + v_b*w
            low_R_eff = yuv_to_rgb_local(y, u_s, v_s)
            low_R_eff = torch.clamp(low_R_eff,
                                    0.0,
                                    1.0 if (self.select == 1 and self.decom_iter is not None) else 20.0)

        # 使用增强后的 enhance_L 和 处理后的 low_R_eff 进行重建
        enhance_img = enhance_L * (low_R_eff * reflectance_factor)
        enhance_img = torch.clamp(enhance_img, 0.0, 1.0)

        # --- 颜色校正（可学习）---
        # 用增强后的图像作为特征来源，使颜色校正更直接地针对当前输出的色偏
        feat = self.cc_conv(self.cc_gap(enhance_img))  # (b,16,1,1)
        params = self.cc_fc(feat.view(feat.size(0), -1))  # (b,12)
        M = params[:, :9].view(-1, 3, 3)
        b = params[:, 9:].view(-1, 3, 1)

        # 应用到增强图：y = M x + b（逐像素线性颜色变换）
        B, C, H, W = enhance_img.shape
        x = enhance_img.view(B, 3, -1)  # (b,3,HW)
        y = torch.bmm(M, x) + b  # (b,3,HW)
        head_out = y.view(B, 3, H, W)
        head_out = torch.clamp(head_out, 0.0, 1.0)
        # RTV路径：将颜色头相对增强，按偏移量放大（head_out - enhance_img）
        if not (self.select == 1 and self.decom_iter is not None):
            head_out = torch.clamp(enhance_img + self.cc_gain * (head_out - enhance_img), 0.0, 1.0)
        warmup_active = self.training and (self.select == 1) and getattr(self, 'warmup_color', False)

        # 先占位，待 base_raw 计算后再进行偏绿检测
        with torch.no_grad():
            trigger = torch.zeros(B, dtype=torch.bool, device=head_out.device)
            strength = torch.zeros(B, device=head_out.device)

        # 基础输出（不经过颜色头）
        base_raw = torch.clamp(enhance_img, 0.0, 1.0)
        # Warmup（UR专用）：直接输出颜色头与base_raw保守融合，保证颜色头有梯度且不过度偏色
        if warmup_active:
            enhance_out = torch.clamp(0.3 * head_out + 0.7 * base_raw, 0.0, 1.0)
            return low_R, low_L, low_k, gamma_R, gamma_L, gamma_k, x_gamma, enhance_L, enhance_out

        # 偏绿检测（放在此处，base_raw已可用）
        with torch.no_grad():
            # RTV路径：使用颜色头输出；UR路径：使用 base_raw 更稳健
            ref_img = base_raw if (self.select == 1 and self.decom_iter is not None) else head_out
            ref_img = torch.clamp(ref_img, 0.0, 1.0)
            r_ref = ref_img[:, 0:1, :, :]
            g_ref = ref_img[:, 1:2, :, :]
            b_ref = ref_img[:, 2:3, :, :]

            # 仅在“近中性区域”上统计（忽略强烈颜色区域）：S低、V不太暗
            hsv_ref = rgb_to_hsv(ref_img)
            S = hsv_ref[:, 1:2, :, :]
            V = hsv_ref[:, 2:3, :, :]
            neutral_mask = ((S < 0.25) & (V > 0.10)).float()  # 经验阈值，可再调

            g_over_rb_map = g_ref - torch.max(r_ref, b_ref)  # (B,1,H,W)
            masked_map = g_over_rb_map * neutral_mask
            eps = 1e-6
            # 全局均值（只在mask区域）
            valid_count = neutral_mask.sum(dim=[1,2,3]).clamp_min(eps)  # (B,)
            g_over_rb_mean = (masked_map.sum(dim=[1,2,3]) / valid_count)  # (B,)

            # 8x8 平均块（只在mask区域且有覆盖才计入）
            pool = nn.AdaptiveAvgPool2d((8, 8))
            masked_mean_patch = pool(masked_map)
            mask_cov_patch = pool(neutral_mask)
            # 只统计覆盖率>20%的patch，避免极少像素误判
            cover_ok = (mask_cov_patch > 0.20).float()
            safe_mean_patch = masked_mean_patch / (mask_cov_patch + eps)
            g_patch = (safe_mean_patch * cover_ok).view(B, -1)
            # 每张图 > 阈值的有效块比例（按覆盖过滤后）
            valid_blocks = cover_ok.view(B, -1).sum(dim=1).clamp_min(1.0)
            frac_pos = ((g_patch > 0.03).float().sum(dim=1) / valid_blocks)
            # 有效块上的std
            # 先将无效块填充为各自图的均值，避免影响std
            patch_mean_per_img = (g_patch.sum(dim=1, keepdim=True) / valid_blocks.view(B,1))
            g_patch_filled = torch.where(cover_ok.view(B,-1) > 0, g_patch, patch_mean_per_img)
            patch_std = g_patch_filled.std(dim=1)
            trigger = (g_over_rb_mean > 0.03) & (frac_pos > 0.7) & (patch_std < 0.03)
            strength = torch.clamp((g_over_rb_mean - 0.04) / 0.06, 0.0, 1.0)

        # 非偏绿图：关闭颜色头融合，避免引入增亮与色偏
        base_non_green = base_raw

        # 偏绿图：跳过颜色头，做亮度保持的受限抗绿
        with torch.no_grad():
            means = torch.mean(base_raw.view(B, 3, -1), dim=2) + 1e-6  # (B,3)
            gray = means.mean(dim=1, keepdim=True)  # (B,1)
            gains_raw = (gray / means)  # (B,3)
            gR = gains_raw[:, 0:1].clamp(0.98, 1.08)
            gG = gains_raw[:, 1:2].clamp(0.90, 1.0)   # 仅削减G，减幅更保守
            gB = gains_raw[:, 2:3].clamp(0.98, 1.08)
            gains_limited = torch.cat([gR, gG, gB], dim=1)

        wb = base_raw * gains_limited.view(B, 3, 1, 1)
        wb = torch.clamp(wb, 0.0, 1.0)
        # 亮度保持：尽量维持Y通道均值不变
        with torch.no_grad():
            def mean_Y(img):
                r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
                return (0.299 * r + 0.587 * g + 0.114 * b).mean(dim=[1, 2, 3])
            y_base = mean_Y(base_raw)
            y_wb = mean_Y(wb)
            k = (y_base / (y_wb + 1e-6)).view(B, 1, 1, 1)
        wb = torch.clamp(wb * k, 0.0, 1.0)

        # 根据是否触发以及强度进行融合；未触发时用 base_non_green
        strength = strength.view(B, 1, 1, 1).to(base_raw.device)
        mask = trigger.float().view(B, 1, 1, 1).to(base_raw.device)
        # 更积极的上限以抑制偏绿，但仍保守；UR更保守
        alpha_cap = 0.3 if (self.select == 1 and self.decom_iter is not None) else float(self.wb_alpha_cap)
        alpha = (strength * mask).clamp(0.0, alpha_cap)
        _blend_vis = torch.clamp(base_non_green * (1 - alpha) + wb * alpha, 0.0, 1.0)

        # 输出融合：
        # RTV路径：保持原逻辑（颜色头与WB融合）；UR路径：未触发时禁用颜色头，触发时仅与WB融合
        if self.select == 1 and self.decom_iter is not None:
            enhance_out = torch.clamp(base_non_green * (1 - alpha) + wb * alpha, 0.0, 1.0)
        else:
            enhance_out = torch.clamp(head_out * (1 - alpha) + wb * alpha, 0.0, 1.0)

        # 后处理降噪（仅当开启，且阶段包含 output）
        if self.use_denoise and (self.denoise_stage in ['output', 'both']):
            noise = self.denoise_net(enhance_out)
            # 复用上面的噪声估计（若未计算，则简单按 base 使用）
            if 'noise_factor' not in locals():
                with torch.no_grad():
                    hsv_in = rgb_to_hsv(torch.clamp(low_img, 0.0, 1.0))
                    V = hsv_in[:, 2:3]
                    S = hsv_in[:, 1:2]
                    mask_o = ((V < self.denoise_dark_thresh) & (S < self.denoise_sat_thresh)).float()
                    k = torch.ones((3,3), device=low_img.device, dtype=low_img.dtype) / 9.0
                    k = k.view(1,1,3,3)
                    li_gray = (0.299*low_img[:,0:1]+0.587*low_img[:,1:2]+0.114*low_img[:,2:3])
                    blur = F.conv2d(li_gray, k, padding=1)
                    hf = torch.abs(li_gray - blur)
                    denom = mask_o.sum(dim=[1,2,3], keepdim=True) + 1e-6
                    score = (hf*mask_o).sum(dim=[1,2,3], keepdim=True) / denom
                    noise_factor = torch.clamp(score/0.08, 0.0, 1.0)
            out_strength = self.denoise_strength_base * (0.8*noise_factor + 0.2)
            enhance_out = torch.clamp(enhance_out - out_strength * noise, 0.0, 1.0)

        if self.use_denoise and (self.denoise_stage in ['output', 'both']):
            hsv_o = rgb_to_hsv(torch.clamp(enhance_out,0.0,1.0))
            with torch.no_grad():
                m_dark_o = (hsv_o[:,2:3] < max(0.45, float(self.denoise_dark_thresh))).float()
                m_sat_o = (hsv_o[:,1:2] < max(0.35, float(self.denoise_sat_thresh))).float()
                m_o = (m_dark_o*m_sat_o)
            def rgb_to_yuv_local(x):
                r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
                y = 0.299*r + 0.587*g + 0.114*b
                u = -0.14713*r - 0.28886*g + 0.436*b
                v = 0.615*r - 0.51499*g - 0.10001*b
                return y, u, v
            def yuv_to_rgb_local(y, u, v):
                r = y + 1.13983*v
                g = y - 0.39465*u - 0.58060*v
                b = y + 2.03211*u
                return torch.cat([r,g,b], dim=1)
            def gauss_blur(ch, sigma=1.0, ksize=5):
                half = ksize // 2
                with torch.no_grad():
                    coords = torch.arange(ksize, device=ch.device, dtype=ch.dtype) - half
                    g1d = torch.exp(-(coords*coords)/(2*sigma*sigma))
                    g1d = (g1d / g1d.sum()).view(1,1,ksize,1)
                ch = F.conv2d(ch, g1d, padding=(half,0), groups=ch.size(1))
                ch = F.conv2d(ch, g1d.transpose(2,3), padding=(0,half), groups=ch.size(1))
                return ch
            y_o, u_o, v_o = rgb_to_yuv_local(enhance_out)
            u_ob = gauss_blur(u_o, 1.0, 5)
            v_ob = gauss_blur(v_o, 1.0, 5)
            u_mix = u_o*(1.0-m_o) + u_ob*m_o
            v_mix = v_o*(1.0-m_o) + v_ob*m_o
            enhance_out = torch.clamp(yuv_to_rgb_local(y_o, u_mix, v_mix), 0.0, 1.0)

        return low_R, low_L, low_k, gamma_R, gamma_L, gamma_k, x_gamma, enhance_L, enhance_out


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
    low_R, low_L, low_k, gamma_R, gamma_L, gamma_k, x_gamma, enhance_L, enhance_img=net(x)
    #print(enhance_img.shape)





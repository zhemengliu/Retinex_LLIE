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


class L_spa(nn.Module):
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
        b, c, h, w = org.shape
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        # 原始权重计算，未修改
        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)
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
        D_left = torch.pow(D_org_left - D_enhance_left, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
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
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()
    def forward(self, x ):
        x = torch.clamp(x, 0.0, 1.0)
        b,c,h,w = x.shape
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        # 关键修改：对 batch 维度取均值，确保返回标量
        return k.mean()


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
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def compute_total_loss(x_low, low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img):
    """总损失：Uretinex分解损失+Noise2noise去噪损失+Zero-DCE照明损失+重建损失"""
    # 添加数值稳定性
    # epsilon = 1e-8
    # 1. Uretinex分解损失（原始逻辑）
    # 分解保真损失：x_low ≈ R*L
    decom_recon_loss = nn.MSELoss()(low_R * low_L, x_low) + nn.MSELoss()(gamma_R * gamma_L, x_gamma)
    r_gradient_loss  = L_spa()(low_R, x_low) + L_spa()(gamma_R, x_gamma)
    decom_consist_loss = nn.MSELoss()(low_R, gamma_R)
    decom_tv_loss = L_TV()(low_L)
    low_L0 = torch.max(x_low, dim=1, keepdim=True)[0]  # 保留批次和通道维度
    gamma_L0 = torch.max(x_gamma, dim=1, keepdim=True)[0]
    low_L0 = torch.clamp(low_L0, 0.1, 1.0)  # 约束范围
    gamma_L0 = torch.clamp(gamma_L0, 0.1, 1.0)

    L_con = nn.MSELoss()(low_L0, low_L) + nn.MSELoss()(gamma_L0, gamma_L)

    total_decom_loss = 10 * decom_consist_loss + decom_tv_loss + r_gradient_loss
    print(f'decom_recon_loss:{decom_recon_loss:.4f}, decom_consist_loss: {10*decom_consist_loss:.4f}, decom_tv_loss:{decom_tv_loss:.4f}, L_con:{0.1 * L_con:.4f}, r_gradient_loss:{r_gradient_loss:.4f}')

    # 2. Noise2noise去噪损失（原始逻辑：带噪反射层→干净反射层）
    # denoise_loss = nn.MSELoss()(denoise_R, gt_R)
    # 3. Zero-DCE照明增强损失（原始逻辑）
    # L_normal = torch.mean(x_normal, dim=1, keepdim=True)  # 正常光照明层标签

    # spa_loss = torch.mean(L_spa()(L_normal, enhance_L1))
    # color_loss = L_color()(enhance_L)  # 扩3通道算颜色损失#######################改enhance_low为enhance_img
    # tv_loss = L_TV()(enhance_img)
    exp_loss = L_exp(patch_size=16, mean_val=0.6)(enhance_L)
    # exp_loss_L = L_exp(patch_size=16, mean_val=0.6)(enhance_L)

    total_illum_loss = exp_loss  #+ 0.01 * tv_loss #+ exp_loss_L+ 0.3 * color_loss

    # 总损失（关键修改：确保所有损失项都是标量，并最终求和为标量）
    total_loss = (total_decom_loss + total_illum_loss)

    # 强制确保 total_loss 是标量（添加均值操作，针对可能的维度残留）
    if total_loss.dim() > 0:
        total_loss = total_loss.mean()

    return total_loss, total_decom_loss, total_illum_loss

# def compute_total_loss(x_low, x_normal, low_R, low_L, gt_R, gt_L, enhance_L1, enhance_L, denoise_R, enhance_img,
#                        uretinex_opts):
#     """修复：添加数据范围检查和处理"""
#
#     # 修复1：确保所有输入在合理范围内
#     x_low = torch.clamp(x_low, 0.0, 1.0)
#     x_normal = torch.clamp(x_normal, 0.0, 1.0)
#     low_R = torch.clamp(low_R, 0.0, 1.0)
#     low_L = torch.clamp(low_L, 0.0, 1.0)
#     gt_R = torch.clamp(gt_R, 0.0, 1.0)
#     gt_L = torch.clamp(gt_L, 0.0, 1.0)
#     enhance_L1 = torch.clamp(enhance_L1, 0.0, 1.0)
#     enhance_L = torch.clamp(enhance_L, 0.0, 1.0)
#     denoise_R = torch.clamp(denoise_R, 0.0, 1.0)
#     enhance_img = torch.clamp(enhance_img, 0.0, 1.0)
#
#     # 修复2：添加数值稳定性处理
#     epsilon = 1e-8
#
#     # 1. Uretinex分解损失
#     decom_recon_loss = nn.MSELoss()(low_R * low_L + epsilon, x_low + epsilon) + \
#                        nn.MSELoss()(gt_R * gt_L + epsilon, x_normal + epsilon)
#
#     decom_consist_loss = nn.MSELoss()(low_R + epsilon, gt_R + epsilon)
#     decom_tv_loss = L_TV()(low_L) * uretinex_opts.tv_weight
#     total_decom_loss = decom_recon_loss + 0.5 * decom_consist_loss + decom_tv_loss
#
#     # 2. Noise2noise去噪损失
#     denoise_loss = nn.MSELoss()(denoise_R + epsilon, gt_R + epsilon)
#
#     # 3. Zero-DCE照明增强损失
#     L_normal = torch.mean(x_normal, dim=1, keepdim=True)
#     # 修复：确保照明层在合理范围内
#     L_normal = torch.clamp(L_normal, 0.0, 1.0)
#     enhance_L1 = torch.clamp(enhance_L1, 0.0, 1.0)
#
#     exp_loss = L_exp(patch_size=16, mean_val=0.5)(enhance_L1)
#     spa_loss = torch.mean(L_spa()(L_normal, enhance_L1))
#     color_loss = L_color()(enhance_L)
#     tv_loss = L_TV()(enhance_L1)
#     total_illum_loss = exp_loss + 0.01 * spa_loss + 0.01 * color_loss + 0.01 * tv_loss
#
#     # 4. 最终重建损失
#     recon_final_loss = nn.L1Loss()(enhance_img + epsilon, x_normal + epsilon)
#
#     # 总损失计算
#     total_loss = (0.2 * total_decom_loss + 0.2 * denoise_loss +
#                   0.3 * total_illum_loss + 0.3 * recon_final_loss)
#
#     # 修复：检查损失是否为NaN
#     if torch.isnan(total_loss).any():
#         print("警告：检测到NaN损失，使用备用损失")
#         # 使用简单的L1损失作为备用
#         total_loss = nn.L1Loss()(enhance_img, x_normal)
#         total_decom_loss = torch.tensor(0.0).to(total_loss.device)
#         denoise_loss = torch.tensor(0.0).to(total_loss.device)
#         total_illum_loss = torch.tensor(0.0).to(total_loss.device)
#         recon_final_loss = total_loss
#
#     return total_loss, total_decom_loss, denoise_loss, total_illum_loss, recon_final_loss
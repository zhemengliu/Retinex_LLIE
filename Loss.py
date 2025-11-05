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

        # 添加数值稳定性
        org = torch.clamp(org, 0.0, 1.0)
        enhance = torch.clamp(enhance, 0.0, 1.0)

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        # 改进权重计算，增加数值稳定性
        weight_diff = torch.clamp(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            min=0.1, max=10.0  # 限制权重范围
        )

        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        # 梯度计算
        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
        D_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        # 梯度差计算，增加稳定性
        D_left = torch.pow(torch.clamp(D_org_left - D_enhance_left, -10, 10), 2)
        D_right = torch.pow(torch.clamp(D_org_right - D_enhance_right, -10, 10), 2)
        D_up = torch.pow(torch.clamp(D_org_up - D_enhance_up, -10, 10), 2)
        D_down = torch.pow(torch.clamp(D_org_down - D_enhance_down, -10, 10), 2)

        E = torch.mean(D_left + D_right + D_up + D_down)
        return torch.clamp(E, 0.0, 1.0)  # 限制损失范围


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
    """改进的总损失函数，平衡各项损失"""

    # 1. 分解重建损失 - 降低权重
    decom_recon_loss = nn.MSELoss()(low_R * low_L, x_low) + nn.MSELoss()(gamma_R * gamma_L, x_gamma)

    # 2. 反射层一致性损失 - 保持
    decom_consist_loss = nn.MSELoss()(low_R, gamma_R)

    # 3. 光照一致性损失 - 改进计算方式
    low_L0 = torch.max(x_low, dim=1, keepdim=True)[0]
    gamma_L0 = torch.max(x_gamma, dim=1, keepdim=True)[0]
    L_con = nn.MSELoss()(low_L0, low_L) + nn.MSELoss()(gamma_L0, gamma_L)

    # 4. 梯度损失 - 增强纹理保持
    r_gradient_loss = L_spa()(low_R, x_low) + L_spa()(gamma_R, x_gamma)

    # # 5. 新增：反射层纹理损失
    # texture_loss = torch.mean(torch.abs(F.conv2d(low_R, torch.ones(1, 1, 3, 3).to(low_R.device) / 9) - low_R))

    # 平衡的分解损失权重
    total_decom_loss = (
            0.1 * decom_recon_loss +  # 降低重建损失权重
            5.0 * decom_consist_loss +  # 保持一致性
            0.02 * L_con +  # 降低光照一致性权重
            0.2 * r_gradient_loss # 保持梯度损失
            # 0.1 * texture_loss  # 新增纹理损失
    )

    print(f'decom_recon_loss:{0.1 * decom_recon_loss:.4f}, decom_consist_loss: {5.0 * decom_consist_loss:.4f}, '
          f'L_con:{0.02 * L_con:.4f}, r_gradient_loss:{0.2 * r_gradient_loss:.4f}')#, texture_loss:{0.1 * texture_loss:.4f}

    # 6. 照明增强损失 - 提高权重
    exp_loss = L_exp(patch_size=16, mean_val=0.6)(enhance_L)
    color_loss = L_color()(enhance_img)  # 对最终增强图像计算颜色损失

    total_illum_loss = 0.5 * exp_loss + 0.1 * color_loss

    # 7. 最终总损失 - 重新平衡
    total_loss = 0.2 * total_decom_loss + 0.8 * total_illum_loss

    return total_loss, total_decom_loss, total_illum_loss
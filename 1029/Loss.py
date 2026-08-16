import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models


class LPIPS(nn.Module):
    """学习感知图像块相似度"""

    def __init__(self):
        super(LPIPS, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.vgg.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # 归一化到ImageNet统计量
        x = (x - 0.5) * 2
        y = (y - 0.5) * 2

        # 提取VGG特征
        feats_x = []
        feats_y = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in [3, 8, 15, 22]:  # relu1_2, relu2_2, relu3_3, relu4_3
                feats_x.append(x)
                feats_y.append(y)

        # 计算LPIPS
        lpips_loss = 0
        for fx, fy in zip(feats_x, feats_y):
            lpips_loss += F.mse_loss(fx, fy)

        return lpips_loss / len(feats_x)


class RetinexLoss(nn.Module):
    def __init__(self):
        super(RetinexLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lpips = LPIPS()

    def forward(self, low_img, R_low, L_low, R_gamma, L_gamma, enhanced_img):
        # 重建损失
        recon_loss_low = self.l1_loss(R_low * L_low, low_img)

        # 反射一致性损失 (无监督核心)
        reflect_consistency_loss = self.l1_loss(R_low, R_gamma)

        # 照明平滑损失
        illumination_smooth_loss = self.smoothness_loss(L_low)

        # 增强图像质量损失
        enhance_quality_loss = self.enhance_quality_loss(enhanced_img)

        # 总损失
        total_loss = (recon_loss_low +
                      reflect_consistency_loss * 0.8 +
                      illumination_smooth_loss * 0.1 +
                      enhance_quality_loss * 0.5)

        return total_loss, {
            'recon_loss': recon_loss_low.item(),
            'reflect_consistency_loss': reflect_consistency_loss.item(),
            'illumination_smooth_loss': illumination_smooth_loss.item(),
            'enhance_quality_loss': enhance_quality_loss.item(),
            'total_loss': total_loss.item()
        }

    def smoothness_loss(self, illumination):
        """照明层平滑损失"""
        dx = torch.abs(illumination[:, :, :, :-1] - illumination[:, :, :, 1:])
        dy = torch.abs(illumination[:, :, :-1, :] - illumination[:, :, 1:, :])
        return torch.mean(dx) + torch.mean(dy)

    def enhance_quality_loss(self, enhanced_img):
        """增强图像质量损失"""
        # 曝光控制损失
        exposure_loss = torch.mean(torch.abs(enhanced_img - 0.6))

        # 颜色恒常性损失
        r, g, b = torch.split(enhanced_img, 1, dim=1)
        color_loss = torch.mean(torch.abs(r - g) + torch.abs(g - b) + torch.abs(b - r))

        return exposure_loss + color_loss * 0.5


class DenoiseLoss(nn.Module):
    """自监督去噪损失 (Noise2Noise)"""

    def __init__(self):
        super(DenoiseLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, denoised_R, noisy_R1, noisy_R2):
        # 两个独立噪声样本的损失
        loss1 = self.l1_loss(denoised_R, noisy_R1)
        loss2 = self.l1_loss(denoised_R, noisy_R2)
        return (loss1 + loss2) * 0.5
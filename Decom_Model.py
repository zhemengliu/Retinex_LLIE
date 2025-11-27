import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------- 基础工具函数 --------------------------
def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                     out_channels=out_c,
                     kernel_size=k,
                     stride=s,
                     padding=p, dilation=dilation, groups=groups)


def get_deconv2d_layer(in_c, out_c, k=1, s=1, p=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=s,
            padding=p
        )
    )


# -------------------------- 关系学习Transformer模块 --------------------------
class IlluminationRelationTransformer(nn.Module):
    """
    专门用于照明比值关系学习的Transformer模块
    输入: [B, C, H, W] 的特征图
    输出: [B, C, H, W] 的关系增强特征
    """

    def __init__(self, dim=256, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        assert self.head_dim * heads == dim, "dim必须能被heads整除"

        # 线性变换
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Dropout(dropout)
        )

        # 位置编码 - 相对位置偏置
        self.rel_pos_bias = nn.Parameter(torch.randn(heads, 49, 49) * 0.02)  # 7x7窗口

        # 局部窗口大小
        self.window_size = 7

        # 层归一化
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_orig = x

        # 层归一化 (在通道维度)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # 生成Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=1)  # 3 * [B, C, H, W]
        q, k, v = map(lambda t: t.reshape(
            B, self.heads, self.head_dim, H * W
        ), qkv)  # [B, heads, head_dim, H*W]

        # 计算注意力分数 (照明关系权重)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('b h d i, b h d j -> b h i j', q, k) * scale  # [B, heads, H*W, H*W]

        # 添加相对位置偏置 (编码空间邻近关系)
        if H * W == self.rel_pos_bias.shape[1]:
            attn = attn + self.rel_pos_bias.unsqueeze(0)  # [B, heads, H*W, H*W]

        # 注意力权重归一化
        attn = F.softmax(attn, dim=-1)

        # 应用注意力到Value
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)  # [B, heads, H*W, head_dim]
        out = out.reshape(B, self.heads, H, W, self.head_dim)
        out = out.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)  # [B, C, H, W]

        # 输出投影 + 残差连接
        out = self.to_out(out)
        out = out + x_orig  # 残差连接

        return out


# -------------------------- 多尺度融合模块 --------------------------
class UpsampleFuser(nn.Module):
    """融合上采样比值图与当前尺度特征的模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels = 当前特征通道数 + 比值图通道数(1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fusion_conv(x)


# -------------------------- 多尺度主干网络 --------------------------
class MultiScaleBackbone(nn.Module):
    """多尺度特征提取主干网络"""

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # 下采样路径 (编码器)
        self.enc1 = nn.Sequential(
            get_conv2d_layer(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            get_conv2d_layer(base_channels, base_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            get_conv2d_layer(base_channels * 2, base_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            get_conv2d_layer(base_channels * 4, base_channels * 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(base_channels * 8, base_channels * 8, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 提取多尺度特征
        f1 = self.enc1(x)  # 1/1尺度
        f2 = self.enc2(f1)  # 1/2尺度
        f3 = self.enc3(f2)  # 1/4尺度
        f4 = self.enc4(f3)  # 1/8尺度

        return [f4, f3, f2, f1]  # 从粗糙到精细


# -------------------------- 核心照明比值估计网络 --------------------------
class IlluminationRatioStarNet(nn.Module):
    """
    基于多尺度比值渐进细化的照明估计网络
    核心思想：从粗糙到精细逐步估计照明比值
    """

    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        # 1. 多尺度特征提取主干
        self.backbone = MultiScaleBackbone(in_channels=3, base_channels=32)

        # 2. 金字塔处理流 - 4个尺度 [1/8, 1/4, 1/2, 1]
        self.pyramid_levels = [1 / 8, 1 / 4, 1 / 2, 1]
        feature_dims = [256, 128, 64, 32]  # 对应每个尺度的特征维度

        # 3. 关系学习Transformer模块 (每个尺度一个)
        self.relation_transformers = nn.ModuleList([
            IlluminationRelationTransformer(dim=dim, heads=8)
            for dim in feature_dims
        ])

        # 4. 渐进上采样融合模块
        self.upsample_fusers = nn.ModuleList([
            UpsampleFuser(in_dim + 1, out_dim)  # +1 用于比值图通道
            for in_dim, out_dim in zip([256 + 1, 128 + 1, 64 + 1], [128, 64, 32])
        ])

        # 5. 比值预测头 (每个尺度一个)
        self.ratio_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, padding=1),
                nn.Sigmoid()  # 输出0-1范围的比值
            ) for dim in feature_dims
        ])

        # 6. 反射率估计网络 (可选)
        self.reflectance_net = nn.Sequential(
            get_conv2d_layer(3 + 1, 64, 3, 1, 1),  # 输入: 图像 + 照明比值
            nn.ReLU(inplace=True),
            get_conv2d_layer(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_img):
        """
        前向传播: 多尺度渐进比值估计
        输入: input_img [B, 3, H, W]
        输出: final_ratio_map [B, 1, H, W], reflectance [B, 3, H, W]
        """
        # 1. 提取多尺度特征 [1/8, 1/4, 1/2, 1]
        features = self.backbone(input_img)  # List[Tensor], 从粗糙到精细

        pyramid_outputs = []
        current_ratio = None

        # 2. 从最粗糙尺度到最精细尺度处理
        for i, (feat, scale) in enumerate(zip(features, self.pyramid_levels)):
            # 2.1 应用关系学习Transformer (路径A: 特征提取)
            relational_feat = self.relation_transformers[i](feat)

            if i == 0:  # 最粗糙尺度 (1/8)
                # 直接从特征预测初始比值图
                current_ratio = self.ratio_heads[i](relational_feat)
            else:  # 精细尺度 (1/4, 1/2, 1)
                # 2.2 路径B: 将上一尺度的比值图上采样到当前尺度
                upsampled_ratio = F.interpolate(
                    current_ratio,
                    size=relational_feat.shape[2:],  # 匹配当前尺度空间尺寸
                    mode='bilinear',
                    align_corners=False
                )

                # 2.3 双路径融合: 上采样的粗糙比值 + 当前尺度的关系特征
                fused_feat = self.upsample_fusers[i - 1](
                    torch.cat([relational_feat, upsampled_ratio], dim=1)
                )

                # 2.4 预测相对于上采样比值的残差
                residual_ratio = self.ratio_heads[i](fused_feat)

                # 2.5 更新当前比值: 上采样基础 × 残差修正
                current_ratio = upsampled_ratio * (1 + residual_ratio)

            pyramid_outputs.append(current_ratio)

        # 3. 最终照明比值图
        final_ratio_map = current_ratio

        # 4. 可选: 估计反射率
        ratio_map_full = F.interpolate(final_ratio_map, size=input_img.shape[2:], mode='bilinear')
        reflectance = self.reflectance_net(torch.cat([input_img, ratio_map_full], dim=1))

        return final_ratio_map, reflectance, pyramid_outputs


# -------------------------- 多尺度损失函数 --------------------------
class MultiScaleIlluminationLoss(nn.Module):
    """多尺度照明比值损失函数"""

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # 最终输出损失权重
        self.beta = beta  # 多尺度监督权重
        self.gamma = gamma  # 一致性损失权重

        self.l1_loss = nn.L1Loss()

    def forward(self, pred_ratio, pred_pyramid, target_ratio=None, target_illumination=None):
        """
        计算多尺度损失
        如果target_ratio为None，则使用自监督损失
        """
        total_loss = 0

        if target_ratio is not None:
            # 有监督训练: 直接比较比值图
            # 1. 最终输出loss
            total_loss += self.alpha * self.l1_loss(pred_ratio, target_ratio)

            # 2. 多尺度辅助loss
            for i, ratio in enumerate(pred_pyramid):
                target_down = F.interpolate(target_ratio, size=ratio.shape[2:], mode='bilinear')
                scale_weight = 0.5 / (2 ** i)  # 粗糙尺度权重递减
                total_loss += self.beta * scale_weight * self.l1_loss(ratio, target_down)

        else:
            # 自监督训练: 使用物理约束
            # 这里可以添加重构损失、平滑约束等
            pass

        # 3. 跨尺度一致性loss
        for i in range(len(pred_pyramid) - 1):
            ratio_up = F.interpolate(pred_pyramid[i],
                                     size=pred_pyramid[i + 1].shape[2:],
                                     mode='bilinear')
            total_loss += self.gamma * self.l1_loss(ratio_up, pred_pyramid[i + 1])

        return total_loss


# -------------------------- 测试代码 --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Scale Illumination Ratio Estimation")
    parser.add_argument("--base_channels", type=int, default=32, help="基础通道数")
    parser.add_argument("--unfolding_round", type=int, default=3, help="保留参数，兼容性")
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建测试数据
    batch_size, channels, height, width = 2, 3, 256, 256
    test_input = torch.randn(batch_size, channels, height, width).to(device)

    # 创建网络
    net = IlluminationRatioStarNet(args).to(device)

    # 测试前向传播
    with torch.no_grad():
        final_ratio, reflectance, pyramid_outputs = net(test_input)

    # 输出形状检查
    print(f"输入形状: {test_input.shape}")
    print(f"最终比值图形状: {final_ratio.shape}")
    print(f"反射率图形状: {reflectance.shape}")
    print(f"金字塔输出层数: {len(pyramid_outputs)}")

    for i, ratio_map in enumerate(pyramid_outputs):
        scale = 1 / (2 ** (3 - i))  # 1/8, 1/4, 1/2, 1
        print(f"金字塔层 {i + 1} (尺度 {scale}): {ratio_map.shape}")

    # 测试损失函数
    loss_fn = MultiScaleIlluminationLoss()
    # 模拟目标比值图 (在实际训练中需要真实数据)
    target_ratio = torch.rand_like(final_ratio)
    loss = loss_fn(final_ratio, pyramid_outputs, target_ratio=target_ratio)
    print(f"多尺度损失值: {loss.item():.4f}")

    print("网络测试完成！架构符合多尺度比值估计设计。")
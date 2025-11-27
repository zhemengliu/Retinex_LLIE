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
import os
import argparse
import time
import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pytorch_msssim import ssim as ssim_torch
from MyDataset import *  # 确保MyDataset、LOLDataset定义正确
from Decom_Model import *
# 新增：导入模型大小测量工具
from thop import profile
from torchsummary import summary
# 在训练脚本开头添加导入
import sys
sys.path.append('./')  # 根据实际路径调整



# -------------------------- 数据加载器设置 --------------------------
def setup_data_loaders(opts):
    """设置训练和验证数据加载器（基于你的MyDataset）"""

    # 数据集路径（你提供的地址）
    train_folder = [
        "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low\\",
        "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high\\"
    ]

    eval_folder = [
        "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low\\",
        "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high\\"
    ]

    # 创建训练数据集 - 使用你的MyDataset类
    train_Data = []
    for patch_id in range(opts.num_patches):  # 原代码使用10个patch增强
        rand_mode = np.random.randint(0, 7)  # 随机数据增强模式
        train_data = MyDataset(
            rand_mode=rand_mode,
            patch_size=opts.patch_size,
            folder=train_folder
        )
        train_Data.extend(train_data)

    print('[*] Number of training data: %d' % len(train_Data))

    # 创建验证数据集 - 固定rand_mode避免随机性
    eval_Data = MyDataset(
        rand_mode=0,  # 验证时不使用数据增强
        patch_size=opts.patch_size,
        folder=eval_folder
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_Data,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        drop_last=True
    )

    eval_loader = DataLoader(
        dataset=eval_Data,
        batch_size=1,  # 评估时batch_size=1
        shuffle=False,
        num_workers=opts.num_workers,
        drop_last=False
    )

    return train_loader, eval_loader


def train_illumination_net(opts):
    """训练照明比值估计网络"""

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = IlluminationRatioStarNet(opts).to(device)

    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=0.1)
    criterion = MultiScaleIlluminationLoss()

    # 设置数据加载器
    train_loader, eval_loader = setup_data_loaders(opts)

    print(f"训练图像数量: {len(train_loader.dataset)}")
    print(f"验证图像数量: {len(eval_loader.dataset)}")

    # 训练循环
    for epoch in range(opts.epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, batch_data in enumerate(train_loader):
            # 你的MyDataset返回 (name, input_low, input_high)
            names, low_imgs, normal_imgs = batch_data
            low_imgs = low_imgs.to(device)
            normal_imgs = normal_imgs.to(device)

            # 前向传播
            ratio_map, reflectance, pyramid_outputs = model(low_imgs)

            # 计算目标比值图 - 基于正常光/低光的亮度比值
            with torch.no_grad():
                # 计算亮度通道 (RGB转亮度)
                low_brightness = 0.299 * low_imgs[:, 0:1, :, :] + 0.587 * low_imgs[:, 1:2, :, :] + 0.114 * low_imgs[:,
                                                                                                           2:3, :, :]
                normal_brightness = 0.299 * normal_imgs[:, 0:1, :, :] + 0.587 * normal_imgs[:, 1:2, :,
                                                                                :] + 0.114 * normal_imgs[:, 2:3, :, :]

                target_ratio = normal_brightness / (low_brightness + 1e-8)
                target_ratio = torch.clamp(target_ratio, 0.1, 10.0)  # 限制范围

            # 计算损失
            loss = criterion(ratio_map, pyramid_outputs, target_ratio=target_ratio)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f} '
                      f'Ratio Range: [{ratio_map.min():.3f}, {ratio_map.max():.3f}]')

        # 学习率调度
        scheduler.step()

        # 评估
        if epoch % opts.eval_interval == 0:
            eval_loss = evaluate_model(model, eval_loader, criterion, device)
            print(f'Epoch: {epoch}, Train Loss: {epoch_loss / len(train_loader):.6f}, '
                  f'Eval Loss: {eval_loss:.6f}')

        # 保存检查点
        if epoch % opts.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(train_loader),
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
            print(f'检查点已保存: checkpoint_epoch_{epoch}.pth')


def evaluate_model(model, eval_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_data in eval_loader:
            names, low_imgs, normal_imgs = batch_data
            low_imgs = low_imgs.to(device)
            normal_imgs = normal_imgs.to(device)

            ratio_map, reflectance, pyramid_outputs = model(low_imgs)

            # 计算目标比值
            low_brightness = 0.299 * low_imgs[:, 0:1, :, :] + 0.587 * low_imgs[:, 1:2, :, :] + 0.114 * low_imgs[:, 2:3,
                                                                                                       :, :]
            normal_brightness = 0.299 * normal_imgs[:, 0:1, :, :] + 0.587 * normal_imgs[:, 1:2, :,
                                                                            :] + 0.114 * normal_imgs[:, 2:3, :, :]

            target_ratio = normal_brightness / (low_brightness + 1e-8)
            target_ratio = torch.clamp(target_ratio, 0.1, 10.0)

            loss = criterion(ratio_map, pyramid_outputs, target_ratio=target_ratio)
            total_loss += loss.item()

    return total_loss / len(eval_loader)


# 训练参数配置
class TrainOptions:
    def __init__(self):
        self.batch_size = 8
        self.lr = 1e-4
        self.epochs = 100
        self.lr_decay_step = 30
        self.weight_decay = 1e-5
        self.eval_interval = 5
        self.save_interval = 10
        self.base_channels = 32
        self.patch_size = 256  # 匹配你的数据集
        self.num_patches = 10  # 原代码使用10个patch增强
        self.num_workers = 4  # 数据加载线程数


# 主训练函数
def main():
    opts = TrainOptions()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("开始训练多尺度照明比值估计网络...")
    print(f"训练配置: batch_size={opts.batch_size}, lr={opts.lr}, epochs={opts.epochs}")
    print(f"数据配置: patch_size={opts.patch_size}, num_patches={opts.num_patches}")

    train_illumination_net(opts)


if __name__ == '__main__':
    main()
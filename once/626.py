import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import lpips
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径配置
TEST_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high"
TRAIN_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"
TRAIN_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"
TEST_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 超参数
BATCH_SIZE = 8
IMAGE_SIZE = 256
EPOCHS = 1
LR = 1e-4

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


def apply_histeq(image):
    """使用PIL实现直方图均衡化"""
    # 转换为YCbCr颜色空间
    ycbcr = image.convert('YCbCr')
    y, cb, cr = ycbcr.split()

    # 对亮度通道进行直方图均衡化
    y_eq = ImageOps.equalize(y)

    # 合并通道并转回RGB
    ycbcr_eq = Image.merge('YCbCr', (y_eq, cb, cr))
    return ycbcr_eq.convert('RGB')


class LowLightDataset(Dataset):
    def __init__(self, low_dir, normal_dir=None, transform=None, use_histeq=True):
        self.low_files = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir)])
        self.normal_files = sorted(
            [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]) if normal_dir else None
        self.transform = transform
        self.use_histeq = use_histeq

        # 预计算直方图均衡化结果（训练集）
        if self.use_histeq and normal_dir is None:
            self.histeq_images = []
            for low_file in self.low_files:
                low_img = Image.open(low_file).convert('RGB')
                histeq_img = apply_histeq(low_img)
                self.histeq_images.append(histeq_img)

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_files[idx]).convert('RGB')

        # 使用直方图均衡化图像作为"正常光"目标
        if self.use_histeq and self.normal_files is None:
            normal_img = self.histeq_images[idx]
        elif self.normal_files:
            normal_img = Image.open(self.normal_files[idx]).convert('RGB')
        else:
            normal_img = low_img.copy()

        if self.transform:
            seed = torch.randint(0, 100000, (1,)).item()
            torch.manual_seed(seed)
            low_img = self.transform(low_img)
            torch.manual_seed(seed)
            normal_img = self.transform(normal_img)

        return low_img, normal_img


# ========== 改进的分解网络 ==========
class DecomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1)
        )

        self.dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 4, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        feat = self.enc(x)
        output = self.dec(feat)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        return R, L


# ========== 改进的增强网络 ==========
class EnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 光照调整模块
        self.illum_adjust = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 反射率去噪模块
        self.denoise = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

        # 颜色校正模块
        self.color_correct = nn.Sequential(
            nn.Conv2d(3, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, L, R):
        # 光照调整
        L_adjusted = self.illum_adjust(L)

        # 反射率去噪
        R_denoised = self.denoise(R)

        # 初步增强
        enhanced = R_denoised * L_adjusted.repeat(1, 3, 1, 1)

        # 颜色校正
        enhanced = self.color_correct(enhanced)

        return enhanced, R_denoised, L_adjusted


# ========== 训练器 ==========
class RetinexTrainer:
    def __init__(self):
        # 初始化网络
        self.decom = DecomNet().to(device)
        self.enhance = EnhanceNet().to(device)

        # 优化器
        self.optim = torch.optim.Adam(
            list(self.decom.parameters()) + list(self.enhance.parameters()),
            lr=LR
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', factor=0.5, patience=3, verbose=True
        )

        # 数据集 - 训练使用直方图均衡化，测试使用真实正常光
        self.train_loader = DataLoader(
            LowLightDataset(TRAIN_LOW_PATH, None, train_transform, use_histeq=True),
            BATCH_SIZE, shuffle=True, pin_memory=True
        )
        self.test_loader = DataLoader(
            LowLightDataset(TEST_LOW_PATH, TEST_NORMAL_PATH, test_transform, use_histeq=False),
            BATCH_SIZE, shuffle=False, pin_memory=True
        )

        # 评估指标
        self.ssim = SSIM(data_range=1.0).to(device)
        self.psnr = PSNR(data_range=1.0).to(device)
        self.lpips = lpips.LPIPS(net='alex').to(device)
        self.perceptual_loss = lpips.LPIPS(net='alex').to(device)

        # 日志文件
        self.log_file = os.path.join(CHECKPOINT_DIR, "training_log.csv")
        with open(self.log_file, 'w') as f:
            f.write("epoch,train_loss,test_psnr,test_ssim,test_lpips\n")

    def gradient_loss(self, img):
        """计算图像梯度，用于平滑约束"""
        # 对于单通道图像
        if img.shape[1] == 1:
            grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
            grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
            return grad_x.mean() + grad_y.mean()

        # 对于多通道图像
        grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        return grad_x.mean() + grad_y.mean()

    def train_epoch(self, epoch):
        self.decom.train()
        self.enhance.train()

        total_loss = 0.0
        for batch_idx, (low, normal) in enumerate(self.train_loader):
            low = low.to(device)
            normal = normal.to(device)

            self.optim.zero_grad()

            # 前向传播
            R, L = self.decom(low)
            enhanced, R_denoised, L_adjusted = self.enhance(L, R)

            # 损失计算
            # 1. 重建损失
            recon_loss = F.l1_loss(enhanced, normal)

            # 2. 分解一致性损失
            consistency_loss = F.l1_loss(R * L.repeat(1, 3, 1, 1), low)

            # 3. 反射率平滑损失
            reflect_smooth_loss = self.gradient_loss(R)

            # 4. 光照平滑损失
            illum_smooth_loss = self.gradient_loss(L)

            # 5. 感知损失
            perceptual_loss = self.perceptual_loss(enhanced, normal).mean()

            # 总损失
            loss = (
                    recon_loss +
                    0.5 * consistency_loss +
                    0.1 * reflect_smooth_loss +
                    0.1 * illum_smooth_loss +
                    0.05 * perceptual_loss
            )

            # 反向传播
            loss.backward()
            self.optim.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self):
        self.decom.eval()
        self.enhance.eval()

        total_ssim, total_psnr, total_lpips = 0.0, 0.0, 0.0
        with torch.no_grad():
            for low, normal in self.test_loader:
                low = low.to(device)
                normal = normal.to(device)

                R, L = self.decom(low)
                enhanced, _, _ = self.enhance(L, R)
                enhanced = torch.clamp(enhanced, 0, 1)

                # 计算指标
                total_ssim += self.ssim(enhanced, normal)
                total_psnr += self.psnr(enhanced, normal)

                # LPIPS需要[-1,1]范围
                enhanced_scaled = enhanced * 2 - 1
                normal_scaled = normal * 2 - 1
                total_lpips += self.lpips(enhanced_scaled, normal_scaled).mean()

        avg_ssim = total_ssim / len(self.test_loader)
        avg_psnr = total_psnr / len(self.test_loader)
        avg_lpips = total_lpips / len(self.test_loader)

        print(f"Test Metrics | SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.4f} | LPIPS: {avg_lpips:.4f}")
        return avg_psnr, avg_ssim, avg_lpips

    def visualize_results(self, epoch):
        self.decom.eval()
        self.enhance.eval()

        # 随机选择一个测试样本
        test_img_path = os.path.join(TEST_LOW_PATH, random.choice(os.listdir(TEST_LOW_PATH)))
        img = Image.open(test_img_path).convert('RGB')
        low_tensor = test_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            R, L = self.decom(low_tensor)
            enhanced, R_denoised, L_adjusted = self.enhance(L, R)
            enhanced = torch.clamp(enhanced, 0, 1)

            # 转换为numpy用于显示
            def to_numpy(x):
                return x.squeeze(0).permute(1, 2, 0).cpu().numpy()

            low_img = to_numpy(low_tensor)
            R_img = to_numpy(R)
            L_img = to_numpy(L).mean(axis=2, keepdims=True)
            L_adjusted_img = to_numpy(L_adjusted).mean(axis=2, keepdims=True)
            enhanced_img = to_numpy(enhanced)

            # 创建直方图均衡化对比
            pil_low = Image.fromarray((low_img * 255).astype(np.uint8))
            histeq_pil = apply_histeq(pil_low)
            histeq_img = np.array(histeq_pil) / 255.0

            # 绘制结果
            plt.figure(figsize=(20, 10))
            titles = ['Low Light', 'HistEQ', 'Reflectance', 'Illumination', 'Adjusted Illum', 'Enhanced']
            images = [low_img, histeq_img, R_img, L_img, L_adjusted_img, enhanced_img]

            for i in range(6):
                plt.subplot(2, 3, i + 1)
                if i in [3, 4]:  # 光照图为单通道
                    plt.imshow(images[i][:, :, 0], cmap='gray', vmin=0, vmax=1)
                else:
                    plt.imshow(images[i], vmin=0, vmax=1)
                plt.title(titles[i])
                plt.axis('off')

            plt.suptitle(f"Epoch {epoch} Results", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(CHECKPOINT_DIR, f"result_epoch_{epoch}.png"))
            plt.close()

    def train(self):
        best_psnr = 0.0
        for epoch in range(1, EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            test_psnr, test_ssim, test_lpips = self.evaluate()

            # 更新学习率
            self.scheduler.step(train_loss)

            # 保存日志
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{test_psnr:.4f},{test_ssim:.4f},{test_lpips:.4f}\n")

            # 可视化结果
            if epoch % 5 == 0 or epoch == EPOCHS:
                self.visualize_results(epoch)

            # 保存最佳模型
            if test_psnr > best_psnr:
                best_psnr = test_psnr
                torch.save({
                    'decom': self.decom.state_dict(),
                    'enhance': self.enhance.state_dict(),
                    'epoch': epoch,
                    'psnr': test_psnr
                }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
                print(f"Saved best model at epoch {epoch} with PSNR: {test_psnr:.4f}")

            # 定期保存
            if epoch % 10 == 0:
                torch.save({
                    'decom': self.decom.state_dict(),
                    'enhance': self.enhance.state_dict(),
                    'epoch': epoch
                }, os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth"))
                print(f"Saved checkpoint at epoch {epoch}")

            print(f"Epoch {epoch} Complete | Train Loss: {train_loss:.4f} | Test PSNR: {test_psnr:.4f}")


if __name__ == "__main__":
    trainer = RetinexTrainer()
    trainer.train()
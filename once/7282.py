import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径配置（请根据实际路径修改）
TEST_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high"
TRAIN_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"
TRAIN_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"
TEST_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"
# 设置新的检查点保存目录（确保路径存在或可创建）
CHECKPOINT_DIR = "D:\\文件\\大创\\train728"  # 新目录，避免覆盖旧模型

# 超参数配置
BATCH_SIZE = 4
IMAGE_SIZE = 256
EPOCHS = 100
LR = 3e-4
ITERATION_NUM = 3  # P/Q迭代次数
GAMMA = 0.1  # P模块正则化参数
LAMDA = 0.1  # Q模块正则化参数

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# 数据集类
class LowLightDataset(Dataset):
    def __init__(self, low_dir, normal_dir=None, transform=None):
        self.low_files = sorted(
            [os.path.join(low_dir, f) for f in os.listdir(low_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.normal_files = sorted(
            [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if
             f.endswith(('png', 'jpg', 'jpeg'))]) if normal_dir else None
        self.transform = transform

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_files[idx]).convert('RGB')
        normal_img = Image.open(self.normal_files[idx]).convert('RGB') if self.normal_files else low_img.copy()

        if self.transform:
            # 保证低光和正常图像使用相同的数据增强
            seed = torch.random.seed()
            torch.manual_seed(seed)
            low_img = self.transform(low_img)
            torch.manual_seed(seed)
            normal_img = self.transform(normal_img)
        return low_img, normal_img


# 工具函数：定义卷积层
def get_conv2d_layer(in_c, out_c, k, s, p):
    """创建卷积层（含BatchNorm）"""
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=s, padding=p),
        nn.BatchNorm2d(out_c)
    )


# 新的Retinex分解网络
class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
            nn.ReLU()  # 确保输出非负
        )

    def forward(self, input):
        """输入：3通道图像；输出：反射分量R(3通道)和照明分量L(1通道)"""
        output = self.decom(input)
        R = output[:, 0:3, :, :]  # 反射分量（3通道）
        L = output[:, 3:4, :, :]  # 照明分量（1通道）
        # 归一化到[0,1]范围
        R = torch.sigmoid(R)
        L = torch.sigmoid(L)
        return R, L


# P迭代模块：优化反射分量
class P(nn.Module):
    """
    求解 min(P) = ||I-PQ||^2 + γ||P-R||^2
    解析解：P* = (γ*R + I*Q) / (γ + Q²)
    """

    def __init__(self):
        super().__init__()

    def forward(self, I, Q, R, gamma):
        """
        I: 输入图像(3通道)
        Q: 照明相关特征(1通道)
        R: 初始反射分量(3通道)
        gamma: 正则化参数
        """
        return ((I * Q + gamma * R) / (gamma + Q * Q)).clamp(0, 1)  # 确保输出在有效范围


# Q迭代模块：优化照明分量
class Q(nn.Module):
    """
    求解 min(Q) = ||I-PQ||^2 + λ||Q-L||^2
    解析解：Q* = (λ*L + I·P) / (P·P + λ)
    """

    def __init__(self):
        super().__init__()

    def forward(self, I, P, L, lamda):
        """
        I: 输入图像(3通道)
        P: 反射相关特征(3通道)
        L: 初始照明分量(1通道)
        lamda: 正则化参数
        """
        # 分离RGB通道
        IR, IG, IB = I[:, 0:1, :, :], I[:, 1:2, :, :], I[:, 2:3, :, :]
        PR, PG, PB = P[:, 0:1, :, :], P[:, 1:2, :, :], P[:, 2:3, :, :]

        # 计算分子和分母
        numerator = (IR * PR + IG * PG + IB * PB) + lamda * L
        denominator = (PR * PR + PG * PG + PB * PB) + lamda
        return (numerator / denominator).clamp(0, 1)  # 确保输出在有效范围


# 基础卷积块（含BN和激活）
class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, pad=1, use_act=True):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=pad)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)  # 保持与原网络一致的LeakyReLU激活

    def forward(self, x):
        x = self.bn(self.conv(x))  # 卷积+批归一化
        if self.use_act:
            x = self.act(x)  # 激活（按需使用）
        return x


# 残差块（核心特征学习单元）
class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResBlock, self).__init__()
        # 两个卷积块构成残差单元（第一个带激活，第二个不带）
        self.block1 = ConvBlock(input_channels, output_channels, kernel_size)
        self.block2 = ConvBlock(input_channels, output_channels, kernel_size, use_act=False)

    def forward(self, x):
        # 残差连接：输入直接与特征提取结果相加（增强梯度传播）
        return x + self.block2(self.block1(x))


# 基于SRResnet的去噪网络（替换原U-Net）
class DenoiseNet(nn.Module):
    """用于反射分量去噪的残差网络（替换原U-Net结构）"""
    def __init__(self, in_channels=3, features=64, res_layers=16):
        super().__init__()
        # 初始卷积（将输入映射到特征空间）
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)  # 激活函数与原网络保持一致

        # 残差层堆叠（核心去噪特征学习）
        self.res_blocks = nn.Sequential(
            *[ResBlock(features, features, kernel_size=3) for _ in range(res_layers)]
        )

        # 残差连接后的卷积（融合特征）
        self.conv2 = ConvBlock(features, features, kernel_size=3, use_act=False)

        # 输出层（将特征映射回3通道）
        self.final_conv = nn.Conv2d(features, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        输入：3通道反射分量（[0,1]范围张量）
        输出：3通道去噪后反射分量（[0,1]范围张量）
        """
        # 初始特征提取
        x1 = self.act(self.conv1(x))

        # 残差块特征学习 + 跳跃连接（残差融合）
        res_out = self.res_blocks(x1)
        x2 = self.conv2(res_out)
        x3 = x1 + x2  # 跳跃连接增强特征保留

        # 输出3通道结果并约束在[0,1]范围（与原网络一致）
        out = self.final_conv(x3)
        return torch.sigmoid(out)  # 确保输出在[0,1]，匹配原网络输出范围


# 光照增强模块（S曲线调整）
class SCurveEstimator(nn.Module):
    """估计S曲线参数，调整照明分量"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 输出3个参数
        )

    def forward(self, L):
        """输入：1通道照明分量；输出：调整后的照明分量"""
        params = self.fc(L)
        # 参数约束
        a = torch.sigmoid(params[:, 0]) * 2 + 0.5  # 缩放因子 (0.5, 2.5)
        b = torch.sigmoid(params[:, 1]) * 5 + 1  # 曲率因子 (1, 6)
        c = params[:, 2] * 0.1  # 偏移量

        # 应用S曲线调整
        B, _, H, W = L.shape
        L_adjusted = a.view(B, 1, 1, 1) * torch.sigmoid(
            b.view(B, 1, 1, 1) * (L + c.view(B, 1, 1, 1))
        )
        return torch.clamp(L_adjusted, 0.1, 1)  # 确保在有效范围


# 增强网络（整合去噪和光照调整）
class EnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise = DenoiseNet()  # 反射分量去噪
        self.s_curve = SCurveEstimator()  # 照明分量增强

    def forward(self, L, R):
        """
        输入：照明分量L(1通道)、反射分量R(3通道)
        输出：增强图像、去噪反射分量、调整后照明分量
        """
        R_denoised = self.denoise(R)  # 反射去噪
        L_adjusted = self.s_curve(L)  # 照明增强
        enhanced = R_denoised * L_adjusted.repeat(1, 3, 1, 1)  # 重构增强图像
        return enhanced, R_denoised, L_adjusted


# 训练器类
class RetinexTrainer:
    def __init__(self, resume_checkpoint=None):
        # 初始化网络
        self.decom = Decom().to(device)  # 新的分解网络
        self.enhance = EnhanceNet().to(device)
        self.p_module = P()  # P迭代模块
        self.q_module = Q()  # Q迭代模块

        # 优化器
        self.optim = torch.optim.Adam(
            list(self.decom.parameters()) + list(self.enhance.parameters()),
            lr=LR
        )

        # 数据加载
        self.train_loader = DataLoader(
            LowLightDataset(TRAIN_LOW_PATH, TRAIN_NORMAL_PATH, train_transform),
            BATCH_SIZE, shuffle=True, num_workers=2
        )
        self.test_loader = DataLoader(
            LowLightDataset(TEST_LOW_PATH, TEST_NORMAL_PATH, transform=test_transform),
            BATCH_SIZE, shuffle=False, num_workers=2
        )

        # 评估指标
        self.ssim = SSIM(data_range=1.0).to(device)
        self.psnr = PSNR(data_range=1.0).to(device)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()

        # 日志和检查点配置
        self.log_file = os.path.join(CHECKPOINT_DIR, "training_log.csv")
        self.best_ssim = 0.0
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # 自动创建新目录（如果不存在）
        self.start_epoch = 1

        # 加载检查点（如果需要）
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            # 加载检查点时添加weights_only=True以消除警告
            checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=True)

            # 处理分解网络的权重不匹配问题
            if 'decom' in checkpoint:
                try:
                    # 尝试直接加载
                    self.decom.load_state_dict(checkpoint['decom'])
                except RuntimeError:
                    # 处理新旧网络结构不匹配的情况
                    print("检测到网络结构变化，尝试转换旧权重...")
                    old_decom_weights = checkpoint['decom']
                    new_decom_weights = self.decom.state_dict()

                    # 只加载名称匹配的权重
                    matched_weights = {}
                    for new_key in new_decom_weights.keys():
                        # 简单的名称映射规则，根据实际情况调整
                        if "decom.0.0." in new_key:
                            old_key = new_key.replace("decom.0.0.", "conv0.")
                        elif "decom.2.0." in new_key:
                            old_key = new_key.replace("decom.2.0.", "convs.0.")
                        elif "decom.4.0." in new_key:
                            old_key = new_key.replace("decom.4.0.", "convs.2.")
                        elif "decom.6.0." in new_key:
                            old_key = new_key.replace("decom.6.0.", "recon.")
                        else:
                            old_key = None

                        if old_key in old_decom_weights and new_decom_weights[new_key].shape == old_decom_weights[
                            old_key].shape:
                            matched_weights[new_key] = old_decom_weights[old_key]
                            print(f"成功映射权重: {old_key} -> {new_key}")

                    # 加载匹配的权重
                    new_decom_weights.update(matched_weights)
                    self.decom.load_state_dict(new_decom_weights, strict=False)
                    print("分解网络权重加载完成（部分权重不匹配，已忽略）")

            # 加载增强网络权重
            if 'enhance' in checkpoint:
                try:
                    self.enhance.load_state_dict(checkpoint['enhance'])
                except RuntimeError as e:
                    print(f"增强网络权重加载警告: {e}")

            # 恢复训练进度
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"加载检查点（epoch {checkpoint['epoch']}），从epoch {self.start_epoch} 继续训练")
            else:
                print("检查点无epoch信息，从epoch 1开始训练")

            # 加载优化器状态（如果有）
            if 'optim' in checkpoint:
                try:
                    self.optim.load_state_dict(checkpoint['optim'])
                except:
                    print("优化器状态加载失败，使用新的优化器")

        # 初始化日志文件
        if self.start_epoch == 1:
            with open(self.log_file, "w") as f:
                f.write("epoch,train_loss,ssim,psnr,lpips\n")
        else:
            print(f"日志文件将从epoch {self.start_epoch} 继续记录")

    def iterative_optimize(self, I, R_init, L_init, iterations=ITERATION_NUM):
        """
        迭代优化反射和照明分量
        I: 输入图像（低光图像）
        R_init: 初始反射分量（来自Decom网络）
        L_init: 初始照明分量（来自Decom网络）
        iterations: 迭代次数
        """
        R = R_init.clone()
        L = L_init.clone()

        for _ in range(iterations):
            # 交替优化P（R）和Q（L）
            R = self.p_module(I, L, R, GAMMA)  # 用当前L优化R
            L = self.q_module(I, R, L, LAMDA)  # 用优化后的R再优化L

        return R, L

    def smooth_loss(self, x):
        """平滑损失（减少边缘噪声）"""
        grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        return torch.mean(grad_x) + torch.mean(grad_y)

    def train_epoch(self, epoch):
        self.decom.train()
        self.enhance.train()
        total_loss = 0.0

        for batch_idx, (low, normal) in enumerate(self.train_loader):
            low = low.to(device)
            normal = normal.to(device)

            self.optim.zero_grad()

            # 1. 分解得到初始R和L
            R_init, L_init = self.decom(low)

            # 2. 迭代优化R和L
            R_opt, L_opt = self.iterative_optimize(low, R_init, L_init)

            # 3. 增强图像
            enhanced, R_denoised, L_adjusted = self.enhance(L_opt, R_opt)

            # 4. 计算损失
            # 重构损失（确保分解合理性）
            recon_loss = F.l1_loss(R_opt * L_opt.repeat(1, 3, 1, 1), low)

            # 增强损失（与正常图像的差异）
            enhance_loss = F.l1_loss(enhanced, normal)

            # 平滑损失（确保R和L的平滑性）
            smooth_R_loss = self.smooth_loss(R_opt)
            smooth_L_loss = self.smooth_loss(L_opt)
            smooth_loss = 0.1 * (smooth_R_loss + smooth_L_loss)

            # 去噪损失（去噪后反射与原始反射的一致性）
            denoise_loss = F.l1_loss(R_denoised, R_opt.detach())

            # 总损失
            total_batch_loss = recon_loss + enhance_loss + smooth_loss + 0.5 * denoise_loss

            # 反向传播
            total_batch_loss.backward()
            self.optim.step()
            total_loss += total_batch_loss.item()

            # 打印进度
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] 损失: {total_batch_loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def evaluate(self):
        """评估模型在测试集上的性能"""
        self.decom.eval()
        self.enhance.eval()
        total_ssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        count = 0

        with torch.no_grad():
            for low, normal in self.test_loader:
                low = low.to(device)
                normal = normal.to(device)

                # 分解与优化
                R_init, L_init = self.decom(low)
                R_opt, L_opt = self.iterative_optimize(low, R_init, L_init)

                # 增强
                enhanced, _, _ = self.enhance(L_opt, R_opt)

                # 计算指标
                total_ssim += self.ssim(enhanced, normal)
                total_psnr += self.psnr(enhanced, normal)

                # LPIPS需要[-1,1]范围输入
                enhanced_scaled = (enhanced * 2) - 1
                normal_scaled = (normal * 2) - 1
                total_lpips += self.lpips_model(enhanced_scaled, normal_scaled).mean()

                count += 1

        # 平均指标
        avg_ssim = (total_ssim / count).item()
        avg_psnr = (total_psnr / count).item()
        avg_lpips = (total_lpips / count).item()
        return avg_ssim, avg_psnr, avg_lpips

    def visualize(self, img_path):
        """可视化增强效果"""
        self.decom.eval()
        self.enhance.eval()
        img = Image.open(img_path).convert('RGB')
        tensor = test_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # 处理流程
            R_init, L_init = self.decom(tensor)
            R_opt, L_opt = self.iterative_optimize(tensor, R_init, L_init)
            enhanced, R_denoised, L_adjusted = self.enhance(L_opt, R_opt)

            # 转换为可视化格式
            def to_numpy(x):
                return x.squeeze().cpu().numpy().transpose(1, 2, 0).clip(0, 1)

        # 可视化
        plt.figure(figsize=(20, 4))
        titles = ['原始低光图像', '初始反射分量', '优化后反射分量', '去噪后反射分量',
                  '初始照明分量', '优化后照明分量', '调整后照明分量', '增强结果']
        images = [
            to_numpy(tensor),
            to_numpy(R_init),
            to_numpy(R_opt),
            to_numpy(R_denoised),
            to_numpy(L_init.repeat(1, 3, 1, 1)),
            to_numpy(L_opt.repeat(1, 3, 1, 1)),
            to_numpy(L_adjusted.repeat(1, 3, 1, 1)),
            to_numpy(enhanced)
        ]

        for i in range(8):
            plt.subplot(1, 8, i + 1)
            if i in [4, 5, 6]:  # 照明分量显示为灰度图
                plt.imshow(images[i][:, :, 0], cmap='gray')
            else:
                plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def train(self):
        """完整训练流程"""
        # 选择一张测试图用于可视化
        test_img = os.path.join(TEST_LOW_PATH, os.listdir(TEST_LOW_PATH)[0])

        for epoch in range(self.start_epoch, EPOCHS + 1):
            # 训练一个epoch
            avg_loss = self.train_epoch(epoch)

            # 评估
            ssim_val, psnr_val, lpips_val = self.evaluate()

            # 保存检查点到新目录
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
            torch.save({
                'decom': self.decom.state_dict(),
                'enhance': self.enhance.state_dict(),
                'epoch': epoch,
                'optim': self.optim.state_dict()
            }, checkpoint_path)

            # 记录日志到新目录
            with open(self.log_file, "a") as f:
                f.write(f"{epoch},{avg_loss:.4f},{ssim_val:.4f},{psnr_val:.4f},{lpips_val:.4f}\n")

            # 打印结果
            print(f"\nEpoch {epoch} 指标:")
            print(f"平均损失: {avg_loss:.4f} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.4f} dB | LPIPS: {lpips_val:.4f}")

            # 每10个epoch可视化一次
            if epoch % 50 == 0:
                self.visualize(test_img)


if __name__ == "__main__":
    # 从头训练，不加载任何检查点
    CHECKPOINT_TO_RESUME = None

    # 初始化训练器并开始训练
    trainer = RetinexTrainer(resume_checkpoint=CHECKPOINT_TO_RESUME)
    trainer.train()
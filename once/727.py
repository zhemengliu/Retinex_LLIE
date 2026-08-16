# 导入操作系统接口库，用于文件路径操作
import os

# 导入PyTorch深度学习框架
import torch

# 导入PyTorch的神经网络模块
import torch.nn as nn

# 导入PyTorch的函数式接口
import torch.nn.functional as F

# 导入PyTorch的数据集处理模块
from torch.utils.data import Dataset, DataLoader

# 导入PyTorch的图像变换模块
from torchvision import transforms

# 导入Python图像处理库
from PIL import Image

# 导入绘图库
import matplotlib.pyplot as plt

# 导入感知相似性指标库
import lpips

# 导入结构相似性指标
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# 导入峰值信噪比指标
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

# 设置环境变量，解决某些库的兼容性问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置计算设备（优先使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # 打印当前使用的设备

# 定义测试集正常光照图像路径
TEST_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high"

# 定义训练集低光图像路径
TRAIN_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"

# 定义训练集正常光照图像路径
TRAIN_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"

# 定义测试集低光图像路径
TEST_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"

# 定义模型检查点保存目录
CHECKPOINT_DIR = "./checkpoints"

# 超参数定义
BATCH_SIZE = 4  # 每个训练批次的样本数量
IMAGE_SIZE = 256  # 图像输入尺寸
EPOCHS = 10     # 训练总轮数
LR = 3e-4       # 初始学习率

# 训练数据预处理流程
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 调整图像尺寸
    transforms.RandomHorizontalFlip(p=0.5),      # 以50%概率水平翻转
    transforms.ToTensor(),                       # 转换为张量格式
])

# 测试数据预处理流程
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 调整图像尺寸
    transforms.ToTensor()                        # 转换为张量格式
])

# 自定义低光数据集类
class LowLightDataset(Dataset):
    def __init__(self, low_dir, normal_dir=None, transform=None):
        # 初始化函数
        self.low_files = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir)])  # 获取低光图像列表
        self.normal_files = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]) if normal_dir else None  # 正常光照图像列表
        self.transform = transform  # 数据转换操作

    def __len__(self):
        return len(self.low_files)  # 返回数据集总样本数

    def __getitem__(self, idx):
        # 获取单个样本
        low_img = Image.open(self.low_files[idx]).convert('RGB')  # 加载低光图像
        normal_img = Image.open(self.normal_files[idx]).convert('RGB') if self.normal_files else low_img.copy()  # 加载正常图像

        if self.transform:
            seed = 42  # 固定随机种子
            torch.manual_seed(seed)
            low_img = self.transform(low_img)  # 应用转换到低光图像
            torch.manual_seed(seed)
            normal_img = self.transform(normal_img)  # 应用相同的转换到正常图像
        return low_img, normal_img  # 返回配对数据

# 改进的分解网络
class DecomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            # 分解网络结构
            nn.Conv2d(3, 32, 3, padding=1),  # 输入3通道，输出32通道，3x3卷积
            nn.LeakyReLU(0.2, inplace=True),  # 带泄露的ReLU激活函数
            nn.Conv2d(32, 32, 3, padding=1),  # 保持通道数不变的卷积
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 4, 3, padding=1),   # 最终输出4通道（3通道反射图+1通道光照图）
            nn.ReLU()  # 使用ReLU保证输出非负
        )

    def forward(self, x):
        output = self.decom(x)  # 前向传播
        R = output[:, 0:3, :, :]  # 提取前3通道作为反射分量
        L = output[:, 3:4, :, :]  # 提取最后1通道作为光照分量
        return R, L

# 物理模型P模块
class P(nn.Module):
    def forward(self, I, Q, R, gamma):
        """
        P模块的物理方程：
        P* = (gamma*R + I*Q) / (gamma + Q^2)
        用于更新反射分量
        """
        return (I * Q + gamma * R) / (gamma + Q * Q + 1e-8)  # 添加极小值防止除零错误

# 物理模型Q模块
class Q(nn.Module):
    def forward(self, I, P, L, lamda):
        """
        Q模块的物理方程：
        Q* = (λL + Σ(I_i*P_i)) / (ΣP_i^2 + λ)
        用于更新光照分量
        """
        # 拆分RGB三通道
        IR, IG, IB = I[:, 0:1], I[:, 1:2], I[:, 2:3]
        PR, PG, PB = P[:, 0:1], P[:, 1:2], P[:, 2:3]

        numerator = IR * PR + IG * PG + IB * PB + lamda * L  # 分子计算
        denominator = PR**2 + PG**2 + PB**2 + lamda + 1e-8  # 分母计算
        return numerator / denominator

# 通道注意力机制模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # 降维全连接层
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),  # 升维全连接层
            nn.Sigmoid()  # 激活函数映射到0-1
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 空间维度压缩
        y = self.fc(y).view(b, c, 1, 1)  # 通道权重计算
        return x * y.expand_as(x)  # 应用通道注意力

# 恢复网络
class RestorationNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)  # 输入通道动态适配
        self.se = SELayer(64)  # 通道注意力模块
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, in_channels, 3, padding=1)  # 输出通道与输入一致

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 激活函数
        x = self.se(x)  # 应用注意力机制
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)  # 最终恢复输出

# 去噪网络（U-Net结构）
class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器部分
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),  # 批归一化
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)  # 下采样

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 解码器部分
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')  # 上采样
        self.dec1 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),  # 拼接后通道数增加
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.final = nn.Conv2d(64, 3, 1)  # 最终输出层

    def forward(self, x):
        enc1 = self.enc1(x)  # 第一层编码
        enc2 = self.pool(enc1)  # 下采样
        bottleneck = self.bottleneck(enc2)  # 瓶颈层处理
        dec1 = self.up(bottleneck)  # 上采样
        dec1 = torch.cat([dec1, enc1], dim=1)  # 跳跃连接
        dec1 = self.dec1(dec1)  # 解码处理
        return torch.sigmoid(self.final(dec1))  # 输出范围限制到0-1

# S曲线估计模块
class SCurveEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),             # 展平为向量
            nn.Linear(1, 32),         # 全连接层
            nn.ReLU(),
            nn.Linear(32, 3)          # 输出三个参数
        )

    def forward(self, L):
        params = self.fc(L)  # 参数预测
        a = torch.sigmoid(params[:, 0]) * 2 + 0.5  # 参数a范围[0.5, 2.5]
        b = torch.sigmoid(params[:, 1]) * 5 + 1    # 参数b范围[1, 6]
        c = params[:, 2] * 0.1                     # 参数c范围小幅度调整
        B, _, H, W = L.shape
        # S曲线公式应用
        L_adjusted = a.view(B, 1, 1, 1) * torch.sigmoid(
            b.view(B, 1, 1, 1) * (L + c.view(B, 1, 1, 1)))
        return torch.clamp(L_adjusted, 0, 1)  # 限制输出范围

# 增强网络
class EnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise = DenoiseNet()    # 去噪模块
        self.s_curve = SCurveEstimator()  # 光照调整模块

    def forward(self, L, R):
        R_denoised = self.denoise(R)  # 反射图去噪
        L_adjusted = self.s_curve(L)  # 光照调整
        enhanced = R_denoised * L_adjusted.repeat(1, 3, 1, 1)  # 合成增强图像
        return enhanced, R_denoised, L_adjusted

# 训练器类
class RetinexTrainer:
    def __init__(self):
        # 初始化分解网络（低光和正常光照版本）
        self.model_Decom_low = DecomNet().to(device)
        self.model_Decom_high = DecomNet().to(device)

        # 初始化恢复网络（反射和光照分别处理）
        self.model_R = RestorationNet(in_channels=3).to(device)  # 反射恢复
        self.model_L = RestorationNet(in_channels=1).to(device)  # 光照恢复

        # 初始化增强网络
        self.enhance = DenoiseNet().to(device)

        # 设置优化器（包含所有网络参数）
        self.optim = torch.optim.Adam(
            list(self.model_Decom_low.parameters()) +
            list(self.model_Decom_high.parameters()) +
            list(self.model_R.parameters()) +
            list(self.model_L.parameters()) +
            list(self.enhance.parameters()),
            lr=LR
        )

        # 物理模型模块
        self.P = P()
        self.Q = Q()

        # 数据加载器
        self.train_loader = DataLoader(
            LowLightDataset(TRAIN_LOW_PATH, TRAIN_NORMAL_PATH, train_transform),
            BATCH_SIZE, shuffle=True
        )
        self.test_loader = DataLoader(
            LowLightDataset(TEST_LOW_PATH, TEST_NORMAL_PATH, test_transform),
            BATCH_SIZE, shuffle=False
        )

        # 评估指标初始化
        self.ssim = SSIM(data_range=1.0).to(device)
        self.psnr = PSNR(data_range=1.0).to(device)
        self.lpips = lpips.LPIPS(net='alex').to(device)

        # 日志文件设置
        self.log_file = os.path.join(CHECKPOINT_DIR, "training_log.csv")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # 创建检查点目录

    def unfolding(self, input_low, rounds=3):
        """多阶段优化过程"""
        with torch.no_grad():  # 不计算梯度
            # 初始分解
            R, L = self.model_Decom_low(input_low)
            for t in range(rounds):
                # 动态调整参数
                gamma = 0.1 + 0.1 * t  # 逐渐增大的gamma
                lamda = 0.1 + 0.1 * t  # 逐渐增大的lambda

                # 使用P和Q模块更新
                P_val = self.P(input_low, L, R, gamma)
                Q_val = self.Q(input_low, R, L, lamda)

                # 恢复网络处理
                R = self.model_R(P_val)
                L = self.model_L(Q_val)
        return R, L

    def smooth_loss(self, I, R):
        """平滑正则化损失"""
        # 计算水平和垂直梯度
        grad_I_x = torch.abs(I[:, :, :, :-1] - I[:, :, :, 1:])
        grad_I_y = torch.abs(I[:, :, :-1, :] - I[:, :, 1:, :])
        grad_R_x = torch.abs(R[:, :, :, :-1] - R[:, :, :, 1:])
        grad_R_y = torch.abs(R[:, :, :-1, :] - R[:, :, 1:, :])
        return torch.mean(grad_I_x) + torch.mean(grad_I_y) + \
            torch.mean(grad_R_x) + torch.mean(grad_R_y)

    def train_epoch(self, epoch):
        """单个训练epoch"""
        # 设置训练模式
        self.model_Decom_low.train()
        self.model_Decom_high.train()
        self.model_R.train()
        self.model_L.train()
        self.enhance.train()

        total_loss = 0.0
        for batch_idx, (low, normal) in enumerate(self.train_loader):
            # 数据转移到设备
            low = low.to(device)
            normal = normal.to(device)

            self.optim.zero_grad()  # 梯度清零

            # 分解处理
            R_low, L_low = self.unfolding(low)
            R_normal, L_normal = self.model_Decom_high(normal)  # 正常光照分解

            # 维度匹配处理
            L_low = L_low.repeat(1, 3, 1, 1)  # 将光照分量扩展为3通道
            L_normal = L_normal.repeat(1, 3, 1, 1)

            # 增强处理
            enhanced = self.enhance(R_low * L_low)  # 合成低光图像并增强

            # 损失计算
            recon_loss = F.l1_loss(R_low * L_low, low)  # 重建损失
            mutual_loss = F.l1_loss(R_normal * L_low, low)  # 互信息损失
            smooth_loss = 0.1 * (self.smooth_loss(L_low, R_low) + self.smooth_loss(L_normal, R_normal))  # 平滑损失
            enhance_loss = F.l1_loss(enhanced, normal)  # 增强损失

            total_loss = recon_loss + 0.1 * mutual_loss + smooth_loss + enhance_loss
            total_loss.backward()  # 反向传播
            self.optim.step()      # 参数更新

            # 进度打印
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {total_loss.item():.4f}")

        return total_loss / len(self.train_loader)  # 返回平均损失

    def evaluate(self):
        """模型评估"""
        self.model_Decom_low.eval()
        self.model_R.eval()
        self.model_L.eval()
        self.enhance.eval()

        total_ssim, total_psnr, total_lpips = 0.0, 0.0, 0.0
        with torch.no_grad():
            for low, normal in self.test_loader:
                low = low.to(device)
                normal = normal.to(device)

                # 分解处理
                R, L = self.unfolding(low)

                # 通道处理
                if L.shape[1] == 1:
                    L = L.repeat(1, 3, 1, 1)

                # 增强处理
                enhanced = self.enhance(R * L)

                # 数值范围限制
                enhanced = torch.clamp(enhanced, 0, 1)

                # 转换到[-1,1]范围
                enhanced_scaled = (enhanced * 2) - 1
                normal_scaled = (normal * 2) - 1

                # 通道数校验
                if enhanced_scaled.shape[1] != 3:
                    enhanced_scaled = enhanced_scaled[:, :3, :, :]

                # 指标计算
                total_ssim += self.ssim(enhanced, normal)
                total_psnr += self.psnr(enhanced, normal)
                total_lpips += self.lpips(enhanced_scaled, normal_scaled).mean()

        # 计算平均指标
        avg_ssim = total_ssim / len(self.test_loader)
        avg_psnr = total_psnr / len(self.test_loader)
        avg_lpips = total_lpips / len(self.test_loader)

        print(f"SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.4f} | LPIPS: {avg_lpips:.4f}")

    def visualize(self, img_path):
        """可视化结果"""
        self.decom.eval()
        self.enhance.eval()
        img = Image.open(img_path).convert('RGB')
        tensor = test_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            R, I = self.decom(tensor)
            enhanced, R_denoised, I_adjusted = self.enhance(I, R)

            def to_numpy(x):
                """张量转numpy格式"""
                return x.squeeze().cpu().numpy().transpose(1, 2, 0).clip(0, 1)

        # 可视化结果布局
        plt.figure(figsize=(25, 5))
        titles = ['Original', 'Reflectance', 'Denoised R', 'Illumination', 'Adjusted I', 'Enhanced']
        images = [
            to_numpy(tensor),
            to_numpy(R),
            to_numpy(R_denoised),
            to_numpy(I.repeat(1, 3, 1, 1)),  # 单通道转为三通道显示
            to_numpy(I_adjusted.repeat(1, 3, 1, 1)),
            to_numpy(enhanced)
        ]

        # 绘制子图
        for i in range(6):
            plt.subplot(1, 6, i + 1)
            if i in [3, 4]:  # 光照图用灰度显示
                plt.imshow(images[i][:, :, 0], cmap='gray')
            else:
                plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def train(self):
        """完整训练流程"""
        for epoch in range(1, EPOCHS + 1):
            loss = self.train_epoch(epoch)  # 训练一个epoch
            self.evaluate()  # 在测试集评估

            # 保存检查点
            if epoch % 10 == 0:
                torch.save({
                    'decom_low': self.model_Decom_low.state_dict(),
                    'decom_high': self.model_Decom_high.state_dict(),
                    'model_R': self.model_R.state_dict(),
                    'model_L': self.model_L.state_dict(),
                    'enhance': self.enhance.state_dict(),
                    'epoch': epoch
                }, os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth"))

            # 可视化样例
            test_img = os.path.join(TEST_LOW_PATH, os.listdir(TEST_LOW_PATH)[0])
            if epoch % 1 == 0:  # 每个epoch都可视化
                self.visualize(test_img)

# 主程序入口
if __name__ == "__main__":
    trainer = RetinexTrainer()  # 初始化训练器
    trainer.train()             # 开始训练
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tvF  # 添加tvF的导入
from PIL import Image, ImageFont, ImageDraw  # 添加ImageFont和ImageDraw的导入
import matplotlib.pyplot as plt
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
import numpy as np
import random
from sys import platform
from string import ascii_letters
from tqdm import tqdm

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径配置（请根据实际路径修改）
TEST_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high"
TRAIN_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"
TRAIN_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"
TEST_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"
# 去噪训练数据集路径（请修改为实际路径）
DENOISE_TRAIN_DIR = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"#"./dataset/train/"
DENOISE_VAL_DIR = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"#"./dataset/val/"
# 设置新的检查点保存目录
CHECKPOINT_DIR = "D:\\文件\\大创\\train728"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("./result", exist_ok=True)

# 超参数配置
BATCH_SIZE = 4
IMAGE_SIZE = 256
EPOCHS = 1
LR = 3e-4
ITERATION_NUM = 3  # P/Q迭代次数
GAMMA = 0.1  # P模块正则化参数
LAMDA = 0.1  # Q模块正则化参数

# 去噪模块参数
DENOISE_CROP_SIZE = 128
DENOISE_BATCH_SIZE = 8
DENOISE_LR = 1e-4
DENOISE_EPOCHS = 1
NOISE_MODEL = ('gaussian', 50)  # 噪声模型：高斯噪声，标准差50

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


# 低光图像数据集类
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


# 噪声数据集类（用于去噪模块训练）
class NoisyDataset(Dataset):
    def __init__(self, root_dir, crop_size=128, train_noise_model=('gaussian', 50), clean_targ=False):
        """
            root_dir: 图像目录路径
            crop_size: 裁剪图像的尺寸
            clean_targ: 是否使用干净图像作为目标
        """
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.clean_targ = clean_targ
        self.noise = train_noise_model[0]
        self.noise_param = train_noise_model[1]
        self.imgs = [f for f in os.listdir(root_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    def _random_crop_to_size(self, imgs):
        w, h = imgs[0].size
        if min(w, h) < self.crop_size:
            imgs = [tvF.resize(img, (self.crop_size, self.crop_size)) for img in imgs]
            w, h = self.crop_size, self.crop_size

        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        cropped_imgs = [tvF.crop(img, i, j, self.crop_size, self.crop_size) for img in imgs]
        return cropped_imgs

    def _add_gaussian_noise(self, image):
        """添加高斯噪声"""
        w, h = image.size
        c = len(image.getbands())

        std = np.random.uniform(0, self.noise_param)
        _n = np.random.normal(0, std, (h, w, c))
        noisy_image = np.array(image) + _n

        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return {'image': Image.fromarray(noisy_image), 'mask': None, 'use_mask': False}

    def _add_poisson_noise(self, image):
        """添加泊松噪声"""
        img_array = np.array(image, dtype=np.float32)
        noise_mask = np.random.poisson(img_array)
        return {'image': Image.fromarray(noise_mask.astype(np.uint8)), 'mask': None, 'use_mask': False}

    def _add_m_bernoulli_noise(self, image):
        """添加乘法伯努利噪声"""
        img_array = np.array(image)
        sz = img_array.shape[:2]
        prob_ = random.uniform(0, self.noise_param)
        mask = np.random.choice([0, 1], size=sz, p=[prob_, 1 - prob_])
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        noisy_img = np.multiply(img_array, mask).astype(np.uint8)
        return {'image': Image.fromarray(noisy_img), 'mask': Image.fromarray((mask * 255).astype(np.uint8)),
                'use_mask': True}

    def _add_text_overlay(self, image):
        """添加文本叠加噪声"""
        assert self.noise_param < 1, '文本参数应该是占据概率'

        w, h = image.size
        c = len(image.getbands())

        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'

        text_img = image.copy()
        text_draw = ImageDraw.Draw(text_img)
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        max_occupancy = np.random.uniform(0, self.noise_param)

        def get_occupancy(x):
            y = np.array(x, np.uint8)
            return np.sum(y) / y.size

        while 1:
            try:
                font = ImageFont.truetype(serif, np.random.randint(16, 21))
                length = np.random.randint(10, 25)
                chars = ''.join(random.choice(ascii_letters) for i in range(length))
                color = tuple(np.random.randint(0, 255, c))
                pos = (np.random.randint(0, w), np.random.randint(0, h))
                text_draw.text(pos, chars, color, font=font)
                mask_draw.text(pos, chars, 1, font=font)

                if get_occupancy(mask_img) > max_occupancy:
                    break
            except:
                # 处理字体加载失败的情况
                break

        return {'image': text_img, 'mask': None, 'use_mask': False}

    def corrupt_image(self, image):
        """根据选择的噪声模型添加噪声"""
        if self.noise == 'gaussian':
            return self._add_gaussian_noise(image)
        elif self.noise == 'poisson':
            return self._add_poisson_noise(image)
        elif self.noise == 'multiplicative_bernoulli':
            return self._add_m_bernoulli_noise(image)
        elif self.noise == 'text':
            return self._add_text_overlay(image)
        else:
            raise ValueError('不支持的图像噪声类型')

    def __getitem__(self, index):
        """读取图像，添加噪声并返回"""
        img_path = os.path.join(self.root_dir, self.imgs[index])
        image = Image.open(img_path).convert('RGB')

        # 对图片进行随机切割
        if self.crop_size > 0:
            image = self._random_crop_to_size([image])[0]

        # 噪声图片1
        source_img_dict = self.corrupt_image(image)
        source_img = transforms.ToTensor()(source_img_dict['image'])

        # 噪声图片2或干净图片
        if self.clean_targ:
            target = transforms.ToTensor()(image)
        else:
            _target_dict = self.corrupt_image(image)
            target = transforms.ToTensor()(_target_dict['image'])

        # 原始干净图像（用于可视化）
        clean_img = np.array(image).astype(np.uint8)

        if source_img_dict['use_mask']:
            mask = transforms.ToTensor()(source_img_dict['mask'])
            return [source_img, mask, target, clean_img]
        else:
            return [source_img, target, clean_img]

    def __len__(self):
        return len(self.imgs)


# 工具函数：定义卷积层
def get_conv2d_layer(in_c, out_c, k, s, p):
    """创建卷积层（含BatchNorm）"""
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=s, padding=p),
        nn.BatchNorm2d(out_c)
    )


# Retinex分解网络
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


# 基于SRResnet的去噪网络（用于反射分量去噪）
class DenoiseNet(nn.Module):
    """用于反射分量去噪的残差网络，学习噪声到噪声的映射或噪声到干净图像的映射"""

    def __init__(self, in_channels=3, features=64, res_layers=16):
        super().__init__()
        # 初始卷积（将输入映射到特征空间）
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)  # 激活函数

        # 残差层堆叠（核心去噪特征学习）
        self.res_blocks = nn.Sequential(
            *[ResBlock(features, features, kernel_size=3) for _ in range(res_layers)]
        )

        # 残差连接后的卷积（融合特征）
        self.conv2 = ConvBlock(features, features, kernel_size=3, use_act=False)

        # 输出层（将特征映射回3通道）
        self.conv3 = nn.Conv2d(features, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        输入：3通道带噪声的反射分量（[0,1]范围张量）
        输出：3通道去噪后反射分量（[0,1]范围张量）
        """
        # 初始特征提取
        x1 = self.act(self.conv1(x))

        # 残差块特征学习 + 跳跃连接（残差融合）
        res_out = self.res_blocks(x1)
        x2 = self.conv2(res_out)
        x3 = x1 + x2  # 跳跃连接增强特征保留

        # 输出3通道结果并约束在[0,1]范围
        out = self.conv3(x3)
        return torch.clamp(out, 0, 1)  # 确保输出在有效范围


# 去噪网络训练器
class DenoiseTrainer:
    def __init__(self, denoise_net, params):
        self.cuda = params['cuda']
        if self.cuda:
            self.model = denoise_net.cuda()
        else:
            self.model = denoise_net

        self.train_dir = params['train_dir']
        self.val_dir = params['val_dir']
        self.noise_model = params['noise_model']
        self.crop_size = params['crop_size']
        self.clean_targs = params['clean_targs']
        self.lr = params['lr']
        self.epochs = params['epochs']
        self.bs = params['batch_size']

        self.train_dl, self.val_dl = self._get_dataloaders()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.epochs // 4, factor=0.5, verbose=True)
        self.loss_fn = nn.L1Loss()  # 使用L1损失更适合去噪任务

        # 创建结果保存目录
        noise_name = f"{self.noise_model[0]}_{self.noise_model[1]}"
        self.result_dir = os.path.join("./result", noise_name)
        os.makedirs(self.result_dir, exist_ok=True)

    def _get_dataloaders(self):
        # 创建噪声训练数据集
        train_ds = NoisyDataset(
            self.train_dir,
            crop_size=self.crop_size,
            train_noise_model=self.noise_model,
            clean_targ=self.clean_targs
        )
        train_dl = DataLoader(train_ds, batch_size=self.bs, shuffle=True, num_workers=2)

        # 创建噪声验证数据集（使用干净图像作为目标）
        val_ds = NoisyDataset(
            self.val_dir,
            crop_size=self.crop_size,
            train_noise_model=self.noise_model,
            clean_targ=True
        )
        val_dl = DataLoader(val_ds, batch_size=self.bs, num_workers=2)

        return train_dl, val_dl

    def evaluate(self):
        """在验证集上评估去噪模型"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data in self.val_dl:
                if self.cuda:
                    source = data[0].cuda()
                    target = data[-2].cuda()
                else:
                    source = data[0]
                    target = data[-2]

                # 模型预测
                output = self.model(source)

                # 计算损失
                if len(data) == 4:  # 有掩码的情况
                    if self.cuda:
                        mask = data[1].cuda()
                    else:
                        mask = data[1]
                    loss = self.loss_fn(mask * output, mask * target)
                else:
                    loss = self.loss_fn(output, target)

                total_loss += loss.item()

        return total_loss / len(self.val_dl)

    def train(self):
        """训练去噪模型"""
        print(f"开始训练去噪模型，共{self.epochs}个epoch...")
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            # 训练一个epoch
            for data in tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                if self.cuda:
                    source = data[0].cuda()
                    target = data[-2].cuda()
                else:
                    source = data[0]
                    target = data[-2]

                # 前向传播
                self.optimizer.zero_grad()
                output = self.model(source)

                # 计算损失
                if len(data) == 4:  # 有掩码的情况
                    if self.cuda:
                        mask = data[1].cuda()
                    else:
                        mask = data[1]
                    loss = self.loss_fn(mask * output, mask * target)
                else:
                    loss = self.loss_fn(output, target)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # 计算平均训练损失
            avg_train_loss = train_loss / len(self.train_dl)

            # 在验证集上评估
            avg_val_loss = self.evaluate()

            # 学习率调整
            self.scheduler.step(avg_val_loss)

            # 打印 epoch 结果
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(CHECKPOINT_DIR, f"best_denoiser.pth")
                )
                print(f"保存最佳去噪模型 (验证损失: {best_val_loss:.4f})")

            # 每10个epoch可视化一次结果
            if (epoch + 1) % 10 == 0:
                self.visualize_results(data)

        # 加载最佳模型权重
        self.model.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, f"best_denoiser.pth"),
                       map_location=device, weights_only=True)
        )
        return self.model

    def visualize_results(self, data):
        """可视化去噪效果"""
        self.model.eval()
        with torch.no_grad():
            if self.cuda:
                source = data[0].cuda()[:4]  # 取前4个样本
            else:
                source = data[0][:4]

            output = self.model(source)

            # 转换为可视化格式
            source_np = np.transpose(source.cpu().numpy(), (0, 2, 3, 1))
            output_np = np.transpose(output.cpu().numpy(), (0, 2, 3, 1))
            clean_np = data[-1].cpu().numpy()[:4]  # 干净图像

            # 创建可视化图像
            plt.figure(figsize=(15, 10))
            for i in range(4):
                # 原始噪声图像
                plt.subplot(3, 4, i + 1)
                plt.imshow(source_np[i].clip(0, 1))
                plt.title("噪声图像")
                plt.axis('off')

                # 去噪结果
                plt.subplot(3, 4, i + 5)
                plt.imshow(output_np[i].clip(0, 1))
                plt.title("去噪结果")
                plt.axis('off')

                # 干净图像
                plt.subplot(3, 4, i + 9)
                plt.imshow(clean_np[i])
                plt.title("干净图像")
                plt.axis('off')

            # 保存可视化结果
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_dir, f"denoise_epoch_{len(os.listdir(self.result_dir)) + 1}.png"))
            plt.close()


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
    def __init__(self, denoise_net):
        super().__init__()
        self.denoise = denoise_net  # 反射分量去噪
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


# 主训练器类
class RetinexTrainer:
    def __init__(self, resume_checkpoint=None):
        # 初始化去噪网络并进行预训练
        self.denoise_net = DenoiseNet()
        self._pretrain_denoiser()

        # 初始化其他网络
        self.decom = Decom().to(device)  # 分解网络
        self.enhance = EnhanceNet(self.denoise_net).to(device)  # 增强网络（包含预训练的去噪网络）
        self.p_module = P()  # P迭代模块
        self.q_module = Q()  # Q迭代模块

        # 优化器（联合训练分解网络和增强网络）
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
        self.start_epoch = 1

        # 加载检查点（如果需要）
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=True)

            # 加载分解网络权重
            if 'decom' in checkpoint:
                try:
                    self.decom.load_state_dict(checkpoint['decom'])
                except RuntimeError as e:
                    print(f"分解网络权重加载警告: {e}")

            # 加载增强网络权重（包含去噪网络）
            if 'enhance' in checkpoint:
                try:
                    self.enhance.load_state_dict(checkpoint['enhance'])
                except RuntimeError as e:
                    print(f"增强网络权重加载警告: {e}")

            # 恢复训练进度
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"加载检查点（epoch {checkpoint['epoch']}），从epoch {self.start_epoch} 继续训练")

            # 加载优化器状态
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

    def _pretrain_denoiser(self):
        """预训练去噪网络"""
        denoise_params = {
            'cuda': device.type == 'cuda',
            'train_dir': DENOISE_TRAIN_DIR,
            'val_dir': DENOISE_VAL_DIR,
            'noise_model': NOISE_MODEL,
            'crop_size': DENOISE_CROP_SIZE,
            'clean_targs': False,  # 先学习噪声到噪声的映射
            'lr': DENOISE_LR,
            'epochs': DENOISE_EPOCHS,
            'batch_size': DENOISE_BATCH_SIZE
        }

        # 初始化去噪训练器并开始训练
        denoise_trainer = DenoiseTrainer(self.denoise_net, denoise_params)
        self.denoise_net = denoise_trainer.train()

        # 如果样本数量多，再微调去噪网络学习噪声到干净图像的映射
        denoise_params['clean_targs'] = True
        denoise_trainer = DenoiseTrainer(self.denoise_net, denoise_params)
        self.denoise_net = denoise_trainer.train()

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

            # 保存检查点
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
            torch.save({
                'decom': self.decom.state_dict(),
                'enhance': self.enhance.state_dict(),
                'epoch': epoch,
                'optim': self.optim.state_dict()
            }, checkpoint_path)

            # 记录日志
            with open(self.log_file, "a") as f:
                f.write(f"{epoch},{avg_loss:.4f},{ssim_val:.4f},{psnr_val:.4f},{lpips_val:.4f}\n")

            # 打印结果
            print(f"\nEpoch {epoch} 指标:")
            print(f"平均损失: {avg_loss:.4f} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.4f} dB | LPIPS: {lpips_val:.4f}")

            # 每50个epoch可视化一次
            if epoch % 50 == 0:
                self.visualize(test_img)


if __name__ == "__main__":
    # 从头训练，不加载任何检查点
    CHECKPOINT_TO_RESUME = None

    # 初始化训练器并开始训练
    trainer = RetinexTrainer(resume_checkpoint=CHECKPOINT_TO_RESUME)
    trainer.train()

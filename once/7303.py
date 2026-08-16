import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tvF
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
import numpy as np
import random
from sys import platform
from string import ascii_letters
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')  # 用文件系统代替共享内存
# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 新增的损失函数
class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
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
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)

        return torch.mean(E)


class L_exp(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return torch.mean(k)


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


# 数据路径配置（请根据实际路径修改）
TEST_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high"
TRAIN_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"
TRAIN_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"
TEST_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"
# 去噪训练数据集路径
DENOISE_TRAIN_DIR = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"
DENOISE_VAL_DIR = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"
# 检查点保存目录
CHECKPOINT_DIR = "D:\\文件\\大创\\train728"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("./result", exist_ok=True)

# 超参数配置
BATCH_SIZE = 4
IMAGE_SIZE = 256
EPOCHS = 100
LR = 3e-4
ITERATION_NUM = 3  # P/Q迭代次数
GAMMA = 0.05  # P模块正则化参数
LAMDA = 0.2  # Q模块正则化参数

# 去噪模块参数
DENOISE_CROP_SIZE = 128
DENOISE_BATCH_SIZE = 8
DENOISE_LR = 1e-4
DENOISE_EPOCHS = 100
NOISE_MODEL = ('gaussian', 50)  # 噪声模型：高斯噪声，标准差50

# 损失函数权重
LOSS_WEIGHTS = {
    'recon': 1.0,
    'enhance': 1.5,
    'smooth': 0.1,
    'denoise': 0.5,
    'spa': 0.01,  # 空间损失权重
    'exp': 0.5,  # 曝光损失权重
    'color': 0.1,  # 颜色损失权重
    'tv': 0.001  # TV损失权重
}

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
        w, h = image.size
        c = len(image.getbands())

        std = np.random.uniform(0, self.noise_param)
        _n = np.random.normal(0, std, (h, w, c))
        noisy_image = np.array(image) + _n

        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return {'image': Image.fromarray(noisy_image), 'mask': None, 'use_mask': False}

    def _add_poisson_noise(self, image):
        img_array = np.array(image, dtype=np.float32)
        noise_mask = np.random.poisson(img_array)
        return {'image': Image.fromarray(noise_mask.astype(np.uint8)), 'mask': None, 'use_mask': False}

    def _add_m_bernoulli_noise(self, image):
        img_array = np.array(image)
        sz = img_array.shape[:2]
        prob_ = random.uniform(0, self.noise_param)
        mask = np.random.choice([0, 1], size=sz, p=[prob_, 1 - prob_])
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        noisy_img = np.multiply(img_array, mask).astype(np.uint8)
        return {'image': Image.fromarray(noisy_img), 'mask': Image.fromarray((mask * 255).astype(np.uint8)),
                'use_mask': True}

    def _add_text_overlay(self, image):
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
                break

        return {'image': text_img, 'mask': None, 'use_mask': False}

    def corrupt_image(self, image):
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
        img_path = os.path.join(self.root_dir, self.imgs[index])
        image = Image.open(img_path).convert('RGB')

        if self.crop_size > 0:
            image = self._random_crop_to_size([image])[0]

        source_img_dict = self.corrupt_image(image)
        source_img = transforms.ToTensor()(source_img_dict['image'])

        if self.clean_targ:
            target = transforms.ToTensor()(image)
        else:
            _target_dict = self.corrupt_image(image)
            target = transforms.ToTensor()(_target_dict['image'])

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
            nn.ReLU()
        )

    def forward(self, input):
        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        R = torch.sigmoid(R)
        L = torch.sigmoid(L)
        return R, L


# P迭代模块：优化反射分量
class P(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, I, Q, R, gamma):
        return ((I * Q + gamma * R) / (gamma + Q * Q)).clamp(0, 1)


# Q迭代模块：优化照明分量
class Q(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, I, P, L, lamda):
        IR, IG, IB = I[:, 0:1, :, :], I[:, 1:2, :, :], I[:, 2:3, :, :]
        PR, PG, PB = P[:, 0:1, :, :], P[:, 1:2, :, :], P[:, 2:3, :, :]

        numerator = (IR * PR + IG * PG + IB * PB) + lamda * L
        denominator = (PR * PR + PG * PG + PB * PB) + lamda
        return (numerator / denominator).clamp(0, 1)


# 基础卷积块（含BN和激活）
class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, pad=1, use_act=True):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=pad)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.use_act:
            x = self.act(x)
        return x


# 残差块（核心特征学习单元）
class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.block1 = ConvBlock(input_channels, output_channels, kernel_size)
        self.block2 = ConvBlock(input_channels, output_channels, kernel_size, use_act=False)

    def forward(self, x):
        return x + self.block2(self.block1(x))


# 基于SRResnet的去噪网络
class DenoiseNet(nn.Module):
    def __init__(self, in_channels=3, features=64, res_layers=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.res_blocks = nn.Sequential(
            *[ResBlock(features, features, kernel_size=3) for _ in range(res_layers)]
        )

        self.conv2 = ConvBlock(features, features, kernel_size=3, use_act=False)
        self.conv3 = nn.Conv2d(features, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        res_out = self.res_blocks(x1)
        x2 = self.conv2(res_out)
        x3 = x1 + x2
        out = self.conv3(x3)
        return torch.clamp(out, 0, 1)


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
        self.loss_fn = nn.L1Loss()

        noise_name = f"{self.noise_model[0]}_{self.noise_model[1]}"
        self.result_dir = os.path.join("./result", noise_name)
        os.makedirs(self.result_dir, exist_ok=True)

    def _get_dataloaders(self):
        train_ds = NoisyDataset(
            self.train_dir,
            crop_size=self.crop_size,
            train_noise_model=self.noise_model,
            clean_targ=self.clean_targs
        )
        train_dl = DataLoader(train_ds, batch_size=self.bs, shuffle=True, num_workers=2)

        val_ds = NoisyDataset(
            self.val_dir,
            crop_size=self.crop_size,
            train_noise_model=self.noise_model,
            clean_targ=True
        )
        val_dl = DataLoader(val_ds, batch_size=self.bs, num_workers=2)

        return train_dl, val_dl

    def evaluate(self):
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

                output = self.model(source)

                if len(data) == 4:
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
        print(f"开始训练去噪模型，共{self.epochs}个epoch...")
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for data in tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                if self.cuda:
                    source = data[0].cuda()
                    target = data[-2].cuda()
                else:
                    source = data[0]
                    target = data[-2]

                self.optimizer.zero_grad()
                output = self.model(source)

                if len(data) == 4:
                    if self.cuda:
                        mask = data[1].cuda()
                    else:
                        mask = data[1]
                    loss = self.loss_fn(mask * output, mask * target)
                else:
                    loss = self.loss_fn(output, target)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_dl)
            avg_val_loss = self.evaluate()
            self.scheduler.step(avg_val_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(CHECKPOINT_DIR, f"best_denoiser.pth")
                )
                print(f"保存最佳去噪模型 (验证损失: {best_val_loss:.4f})")

            if (epoch + 1) % 10 == 0:
                self.visualize_results(data)

        self.model.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, f"best_denoiser.pth"),
                       map_location=device, weights_only=True)
        )
        return self.model

    def visualize_results(self, data):
        self.model.eval()
        with torch.no_grad():
            if self.cuda:
                source = data[0].cuda()[:4]
            else:
                source = data[0][:4]

            output = self.model(source)

            source_np = np.transpose(source.cpu().numpy(), (0, 2, 3, 1))
            output_np = np.transpose(output.cpu().numpy(), (0, 2, 3, 1))
            clean_np = data[-1].cpu().numpy()[:4]

            plt.figure(figsize=(15, 10))
            for i in range(4):
                plt.subplot(3, 4, i + 1)
                plt.imshow(source_np[i].clip(0, 1))
                plt.title("噪声图像")
                plt.axis('off')

                plt.subplot(3, 4, i + 5)
                plt.imshow(output_np[i].clip(0, 1))
                plt.title("去噪结果")
                plt.axis('off')

                plt.subplot(3, 4, i + 9)
                plt.imshow(clean_np[i])
                plt.title("干净图像")
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(self.result_dir, f"denoise_epoch_{len(os.listdir(self.result_dir)) + 1}.png"))
            plt.close()


# 基于enhance_net_nopool改进的S曲线增强模块
# 替换原SCurveEnhanceNet的forward方法
class SCurveEnhanceNet(nn.Module):
    def __init__(self, num_steps=4):
        super().__init__()
        self.num_steps = num_steps
        self.debug = False  # 可设为True观察参数变化
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * num_steps, 3, padding=1)  # 输出调整参数
        )
        # 初始化权重，偏向更大的调整幅度
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # 偏置稍大，鼓励更强调整

    def forward(self, L):
        if self.debug:
            print(f"输入L的范围: [{L.min().item():.3f}, {L.max().item():.3f}], 均值: {L.mean().item():.3f}")
        params = self.net(L)
        current_L = L

        # 增加迭代步数（从1→2），分阶段增强
        for i in range(2):  # 两步调整，先提亮再增强对比度
            a, b, c = torch.split(params[:, i * 3:(i + 1) * 3], 1, dim=1)

            # 第1步：主要提亮（更大的a和正向c）
            if i == 0:
                a = torch.sigmoid(a) * 2.0 + 0.5  # 放大因子范围[0.5, 2.5]（原[0.5,2.0]）
                b = torch.sigmoid(b) * 1.5 + 0.5  # 曲率[0.5, 2.0]（适中，避免过度陡峭）
                c = torch.tanh(c) * 0.6  # 偏移量[-0.6, 0.6]（更大正向偏移，增强提亮）
            # 第2步：主要增强对比度（更大的b）
            else:
                a = torch.sigmoid(a) * 0.5 + 0.8  # 放大因子[0.8, 1.3]（微调亮度）
                b = torch.sigmoid(b) * 3.0 + 1.0  # 曲率[1.0, 4.0]（更陡峭，增强对比度）
                c = torch.tanh(c) * 0.3  # 小偏移，聚焦对比度

            if self.debug:
                print(f"第{i+1}步 - a: [{a.min():.3f},{a.max():.3f}], b: [{b.min():.3f},{b.max():.3f}], c: [{c.min():.3f},{c.max():.3f}]")

            # S曲线公式：output = a * sigmoid(b*(input + c))
            current_L = a * torch.sigmoid(b * (current_L + c))
            current_L = torch.clamp(current_L, 0.05, 0.95)  # 避免过曝（上限从1.0→0.95）

        return current_L


# 增强网络（整合去噪和光照调整）
class EnhanceNet(nn.Module):
    def __init__(self, denoise_net):
        super().__init__()
        self.denoise = denoise_net  # 反射分量去噪
        self.s_curve = SCurveEnhanceNet(num_steps=8)  # 使用新的S曲线增强模块

    def forward(self, L, R):
        R_denoised = self.denoise(R)  # 反射去噪
        L_adjusted = self.s_curve(L)  # 照明增强（S曲线多步调整）
        enhanced = R_denoised * L_adjusted.repeat(1, 3, 1, 1)  # 重构增强图像
        return enhanced, R_denoised, L_adjusted


# 主训练器类
class RetinexTrainer:
    def __init__(self, resume_checkpoint=None, load_denoiser=True):
        # 初始化去噪网络，根据参数决定是否加载预训练模型
        self.denoise_net = DenoiseNet()
        self._init_denoiser(load_denoiser)

        # 初始化其他网络
        self.decom = Decom().to(device)
        self.enhance = EnhanceNet(self.denoise_net).to(device)
        self.p_module = P()
        self.q_module = Q()

        # 初始化新增的损失函数
        self.l_spa = L_spa().to(device)
        self.l_exp = L_exp(patch_size=16, mean_val=0.6).to(device)
        self.l_color = L_color().to(device)
        self.l_tv = L_TV(TVLoss_weight=1).to(device)

        # 优化器
        self.optim = torch.optim.Adam(
            list(self.decom.parameters()) + list(self.enhance.parameters()),
            lr=LR
        )

        # 数据加载
        self.train_loader = DataLoader(
            LowLightDataset(TRAIN_LOW_PATH, TRAIN_NORMAL_PATH, train_transform),
            BATCH_SIZE, shuffle=True, num_workers=0
        )
        self.test_loader = DataLoader(
            LowLightDataset(TEST_LOW_PATH, TEST_NORMAL_PATH, transform=test_transform),
            BATCH_SIZE, shuffle=False, num_workers=0
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

        # 加载检查点
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=True)

            if 'decom' in checkpoint:
                try:
                    self.decom.load_state_dict(checkpoint['decom'])
                except RuntimeError as e:
                    print(f"分解网络权重加载警告: {e}")

            if 'enhance' in checkpoint:
                try:
                    self.enhance.load_state_dict(checkpoint['enhance'])
                except RuntimeError as e:
                    print(f"增强网络权重加载警告: {e}")

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"加载检查点（epoch {checkpoint['epoch']}），从epoch {self.start_epoch} 继续训练")

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

    def _init_denoiser(self, load_denoiser):
        """初始化去噪网络，可以选择训练新的或加载已有的"""
        denoise_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_denoiser.pth")

        if load_denoiser and os.path.exists(denoise_checkpoint_path):
            # 加载预训练模型
            try:
                self.denoise_net.load_state_dict(
                    torch.load(denoise_checkpoint_path, map_location=device, weights_only=True)
                )
                print(f"成功加载预训练去噪模型: {denoise_checkpoint_path}")
                self.denoise_net = self.denoise_net.to(device)
                return  # 加载成功则直接返回
            except RuntimeError as e:
                print(f"去噪模型权重加载失败: {e}，将重新训练去噪模型")

        # 如果不需要加载或加载失败，则训练新的去噪模型
        print("开始训练新的去噪模型...")
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

        denoise_trainer = DenoiseTrainer(self.denoise_net, denoise_params)
        self.denoise_net = denoise_trainer.train()

        # 微调去噪网络学习噪声到干净图像的映射
        denoise_params['clean_targs'] = True
        denoise_trainer = DenoiseTrainer(self.denoise_net, denoise_params)
        self.denoise_net = denoise_trainer.train()

    def iterative_optimize(self, I, R_init, L_init, iterations=ITERATION_NUM):
        R = R_init.clone()
        L = L_init.clone()

        for _ in range(iterations):
            R = self.p_module(I, L, R, GAMMA)
            L = self.q_module(I, R, L, LAMDA)

        return R, L

    def smooth_loss(self, x):
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

            # 4. 计算原有损失
            recon_loss = F.l1_loss(R_opt * L_opt.repeat(1, 3, 1, 1), low) * LOSS_WEIGHTS['recon']
            enhance_loss = F.l1_loss(enhanced, normal) * LOSS_WEIGHTS['enhance']
            smooth_R_loss = self.smooth_loss(R_opt)
            smooth_L_loss = self.smooth_loss(L_opt)
            smooth_loss = (smooth_R_loss + smooth_L_loss) * LOSS_WEIGHTS['smooth']
            denoise_loss = F.l1_loss(R_denoised, R_opt.detach()) * LOSS_WEIGHTS['denoise']

            # 5. 计算新增的损失
            spa_loss = self.l_spa(normal, enhanced) * LOSS_WEIGHTS['spa']
            exp_loss = self.l_exp(enhanced) * LOSS_WEIGHTS['exp']
            color_loss = self.l_color(enhanced) * LOSS_WEIGHTS['color']
            tv_loss = self.l_tv(enhanced) * LOSS_WEIGHTS['tv']

            # 6. 总损失
            total_batch_loss = recon_loss + enhance_loss + smooth_loss + denoise_loss + \
                               spa_loss + exp_loss + color_loss + tv_loss

            # 反向传播
            total_batch_loss.backward()
            self.optim.step()
            total_loss += total_batch_loss.item()

            # 打印进度
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] 损失: {total_batch_loss.item():.4f}")

        # 清理内存
        del R_init, L_init, R_opt, L_opt, R_denoised, L_adjusted
        torch.cuda.empty_cache()
        return total_loss / len(self.train_loader)

    def evaluate(self):
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

        avg_ssim = (total_ssim / count).item()
        avg_psnr = (total_psnr / count).item()
        avg_lpips = (total_lpips / count).item()
        return avg_ssim, avg_psnr, avg_lpips

    def visualize(self, img_path):
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

            # 每1个epoch可视化一次
            if epoch % 100 == 0:
                self.visualize(test_img)


if __name__ == "__main__":
    print(f"当前使用设备: {device}")
    print(torch.cuda.is_available())  # 应输出 True
    print(torch.cuda.device_count())  # 应输出 1
    print(torch.cuda.get_device_name(0))  # 应输出 RTX 3060
    # 配置参数：是否加载预训练模型
    LOAD_DENOISER = True  # 设置为True加载已训练的去噪模型，False则重新训练
    CHECKPOINT_TO_RESUME = None  # 主模型检查点路径，None表示从头开始

    # 初始化训练器并开始训练
    trainer = RetinexTrainer(resume_checkpoint=CHECKPOINT_TO_RESUME, load_denoiser=LOAD_DENOISER)
    trainer.train()
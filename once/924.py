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

# -------------------------- 1. 数据集路径（用户指定，未修改） --------------------------
TEST_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high"
TRAIN_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"
TRAIN_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"
TEST_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"


# -------------------------- 2. 基础工具函数（保留Uretinex原始架构） --------------------------
def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    """Uretinex原始卷积层封装，未修改"""
    return nn.Conv2d(
        in_channels=in_c, out_channels=out_c,
        kernel_size=k, stride=s, padding=p,
        dilation=dilation, groups=groups
    )


def get_batchnorm_layer(opts):
    """Uretinex原始批归一化层封装，未修改"""
    if opts.norm_layer == "batch":
        return nn.BatchNorm2d
    elif opts.norm_layer == "spectral_instance":
        return nn.InstanceNorm2d
    else:
        print("not implemented")
        exit()


class Identity(nn.Module):
    """Uretinex原始恒等映射层，未修改"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# -------------------------- 3. 迭代Uretinex分解（100%保留原始P/Q更新逻辑） --------------------------
class P(nn.Module):
    """原始P求解模块：min(P) = ||I-PQ||^2 + γ||P-R||^2 → 公式未修改"""

    def __init__(self):
        super().__init__()

    def forward(self, I, Q, R, gamma):
        return ((I * Q + gamma * R) / (gamma + Q * Q + 1e-8))  # 仅加1e-8防除零，公式不变


class Q(nn.Module):
    """原始Q求解模块：min(Q) = ||I-PQ||^2 + λ||Q-L||^2 → 通道拆分+公式未修改"""

    def __init__(self):
        super().__init__()

    def forward(self, I, P, L, lamda):
        # 原始R/G/B通道拆分逻辑，未修改
        IR = I[:, 0:1, :, :]
        IG = I[:, 1:2, :, :]
        IB = I[:, 2:3, :, :]
        PR = P[:, 0:1, :, :]
        PG = P[:, 1:2, :, :]
        PB = P[:, 2:3, :, :]
        # 原始公式，未修改
        numerator = (IR * PR + IG * PG + IB * PB) + lamda * L
        denominator = (PR * PR + PG * PG + PB * PB) + lamda + 1e-8
        return numerator / denominator


class Decom(nn.Module):
    """Uretinex原始分解网络（初始化P0/L0），结构未修改"""

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
        R = torch.clamp(R, 0.0, 1.0)
        L = torch.clamp(L, 0.0, 1.0)
        return R, L


class SELayer(nn.Module):
    """Uretinex原始SE层，结构未修改"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HalfDnCNNSE(nn.Module):
    """Uretinex原始反射层恢复网络，结构未修改"""

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if self.opts.concat_L:
            self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = get_conv2d_layer(in_c=1, out_c=32, k=3, s=1, p=1)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.conv1 = get_conv2d_layer(in_c=3, out_c=64, k=3, s=1, p=1)
            self.relu1 = nn.ReLU(inplace=True)
        self.se_layer = SELayer(channel=64)
        self.conv3 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = get_conv2d_layer(in_c=64, out_c=3, k=3, s=1, p=1)

    def forward(self, r, l):
        if self.opts.concat_L:
            r_fs = self.relu1(self.conv1(r))
            l_fs = self.relu2(self.conv2(l))
            inf = torch.cat([r_fs, l_fs], dim=1)
            se_inf = self.se_layer(inf)
        else:
            r_fs = self.relu1(self.conv1(r))
            se_inf = self.se_layer(r_fs)
        x1 = self.relu3(self.conv3(se_inf))
        x2 = self.relu4(self.conv4(x1))
        x3 = self.relu5(self.conv5(x2))
        x4 = self.relu6(self.conv6(x3))
        x5 = self.relu7(self.conv7(x4))
        n = self.conv8(x5)
        r_restore = r + n
        return r_restore


class IterativeUretinex(nn.Module):
    """迭代Uretinex分解（保留原始多轮更新+反射层恢复），逻辑未修改"""

    def __init__(self, opts, unfolding_round=3):
        super().__init__()
        self.opts = opts
        self.unfolding_round = unfolding_round  # 迭代轮数
        self.init_decom = Decom()  # 初始化P0/L0
        self.P_solver = P()  # P求解器
        self.Q_solver = Q()  # Q求解器
        self.r_restore_net = HalfDnCNNSE(opts)  # 原始反射层恢复网络

    def forward(self, input_img):
        """原始迭代逻辑，未修改"""
        # 1. 初始化P0、L0
        P, Q = self.init_decom(input_img)
        # 2. 多轮迭代更新P、Q
        for t in range(self.unfolding_round):
            if t > 0:  # 从第2轮开始更新参数
                w_p = self.opts.gamma + self.opts.Roffset * t
                w_q = self.opts.lamda + self.opts.Loffset * t
                P = self.P_solver(I=input_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q_solver(I=input_img, P=P, L=L, lamda=w_q)
            # 反射层恢复（原始逻辑）
            R = self.r_restore_net(r=P, l=Q)
            L = Q  # Q即照明层
        return R, L   #####********


# -------------------------- 4. Noise2noise（100%保留原始代码逻辑，未修改） --------------------------
class ConvBlock(nn.Module):
    """Noise2noise原始卷积块，结构+前向未修改"""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, pad=1, use_act=True):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=pad)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """原始前向逻辑：conv→bn→act（可选），未修改"""
        op = self.bn(self.conv(x))
        if self.use_act:
            return self.act(op)
        else:
            return op


class ResBlock(nn.Module):
    """Noise2noise原始残差块，结构+前向未修改"""

    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.block1 = ConvBlock(input_channels, output_channels, kernel_size)
        self.block2 = ConvBlock(input_channels, output_channels, kernel_size, use_act=False)

    def forward(self, x):
        """原始残差逻辑：x + (block1→block2)，未修改"""
        return x + self.block2(self.block1(x))


class SRResnet(nn.Module):
    """Noise2noise原始SRResnet网络，结构+前向未修改"""

    def __init__(self, input_channels, output_channels, res_layers=16):
        super(SRResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.resl = nn.Sequential(*[ResBlock(output_channels, output_channels, 3) for _ in range(res_layers)])
        self.conv2 = ConvBlock(output_channels, output_channels, 3, use_act=False)
        self.conv3 = nn.Conv2d(output_channels, input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        """原始前向逻辑，未修改"""
        _op1 = self.act(self.conv1(input))
        _op2 = self.conv2(self.resl(_op1))
        op = self.conv3(torch.add(_op1, _op2))
        return op


class NoisyDataset(Dataset):
    """Noise2noise原始数据集加载（含所有噪声类型），逻辑未修改"""

    def __init__(self, root_dir, crop_size=128, train_noise_model=('gaussian', 50), clean_targ=False):
        super(NoisyDataset, self).__init__()
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.clean_targ = clean_targ
        self.noise = train_noise_model[0]
        self.noise_param = train_noise_model[1]
        self.imgs = os.listdir(root_dir)

    def _random_crop_to_size(self, imgs):
        """原始随机裁剪逻辑，未修改"""
        w, h = imgs[0].size
        assert w >= self.crop_size and h >= self.crop_size, 'Cannot be croppped. Invalid size'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 2)
        j = np.random.randint(0, w - self.crop_size + 2)
        for img in imgs:
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))
        return cropped_imgs

    def _add_gaussian_noise(self, image):
        """原始高斯噪声添加，未修改"""
        w, h = image.size
        c = len(image.getbands())
        std = np.random.uniform(0, self.noise_param)
        _n = np.random.normal(0, std, (h, w, c))
        noisy_image = np.array(image) + _n
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return {'image': Image.fromarray(noisy_image), 'mask': None, 'use_mask': False}

    def _add_poisson_noise(self, image):
        """原始泊松噪声添加，未修改"""
        noise_mask = np.random.poisson(np.array(image))
        return {'image': noise_mask.astype(np.uint8), 'mask': None, 'use_mask': False}

    def _add_m_bernoulli_noise(self, image):
        """原始乘性伯努利噪声添加，未修改"""
        sz = np.array(image).shape[0]
        prob_ = random.uniform(0, self.noise_param)
        mask = np.random.choice([0, 1], size=(sz, sz), p=[prob_, 1 - prob_])
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return {'image': np.multiply(image, mask).astype(np.uint8), 'mask': mask.astype(np.uint8), 'use_mask': True}

    def _add_text_overlay(self, image):
        """原始文本叠加噪声添加，未修改"""
        assert self.noise_param < 1, 'Text parameter should be probability of occupancy'
        w, h = image.size
        c = len(image.getbands())
        # 原始字体路径逻辑
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
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(random.choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break
        return {'image': text_img, 'mask': None, 'use_mask': False}

    def corrupt_image(self, image):
        try:
            if self.noise == 'gaussian':
                return self._add_gaussian_noise(image)
            elif self.noise == 'poisson':
                return self._add_poisson_noise(image)
            elif self.noise == 'multiplicative_bernoulli':
                return self._add_m_bernoulli_noise(image)
            elif self.noise == 'text':
                return self._add_text_overlay(image)
            else:
                raise ValueError('No such image corruption supported')
        except Exception as e:
            print(f"Noise generation failed: {e}")
            # 生成默认有效数据（避免返回None）
            return {'image': image, 'mask': None, 'use_mask': False}

    def __getitem__(self, index):
        """原始数据加载逻辑，未修改"""
        img_path = os.path.join(self.root_dir, self.imgs[index])
        image = Image.open(img_path).convert('RGB')
        # 随机裁剪
        if self.crop_size > 0:
            image = self._random_crop_to_size([image])[0]
        # 生成噪声图像1（输入）
        source_img_dict = self.corrupt_image(image)
        source_img_dict['image'] = tvF.to_tensor(source_img_dict['image'])
        if source_img_dict['use_mask']:
            source_img_dict['mask'] = tvF.to_tensor(source_img_dict['mask'])
        # 生成目标（干净/噪声）
        if self.clean_targ:
            target = tvF.to_tensor(image)
        else:
            _target_dict = self.corrupt_image(image)
            target = tvF.to_tensor(_target_dict['image'])
        # 返回格式（原始顺序）
        image_np = np.array(image).astype(np.uint8)
        if source_img_dict['use_mask']:
            return [source_img_dict['image'], source_img_dict['mask'], target, image_np]
        else:
            return [source_img_dict['image'], target, image_np]

    def __len__(self):
        return len(self.imgs)


# -------------------------- 5. Zero-DCE（100%保留原始代码逻辑，未修改） --------------------------
class enhance_net_nopool(nn.Module):
    """Zero-DCE原始网络，结构+前向未修改"""

    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32  # 原始通道数，未修改
        # 原始7层卷积，未修改
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)  # 输出24通道（8*3）
        # 原始池化/上采样（虽未使用，但保留代码）
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        """Zero-DCE原始8轮迭代增强逻辑，未修改"""
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))  # 24通道输出
        # 原始8轮迭代增强，未修改
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        # 原始返回值：中间增强图+最终增强图+增强系数，未修改
        return enhance_image_1, enhance_image, r


# -------------------------- 6. Zero-DCE原始损失函数（100%保留，未修改） --------------------------
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
        E = (D_left + D_right + D_up + D_down)
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


# -------------------------- 7. 数据集适配（整合LOL与Noise2noise噪声逻辑） --------------------------
class LOLNoisyDataset(Dataset):
    """LOL低光数据集+Noise2noise噪声添加（适配反射层去噪）"""

    def __init__(self, low_dir, normal_dir, crop_size=128, train_noise_model=('gaussian', 50)):
        self.low_dir = low_dir
        self.normal_dir = normal_dir
        self.crop_size = crop_size
        self.noise_model = train_noise_model
        self.filenames = [f for f in os.listdir(low_dir) if f.endswith((".png", ".jpg"))]
        # 初始化Noise2noise噪声生成器
        self.noisy_gen = NoisyDataset(
            root_dir=normal_dir,  # 用正常光图像生成干净反射层标签
            crop_size=crop_size,
            train_noise_model=train_noise_model,
            clean_targ=True
        )

    def _crop_image(self, img):
        """裁剪图像（适配LOL数据集）"""
        w, h = img.size
        if min(w, h) < self.crop_size:
            img = tvF.resize(img, (self.crop_size, self.crop_size))
        i = np.random.randint(0, h - self.crop_size + 2)
        j = np.random.randint(0, w - self.crop_size + 2)
        return tvF.crop(img, i, j, self.crop_size, self.crop_size)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low_path = os.path.join(self.low_dir, filename)
        normal_path = os.path.join(self.normal_dir, filename)

        # 1. 确保文件存在
        if not (os.path.exists(low_path) and os.path.exists(normal_path)):
            print(f"Warning: Missing files for {filename}, skipping...")
            # 跳转到下一个有效样本（避免返回None）
            return self.__getitem__((idx + 1) % len(self))

        try:
            # 2. 确保图像能正常打开
            low_img = Image.open(low_path).convert("RGB")
            normal_img = Image.open(normal_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {filename}: {e}, skipping...")
            return self.__getitem__((idx + 1) % len(self))

        # 3. 裁剪图像（确保裁剪后有效）
        try:
            low_img = self._crop_image(low_img)
            normal_img = self._crop_image(normal_img)
        except Exception as e:
            print(f"Error cropping {filename}: {e}, skipping...")
            return self.__getitem__((idx + 1) % len(self))

        # 4. 生成带噪反射层（确保噪声生成有效）
        try:
            noisy_dict = self.noisy_gen.corrupt_image(normal_img)
            # 强制检查噪声图像是否有效
            if noisy_dict['image'] is None:
                raise ValueError("Noisy image is None")
        except Exception as e:
            print(f"Error generating noise for {filename}: {e}, skipping...")
            return self.__getitem__((idx + 1) % len(self))

        # 5. 转换为Tensor（确保转换后有效）
        try:
            low_tensor = tvF.to_tensor(low_img).clip(0.0, 1.0)  # 强制clip到0-1
            normal_tensor = tvF.to_tensor(normal_img).clip(0.0, 1.0)
            noisy_reflect = tvF.to_tensor(noisy_dict['image']).clip(0.0, 1.0)

            # 关键修改：当use_mask为False时，返回空张量而非None
            if noisy_dict['use_mask']:
                mask = tvF.to_tensor(noisy_dict['mask'])
            else:
                # 创建一个与图像同形状的空张量（1通道，用于占位）
                mask = torch.zeros(1, self.crop_size, self.crop_size)  # 确保是张量类型

        except Exception as e:
            print(f"Error converting to tensor for {filename}: {e}, skipping...")
            return self.__getitem__((idx + 1) % len(self))

        # 6. 最终校验：确保没有None
        if any(x is None for x in [low_tensor, normal_tensor, noisy_reflect, normal_tensor]):
            print(f"Warning: None value found in {filename}, skipping...")
            return self.__getitem__((idx + 1) % len(self))

        return (low_tensor, normal_tensor, noisy_reflect, normal_tensor,
                mask, noisy_dict['use_mask'], filename)

    def __len__(self):
        return len(self.filenames)


def get_lol_dataloaders(batch_size=8, crop_size=128, noise_model=('gaussian', 50)):
    """获取LOL数据集加载器（含噪声）"""
    # 训练集
    train_dataset = LOLNoisyDataset(
        TRAIN_LOW_PATH, TRAIN_NORMAL_PATH,
        crop_size=crop_size, train_noise_model=noise_model
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    # 测试集（无噪声，仅评估）
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = LOLDataset(TEST_LOW_PATH, TEST_NORMAL_PATH, test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, test_loader


class LOLDataset(Dataset):
    """基础LOL数据集（测试用）"""

    def __init__(self, low_dir, normal_dir, transform):
        self.low_dir = low_dir
        self.normal_dir = normal_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(low_dir) if f.endswith((".png", ".jpg"))]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low_img = Image.open(os.path.join(self.low_dir, filename)).convert("RGB")
        normal_img = Image.open(os.path.join(self.normal_dir, filename)).convert("RGB")
        if self.transform:
            low_img = self.transform(low_img)
            normal_img = self.transform(normal_img)
        return low_img, normal_img, filename

    def __len__(self):
        return len(self.filenames)


# -------------------------- 8. 主模型（整合三模块，未修改子模块逻辑） --------------------------
class LowLightEnhanceModel(nn.Module):
    """完整模型：迭代Uretinex+Noise2noise+Zero-DCE（子模块逻辑未修改）"""

    def __init__(self, uretinex_opts, unfolding_round=3, noise2noise_res_layers=16):
        super().__init__()
        # 1. 迭代Uretinex分解（原始模块）
        self.uretinex_decom = IterativeUretinex(uretinex_opts, unfolding_round=unfolding_round)
        # 2. Noise2noise反射层去噪（原始SRResnet）
        self.noise2noise_denoise = SRResnet(
            input_channels=3, output_channels=3, res_layers=noise2noise_res_layers
        )
        # 3. Zero-DCE照明层增强（适配：先扩3通道，增强后取单通道）
        self.zerodce_illum = enhance_net_nopool()  # 原始Zero-DCE模块

    def forward(self, x_low):
        """端到端前向（子模块调用，未修改逻辑）"""
        # 1. 迭代Uretinex分解：获取反射层R、照明层L
        R, L = self.uretinex_decom(x_low)
        # 2. Noise2noise反射层去噪（原始前向）
        R_denoised = self.noise2noise_denoise(R)
        R_denoised = torch.clamp(R_denoised, 0, 1)
        # 3. Zero-DCE照明层增强（适配单通道→3通道→单通道）
        L_3ch = L.repeat(1, 3, 1, 1)  # 原始Zero-DCE需3通道输入
        _, L_enhanced_3ch, _ = self.zerodce_illum(L_3ch)  # 调用原始Zero-DCE前向
        L_enhanced = torch.mean(L_enhanced_3ch, dim=1, keepdim=True)  # 3通道→1通道
        L_enhanced = torch.clamp(L_enhanced, 0.1, 0.9)
        # 4. 图像重建：R_denoised * L_enhanced（扩3通道相乘）
        L_enhanced_3ch_final = L_enhanced.repeat(1, 3, 1, 1)
        x_enhanced = R_denoised * L_enhanced_3ch_final
        x_enhanced = torch.clamp(x_enhanced, 0, 1)
        # 返回所有中间结果（可视化用）
        return {
            "x_low": x_low, "R": R, "L": L,
            "R_denoised": R_denoised, "L_enhanced": L_enhanced,
            "x_enhanced": x_enhanced
        }


# -------------------------- 9. 损失函数整合（保留各模块原始损失） --------------------------
def compute_total_loss(model_outputs, x_normal, noisy_reflect, clean_reflect, use_mask, mask, uretinex_opts):
    """总损失：Uretinex分解损失+Noise2noise去噪损失+Zero-DCE照明损失+重建损失"""
    # 1. Uretinex分解损失（原始逻辑）
    R = model_outputs["R"]
    L = model_outputs["L"]
    # 分解保真损失：x_low ≈ R*L
    decom_recon_loss = nn.MSELoss()(R * L, model_outputs["x_low"])
    # 反射层一致性损失：R ≈ 正常光反射层（用正常光图像分解的R作为标签）
    with torch.no_grad():
        R_normal, _ = model_outputs["uretinex_decom"](x_normal)
    decom_consist_loss = nn.MSELoss()(R, R_normal)
    # 照明层TV损失
    decom_tv_loss = L_TV()(L) * uretinex_opts.tv_weight
    total_decom_loss = decom_recon_loss + 0.5 * decom_consist_loss + decom_tv_loss

    # 2. Noise2noise去噪损失（原始逻辑：带噪反射层→干净反射层）
    R_denoised = model_outputs["R_denoised"]
    if use_mask:
        denoise_loss = nn.MSELoss()(mask * R_denoised, mask * clean_reflect)
    else:
        denoise_loss = nn.MSELoss()(R_denoised, clean_reflect)

    # 3. Zero-DCE照明增强损失（原始逻辑）
    L_enhanced = model_outputs["L_enhanced"]
    L_normal = torch.mean(x_normal, dim=1, keepdim=True)  # 正常光照明层标签
    exp_loss = L_exp(patch_size=16, mean_val=0.5)(L_enhanced)
    spa_loss = torch.mean(L_spa()(L_normal, L_enhanced))
    color_loss = L_color()(L_enhanced.repeat(1, 3, 1, 1))  # 扩3通道算颜色损失
    tv_loss = L_TV()(L_enhanced)
    total_illum_loss = exp_loss + 0.1 * spa_loss + 0.01 * color_loss + 0.01 * tv_loss

    # 4. 最终重建损失：增强图≈正常光图
    recon_final_loss = nn.L1Loss()(model_outputs["x_enhanced"], x_normal)

    # 总损失（关键修改：确保所有损失项都是标量，并最终求和为标量）
    total_loss = (0.2 * total_decom_loss + 0.2 * denoise_loss + 0.3 * total_illum_loss + 0.3 * recon_final_loss)

    # 强制确保 total_loss 是标量（添加均值操作，针对可能的维度残留）
    if total_loss.dim() > 0:
        total_loss = total_loss.mean()

    return (total_loss, total_decom_loss, denoise_loss, total_illum_loss, recon_final_loss)


# -------------------------- 10. 评估与可视化（保留模块级细节） --------------------------
def calculate_metrics(x_enhanced, x_normal, device="cuda"):
    """计算SSIM、PSNR、LPIPS（未修改指标逻辑）"""
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    b = x_enhanced.size(0)
    ssim_sum, psnr_sum, lpips_sum = 0.0, 0.0, 0.0
    with torch.no_grad():
        for i in range(b):
            # 转numpy（0-255）
            enh_np = (x_enhanced[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            norm_np = (x_normal[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            # SSIM
            ssim_sum += ssim(enh_np, norm_np, channel_axis=2, data_range=255)
            # PSNR
            psnr_sum += psnr(enh_np, norm_np, data_range=255)
            # LPIPS（-1~1）
            enh_lpips = (x_enhanced[i].unsqueeze(0) * 2) - 1
            norm_lpips = (x_normal[i].unsqueeze(0) * 2) - 1
            lpips_sum += lpips_model(enh_lpips, norm_lpips).item()
    return (ssim_sum / b, psnr_sum / b, lpips_sum / b)


def visualize_modules(outputs, x_normal, filename, save_dir="./visualization"):
    """模块级可视化：输入→分解→去噪→增强→最终结果（未修改可视化逻辑）"""
    os.makedirs(save_dir, exist_ok=True)

    # tensor转numpy（0-1）
    def tensor2np(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def single2three(tensor):
        img = tensor.squeeze(0).squeeze(0).cpu().numpy()
        return np.stack([img, img, img], axis=2)

    # 提取中间结果
    x_low = tensor2np(outputs["x_low"])
    R = tensor2np(outputs["R"])
    L = single2three(outputs["L"])
    R_denoised = tensor2np(outputs["R_denoised"])
    L_enhanced = single2three(outputs["L_enhanced"])
    x_enhanced = tensor2np(outputs["x_enhanced"])
    x_normal = tensor2np(x_normal)

    # 绘制子图（保留原始模块顺序）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    # 1. 输入低光图
    axes[0].imshow(x_low)
    axes[0].set_title("Input Low-Light Image")
    axes[0].axis("off")
    # 2. Uretinex分解反射层
    axes[1].imshow(R)
    axes[1].set_title("Decomposed Reflectance (Iterative Uretinex)")
    axes[1].axis("off")
    # 3. Uretinex分解照明层
    axes[2].imshow(L)
    axes[2].set_title("Decomposed Illumination (Iterative Uretinex)")
    axes[2].axis("off")
    # 4. Noise2noise去噪反射层
    axes[3].imshow(R_denoised)
    axes[3].set_title("Denoised Reflectance (Noise2noise)")
    axes[3].axis("off")
    # 5. Zero-DCE增强照明层
    axes[4].imshow(L_enhanced)
    axes[4].set_title("Enhanced Illumination (Zero-DCE)")
    axes[4].axis("off")
    # 6. 最终增强图vs正常光图
    axes[5].imshow(np.hstack([x_enhanced, x_normal]))
    axes[5].set_title("Enhanced Image (Left) vs Normal Image (Right)")
    axes[5].axis("off")

    # 保存
    save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_modules.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------- 11. 训练与测试主函数（整合三模块训练逻辑） --------------------------
def train(args, model, train_loader, optimizer, scheduler, device):
    """端到端训练（保留各模块原始训练逻辑）"""
    model.train()
    best_loss = float("inf")
    # 初始化Zero-DCE损失类（原始参数）
    l_spa = L_spa()
    l_exp = L_exp(patch_size=args.patch_size, mean_val=args.mean_val)
    l_color = L_color()
    l_tv = L_TV()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss, decom_loss, denoise_loss, illum_loss, recon_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, (x_low, x_normal, noisy_reflect, clean_reflect, mask, use_mask, _) in enumerate(train_loader):
            # 数据移至设备
            x_low = x_low.to(device)
            x_normal = x_normal.to(device)
            print(f"x_low range: {x_low.min():.4f} ~ {x_low.max():.4f}")
            print(f"x_normal range: {x_normal.min():.4f} ~ {x_normal.max():.4f}")
            noisy_reflect = noisy_reflect.to(device)
            clean_reflect = clean_reflect.to(device)
            if use_mask[0]:  # 处理掩码（原始Noise2noise逻辑）
                mask = mask.to(device)
            else:
                mask = None

            # 前向传播
            outputs = model(x_low)
            # 补充Uretinex模块引用（用于计算分解损失）
            outputs["uretinex_decom"] = model.uretinex_decom

            # 计算总损失（保留各模块原始损失）
            loss, d_loss, n_loss, i_loss, r_loss = compute_total_loss(
                outputs, x_normal, noisy_reflect, clean_reflect, use_mask[0], mask, args
            )

            # 反向传播（原始逻辑）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            batch_size = x_low.size(0)
            total_loss += loss.item() * batch_size
            decom_loss += d_loss.item() * batch_size
            denoise_loss += n_loss.item() * batch_size
            illum_loss += i_loss.item() * batch_size
            recon_loss += r_loss.item() * batch_size

        # 平均损失
        avg_total = total_loss / len(train_loader.dataset)
        avg_d = decom_loss / len(train_loader.dataset)
        avg_n = denoise_loss / len(train_loader.dataset)
        avg_i = illum_loss / len(train_loader.dataset)
        avg_r = recon_loss / len(train_loader.dataset)

        # 学习率调度（原始Noise2noise逻辑）
        scheduler.step(avg_total)

        # 保存最佳模型
        if avg_total < best_loss:
            best_loss = avg_total
            os.makedirs(args.ckpt_dir, exist_ok=True)
            # 正确写法：保存到 args.ckpt_dir 目录下的 best_model.pth
            save_path = os.path.join(args.ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存至: {save_path}")  # 打印保存路径，方便确认

        # 打印日志（保留各模块损失细节）
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch + 1}/{args.epochs}] | Time: {epoch_time:.2f}s | "
              f"Total Loss: {avg_total:.6f} | Decom Loss: {avg_d:.6f} | "
              f"Denoise Loss: {avg_n:.6f} | Illum Loss: {avg_i:.6f} | "
              f"Recon Loss: {avg_r:.6f}")


def test(args, model, test_loader, device):
    """测试与评估（保留原始指标逻辑）"""
    # 加载最佳模型
    model_path = os.path.join(args.ckpt_dir, "best_model.pth")
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "E:\\大创\\924")))
    model.eval()

    # 指标累加
    total_ssim, total_psnr, total_lpips = 0.0, 0.0, 0.0
    num_samples = len(test_loader.dataset)

    with torch.no_grad():
        for idx, (x_low, x_normal, filename) in enumerate(test_loader):
            x_low, x_normal = x_low.to(device), x_normal.to(device)
            print(f"Processing {idx + 1}/{num_samples}: {filename[0]}")

            # 前向传播（获取所有模块输出）
            outputs = model(x_low)
            x_enhanced = outputs["x_enhanced"]

            # 计算指标（原始逻辑）
            ssim_val, psnr_val, lpips_val = calculate_metrics(x_enhanced, x_normal, device)
            total_ssim += ssim_val
            total_psnr += psnr_val
            total_lpips += lpips_val

            # 模块可视化（保留原始细节）
            visualize_modules(outputs, x_normal, filename[0], save_dir=args.vis_dir)

            # 单样本日志
            print(f"Sample {filename[0]} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.4f} | LPIPS: {lpips_val:.4f}")

    # 平均指标
    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples
    avg_lpips = total_lpips / num_samples

    # 保存结果（原始格式）
    with open(os.path.join(args.vis_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Test Metrics (Average over {num_samples} samples):\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")

    # 最终日志
    print("\n" + "=" * 50)
    print(f"Test Results (Average):")
    print(f"SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.4f} | LPIPS: {avg_lpips:.4f}")
    print("=" * 50)


# -------------------------- 12. 入口函数（未修改参数逻辑） --------------------------
if __name__ == "__main__":
    # 命令行参数（整合三模块参数，未修改原始参数含义）
    parser = argparse.ArgumentParser(description="Low Light Enhancement (Uretinex+Noise2noise+Zero-DCE)")
    # 通用参数
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (Noise2noise原始默认1000，此处适配)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (原始Noise2noise默认32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (原始Noise2noise默认0.001)")
    parser.add_argument("--crop_size", type=int, default=64, help="Crop size (原始Noise2noise默认64)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID (原始逻辑)")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt", help="Checkpoint dir (原始逻辑)")
    parser.add_argument("--vis_dir", type=str, default="./test_visualization", help="Visualization dir")
    # Uretinex参数（原始含义）
    parser.add_argument("--unfolding_round", type=int, default=3, help="Uretinex iterative rounds")
    parser.add_argument("--gamma", type=float, default=0.1, help="P's lambda (原始参数)")
    parser.add_argument("--lamda", type=float, default=0.1, help="Q's lambda (原始参数)")
    parser.add_argument("--Roffset", type=float, default=0.05, help="gamma increment (原始参数)")
    parser.add_argument("--Loffset", type=float, default=0.05, help="lamda increment (原始参数)")
    parser.add_argument("--tv_weight", type=float, default=0.01, help="TV loss weight (原始参数)")
    parser.add_argument("--norm_layer", type=str, default="batch", help="Norm layer (原始参数)")
    parser.add_argument("--concat_L", type=bool, default=False, help="Concat L to R (原始参数)")
    # Zero-DCE参数（原始含义）
    parser.add_argument("--patch_size", type=int, default=16, help="L_exp patch size (原始参数)")
    parser.add_argument("--mean_val", type=float, default=0.5, help="L_exp target mean (原始参数)")
    # Noise2noise参数（原始含义）
    parser.add_argument("--noise_model", type=tuple, default=('gaussian', 50), help="Noise type (原始参数)")
    parser.add_argument("--noise2noise_res_layers", type=int, default=16, help="SRResnet res layers (原始参数)")
    args = parser.parse_args()

    # 设备配置（原始逻辑）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载（整合LOL与Noise2noise）
    train_loader, test_loader = get_lol_dataloaders(
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        noise_model=args.noise_model
    )
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # 模型初始化（未修改子模块）
    model = LowLightEnhanceModel(
        uretinex_opts=args,
        unfolding_round=args.unfolding_round,
        noise2noise_res_layers=args.noise2noise_res_layers
    ).to(device)
    print(
        f"Model initialized: {args.unfolding_round} Uretinex rounds, {args.noise2noise_res_layers} Noise2noise res layers")

    # 优化器（原始Noise2noise逻辑）
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 学习率调度（原始Noise2noise逻辑）
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.epochs // 4, factor=0.5, verbose=True)

    # 训练（保留原始逻辑）
    print("\nStart Training...")
    train(args, model, train_loader, optimizer, scheduler, device)

    # 测试（保留原始逻辑）
    print("\nStart Testing...")
    test(args, model, test_loader, device)

    print("\nAll Done!")
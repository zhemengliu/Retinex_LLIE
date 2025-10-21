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
        op = torch.clamp(op, 0.0, 1.0)
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


if __name__== '__main__':
    parser = argparse.ArgumentParser(description="Low Light Enhancement (Uretinex+Noise2noise+Zero-DCE)")
    parser.add_argument("--noise_model", type=tuple, default=('gaussian', 50), help="Noise type (原始参数)")
    parser.add_argument("--noise2noise_res_layers", type=int, default=16, help="SRResnet res layers (原始参数)")
    args = parser.parse_args()
    x = torch.rand(1,3,128,128).cuda()
    net=SRResnet(input_channels=3, output_channels=3).cuda()
    op=net(x)
    print(op.shape)
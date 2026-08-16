import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob


# -------------------------- 配置参数定义 --------------------------
class Options:
    def __init__(self):
        # 数据路径
        self.TRAIN_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"
        self.TRAIN_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"
        self.TEST_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"
        self.TEST_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high"
        self.CHECKPOINT_DIR = "./checkpoints"

        # 训练参数
        self.epochs = 1
        self.batch_size = 8
        self.lr = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型参数
        self.norm_layer = "batch"
        self.concat_L = True
        self.gamma = 0.1
        self.lamda = 0.1


# -------------------------- 基础网络层 --------------------------
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_batchnorm_layer(opts):
    if opts.norm_layer == "batch":
        return nn.BatchNorm2d
    elif opts.norm_layer == "instance":
        return nn.InstanceNorm2d
    else:
        raise NotImplementedError(f"Norm layer {opts.norm_layer} not implemented")


def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=k,
        stride=s,
        padding=p,
        dilation=dilation,
        groups=groups
    )


def get_deconv2d_layer(in_c, out_c, k=1, s=1, p=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
    )


# -------------------------- 分解网络 --------------------------
class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(32, 4, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.decom(input)
        R = output[:, 0:3, :, :]  # 反射分量
        L = output[:, 3:4, :, :]  # 照明分量
        return R, L


# -------------------------- 展开模块 --------------------------
class P(nn.Module):
    def forward(self, I, Q, R, gamma):
        return (I * Q + gamma * R) / (gamma + Q * Q + 1e-8)


class Q(nn.Module):
    def forward(self, I, P, L, lamda):
        IR, IG, IB = I[:, 0:1, :, :], I[:, 1:2, :, :], I[:, 2:3, :, :]
        PR, PG, PB = P[:, 0:1, :, :], P[:, 1:2, :, :], P[:, 2:3, :, :]
        numerator = IR * PR + IG * PG + IB * PB + lamda * L
        denominator = (PR ** 2 + PG ** 2 + PB ** 2) + lamda + 1e-8
        return numerator / denominator


# -------------------------- 反射分量恢复网络 --------------------------
class SELayer(nn.Module):
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
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        if self.opts.concat_L:
            self.conv1 = get_conv2d_layer(3, 32, 3, 1, 1)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = get_conv2d_layer(1, 32, 3, 1, 1)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.conv1 = get_conv2d_layer(3, 64, 3, 1, 1)
            self.relu1 = nn.ReLU(inplace=True)

        self.se_layer = SELayer(64)
        self.conv3 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = get_conv2d_layer(64, 3, 3, 1, 1)

    def forward(self, r, l):
        if self.opts.concat_L:
            r_fs = self.relu1(self.conv1(r))
            l_fs = self.relu2(self.conv2(l))
            inf = torch.cat([r_fs, l_fs], dim=1)
        else:
            inf = self.relu1(self.conv1(r))
        se_inf = self.se_layer(inf)
        x = self.relu3(self.conv3(se_inf))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        n = self.conv8(x)
        r_restore = r + n
        return r_restore


# -------------------------- 新增增强网络（enhance_net_nopool） --------------------------
class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
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
        return enhance_image_1, enhance_image, r


# -------------------------- 新增损失函数（修复L_spa） --------------------------
class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        # 初始化卷积核为Parameter，不指定设备
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        # 获取输入数据所在设备
        device = org.device

        # 使用detach()和新Parameter保持类型，避免类型错误
        weight_left = nn.Parameter(self.weight_left.detach().to(device), requires_grad=False)
        weight_right = nn.Parameter(self.weight_right.detach().to(device), requires_grad=False)
        weight_up = nn.Parameter(self.weight_up.detach().to(device), requires_grad=False)
        weight_down = nn.Parameter(self.weight_down.detach().to(device), requires_grad=False)

        b, c, h, w = org.shape
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        # 使用device参数确保张量在正确设备上
        weight_diff = torch.max(torch.tensor([1.0], device=device) + 10000 * torch.min(
            org_pool - torch.tensor([0.3], device=device),
            torch.tensor([0.0], device=device)
        ), torch.tensor([0.5], device=device))

        E_1 = torch.mul(torch.sign(enhance_pool - torch.tensor([0.5], device=device)),
                        enhance_pool - org_pool)

        # 使用迁移后的权重进行卷积
        D_org_letf = F.conv2d(org_pool, weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        return (D_left + D_right + D_up + D_down).mean()


class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        device = x.device
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        return torch.mean(torch.pow(mean - torch.tensor([self.mean_val], device=device), 2))


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
        return torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5).mean()


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


# -------------------------- 完整增强模型（整合后） --------------------------
class LowLightEnhancer(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.decom = Decom().to(opts.device)
        self.p_module = P()
        self.q_module = Q()
        self.refine_r = HalfDnCNNSE(opts).to(opts.device)
        self.enhance_net = enhance_net_nopool().to(opts.device)  # 新增增强网络

    def forward(self, low_img, is_train=True):
        R_low, L_low = self.decom(low_img)

        if is_train:
            # 原有流程
            P_out = self.p_module(low_img, L_low, R_low, self.opts.gamma)
            Q_out = self.q_module(low_img, P_out, L_low, self.opts.lamda)
            R_restore = self.refine_r(P_out, Q_out)

            # 新增增强网络处理
            _, enhance_image, r = self.enhance_net(low_img)
            return R_low, L_low, R_restore, P_out, Q_out, enhance_image
        else:
            # 测试时同时输出两种增强结果
            R_restore = self.refine_r(R_low, L_low)
            L_enhance = torch.clamp(L_low * 1.5, 0, 1)
            enhance_img1 = R_restore * L_enhance  # 原有增强结果

            # 新增增强网络结果
            _, enhance_img2, _ = self.enhance_net(low_img)
            return R_low, L_low, R_restore, enhance_img1, enhance_img2


# -------------------------- 数据加载器 --------------------------
class LOLDataset(Dataset):
    def __init__(self, low_dir, normal_dir, transform=None):
        self.low_paths = sorted(glob.glob(os.path.join(low_dir, "*.png")))
        self.normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
        self.transform = transform
        assert len(self.low_paths) == len(self.normal_paths), "低光与正常图像数量不匹配"

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_paths[idx]).convert("RGB")
        normal_img = Image.open(self.normal_paths[idx]).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            normal_img = self.transform(normal_img)
        return low_img, normal_img


# -------------------------- 训练函数（整合新增损失） --------------------------
def train_model(opts):
    os.makedirs(opts.CHECKPOINT_DIR, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_dataset = LOLDataset(
        low_dir=opts.TRAIN_LOW_PATH,
        normal_dir=opts.TRAIN_NORMAL_PATH,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=2
    )

    # 初始化模型和损失
    model = LowLightEnhancer(opts).to(opts.device)
    criterion_mse = nn.MSELoss()
    criterion_spa = L_spa().to(opts.device)  # 移动损失函数到设备
    criterion_exp = L_exp(patch_size=4, mean_val=0.5).to(opts.device)
    criterion_color = L_color().to(opts.device)
    criterion_tv = L_TV(TVLoss_weight=1).to(opts.device)
    optimizer = optim.Adam(model.parameters(), lr=opts.lr)

    model.train()
    for epoch in range(opts.epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{opts.epochs}")

        for low_imgs, normal_imgs in pbar:
            low_imgs = low_imgs.to(opts.device)
            normal_imgs = normal_imgs.to(opts.device)

            # 前向传播（含新增增强网络输出）
            R_low, L_low, R_restore, P_out, Q_out, enhance_image = model(low_imgs)
            R_normal, L_normal = model.decom(normal_imgs)

            # 原有损失
            loss = criterion_mse(R_restore, R_normal)
            loss += 0.1 * criterion_mse(P_out, R_normal)
            loss += 0.1 * criterion_mse(Q_out, L_normal)

            # 新增损失（基于enhance_net输出）
            loss += 0.3 * criterion_spa(normal_imgs, enhance_image)  # 空间一致性损失
            loss += 0.2 * criterion_exp(enhance_image)  # 曝光损失
            loss += 0.2 * criterion_color(enhance_image)  # 色彩损失
            loss += 0.1 * criterion_tv(enhance_image)  # 平滑损失

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.6f}")
        torch.save(model.state_dict(), os.path.join(opts.CHECKPOINT_DIR, f"model_epoch{epoch + 1}.pth"))
    print("训练完成！")


# -------------------------- 测试与可视化（含新增增强结果） --------------------------
def test_and_visualize(opts, num_vis=3):
    # 解决中文显示
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 加载模型
    model = LowLightEnhancer(opts).to(opts.device)
    checkpoint_path = os.path.join(opts.CHECKPOINT_DIR, f"model_epoch{opts.epochs}.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=opts.device, weights_only=True))
    model.eval()

    # 加载测试数据
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    test_dataset = LOLDataset(
        low_dir=opts.TEST_LOW_PATH,
        normal_dir=opts.TEST_NORMAL_PATH,
        transform=transform
    )

    # 定义归一化函数
    def normalize_to_01(x):
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    # 可视化（对比两种增强结果）
    plt.figure(figsize=(20, 5 * num_vis))
    with torch.no_grad():
        for i in range(num_vis):
            low_img, normal_img = test_dataset[i]
            low_img = low_img.unsqueeze(0).to(opts.device)
            R_low, L_low, R_restore, enhance_img1, enhance_img2 = model(low_img, is_train=False)

            # 转为numpy并处理
            low_np = low_img.squeeze().cpu().permute(1, 2, 0).numpy()
            R_low_np = normalize_to_01(R_low.squeeze().cpu().permute(1, 2, 0).numpy())
            L_low_np = normalize_to_01(L_low.squeeze().cpu().numpy())
            enhance1_np = np.clip(enhance_img1.squeeze().cpu().permute(1, 2, 0).numpy(), 0, 1)
            enhance2_np = np.clip(enhance_img2.squeeze().cpu().permute(1, 2, 0).numpy(), 0, 1)  # 新增增强结果
            normal_np = normal_img.permute(1, 2, 0).numpy()

            # 绘制子图（6列：输入+分解+两种增强+参考）
            plt.subplot(num_vis, 6, i * 6 + 1)
            plt.imshow(low_np)
            plt.title("低光输入")
            plt.axis("off")

            plt.subplot(num_vis, 6, i * 6 + 2)
            plt.imshow(R_low_np)
            plt.title("反射分量")
            plt.axis("off")

            plt.subplot(num_vis, 6, i * 6 + 3)
            plt.imshow(L_low_np, cmap="gray")
            plt.title("照明分量")
            plt.axis("off")

            plt.subplot(num_vis, 6, i * 6 + 4)
            plt.imshow(enhance1_np)
            plt.title("原有增强结果")
            plt.axis("off")

            plt.subplot(num_vis, 6, i * 6 + 5)
            plt.imshow(enhance2_np)
            plt.title("新增增强结果")
            plt.axis("off")

            plt.subplot(num_vis, 6, i * 6 + 6)
            plt.imshow(normal_np)
            plt.title("正常光参考")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig("low_light_enhancement_comparison.png", dpi=300)
    plt.show()


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    opts = Options()
    train_model(opts)  # 训练模型
    test_and_visualize(opts)  # 测试可视化

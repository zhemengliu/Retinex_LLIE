import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

# -------------------------- 数据路径配置 --------------------------
TRAIN_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"
TRAIN_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"
INIT_MODEL_PATH = "./ckpt/init_low.pth"  # 需先训练好的初始分解模型


# -------------------------- 1. 数据集定义（需低光+正常光配对数据） --------------------------
class UnfoldingDataset(Dataset):
    """用于训练迭代优化网络的数据集，需要低光图像和对应的正常光图像"""

    def __init__(self, low_dir, normal_dir, transform=None):
        self.low_paths = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir)
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.normal_paths = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                                    if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
        assert len(self.low_paths) == len(self.normal_paths), "低光和正常光图像数量不匹配"
        print(f"加载迭代优化网络训练数据：{len(self.low_paths)}对图像")

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_paths[idx]).convert('RGB')
        normal_img = Image.open(self.normal_paths[idx]).convert('RGB')

        if self.transform:
            low_img = self.transform(low_img)
            normal_img = self.transform(normal_img)

        return {
            'low': low_img,
            'normal': normal_img,
            'filename': os.path.basename(self.low_paths[idx])
        }


# -------------------------- 2. 核心网络定义（与之前保持一致） --------------------------
class SELayer(nn.Module):
    """通道注意力层"""

    def __init__(self, channel, reduction=16):
        super().__init__()
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


def get_conv2d_layer(in_c, out_c, k, s, p=0):
    return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)


class HalfDnCNNSE(nn.Module):
    """反射优化网络：输入初始反射P和光照Q，输出优化后的反射R"""

    def __init__(self, concat_L=True):
        super().__init__()
        self.concat_L = concat_L

        if self.concat_L:
            self.conv1 = get_conv2d_layer(3, 32, 3, 1, 1)  # 处理反射P
            self.conv2 = get_conv2d_layer(1, 32, 3, 1, 1)  # 处理光照Q
        else:
            self.conv1 = get_conv2d_layer(3, 64, 3, 1, 1)  # 仅处理反射P

        self.se_layer = SELayer(64)
        self.conv3 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.conv4 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.conv5 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.conv6 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.conv7 = get_conv2d_layer(64, 64, 3, 1, 1)
        self.conv8 = get_conv2d_layer(64, 3, 3, 1, 1)  # 输出反射残差

        self.relu = nn.ReLU(inplace=True)

    def forward(self, r, l):
        if self.concat_L:
            r_fs = self.relu(self.conv1(r))
            l_fs = self.relu(self.conv2(l))
            x = torch.cat([r_fs, l_fs], dim=1)  # 拼接反射和光照特征
        else:
            x = self.relu(self.conv1(r))

        x = self.se_layer(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        residual = self.conv8(x)  # 反射残差
        return r + residual  # 残差连接：R = P + 残差


class Illumination_Alone(nn.Module):
    """光照优化网络：输入初始光照Q，输出优化后的光照L"""

    def __init__(self):
        super().__init__()
        self.conv1 = get_conv2d_layer(1, 32, 5, 1, 2)
        self.conv2 = get_conv2d_layer(32, 32, 5, 1, 2)
        self.conv3 = get_conv2d_layer(32, 32, 5, 1, 2)
        self.conv4 = get_conv2d_layer(32, 32, 5, 1, 2)
        self.conv5 = get_conv2d_layer(32, 1, 1, 1, 0)  # 输出光照

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)  # 确保光照非负

    def forward(self, l):
        x = self.leaky_relu(self.conv1(l))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        return self.relu(self.conv5(x))  # 光照必须非负


class Decom(nn.Module):
    """复用初始分解网络（用于获取P和Q）"""

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

    def forward(self, x):
        output = self.decom(x)
        return output[:, 0:3, :, :], output[:, 3:4, :, :]  # P, Q


# -------------------------- 3. 迭代优化模块（用于训练） --------------------------
class UnfoldingTrainer(nn.Module):
    """迭代优化网络训练器，整合Decom、model_R、model_L"""

    def __init__(self, init_model_path, concat_L=True, round=5):
        super().__init__()
        # 加载初始分解网络（固定参数，不参与训练）
        self.decom_net = Decom()
        self.decom_net.load_state_dict(torch.load(init_model_path, map_location="cpu"))
        for param in self.decom_net.parameters():
            param.requires_grad = False  # 冻结初始分解网络

        # 待训练的优化网络
        self.model_R = HalfDnCNNSE(concat_L=concat_L)
        self.model_L = Illumination_Alone()

        # 迭代参数（训练中会优化）
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 反射迭代正则化参数
        self.lamda = nn.Parameter(torch.tensor(0.1))  # 光照迭代正则化参数
        self.Roffset = nn.Parameter(torch.tensor(0.01))  # gamma增量
        self.Loffset = nn.Parameter(torch.tensor(0.01))  # lamda增量
        self.round = round  # 迭代轮次（固定）

    def forward(self, low_img):
        # 1. 初始分解（固定）
        P, Q = self.decom_net(low_img)

        # 2. 多轮迭代优化
        for t in range(self.round):
            if t == 0:
                P_curr, Q_curr = P, Q
            else:
                # 迭代更新P和Q（基于上一轮的R和L）
                w_p = self.gamma + self.Roffset * t  # 动态调整gamma
                w_q = self.lamda + self.Loffset * t  # 动态调整lamda

                # 反射迭代公式（最小二乘）
                numerator_P = low_img * Q_curr + w_p * R_prev
                denominator_P = w_p + Q_curr * Q_curr
                P_curr = numerator_P / (denominator_P + 1e-8)

                # 光照迭代公式（最小二乘）
                IR, IG, IB = low_img[:, 0:1], low_img[:, 1:2], low_img[:, 2:3]
                PR, PG, PB = P_curr[:, 0:1], P_curr[:, 1:2], P_curr[:, 2:3]
                numerator_Q = IR * PR + IG * PG + IB * PB + w_q * L_prev
                denominator_Q = (PR ** 2 + PG ** 2 + PB ** 2) + w_q
                Q_curr = numerator_Q / (denominator_Q + 1e-8)

            # 优化反射和光照
            R_prev = self.model_R(P_curr, Q_curr)
            L_prev = self.model_L(Q_curr)

        return R_prev, L_prev


# -------------------------- 4. 损失函数定义 --------------------------
class UnfoldingLoss(nn.Module):
    """迭代优化网络的损失函数：
    1. 增强损失：优化后的R×L应接近正常光图像
    2. 反射一致性损失：R应接近正常光图像的反射（用正常光分解的P作为参考）
    3. 光照平滑损失：L应保持平滑
    """

    def __init__(self, decom_net, lambda_reflect=1.0, lambda_illum=0.1):
        super().__init__()
        self.decom_net = decom_net  # 用于获取正常光图像的反射（作为参考）
        self.lambda_reflect = lambda_reflect  # 反射一致性权重
        self.lambda_illum = lambda_illum  # 光照平滑权重
        self.mse = nn.MSELoss()

    def forward(self, low_img, normal_img, R, L):
        # 1. 增强损失：R×L 应接近正常光图像
        enhanced = R * L
        enhance_loss = self.mse(enhanced, normal_img)

        # 2. 反射一致性损失：R 应接近正常光图像的反射（用初始分解网络提取）
        normal_P, _ = self.decom_net(normal_img)  # 正常光的初始反射（作为参考）
        reflect_loss = self.mse(R, normal_P)

        # 3. 光照平滑损失：L 应平滑变化
        grad_x_L = torch.abs(L[:, :, 1:, :] - L[:, :, :-1, :])
        grad_y_L = torch.abs(L[:, :, :, 1:] - L[:, :, :, :-1])
        illum_loss = torch.mean(grad_x_L + grad_y_L)

        # 总损失
        total_loss = enhance_loss + self.lambda_reflect * reflect_loss + self.lambda_illum * illum_loss
        return total_loss, {
            "enhance": enhance_loss.item(),
            "reflect": reflect_loss.item(),
            "illum": illum_loss.item()
        }


# -------------------------- 5. 训练函数 --------------------------
def train_unfolding_network(args):
    # 设备设置
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # 数据集和加载器
    dataset = UnfoldingDataset(TRAIN_LOW_PATH, TRAIN_NORMAL_PATH, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 初始化模型（包含待训练的model_R、model_L和迭代参数）
    model = UnfoldingTrainer(
        init_model_path=INIT_MODEL_PATH,
        concat_L=args.concat_L,
        round=args.round
    ).to(device)

    # 损失函数（需要用到冻结的分解网络）
    criterion = UnfoldingLoss(
        model.decom_net,  # 传入冻结的分解网络用于计算参考反射
        lambda_reflect=args.lambda_reflect,
        lambda_illum=args.lambda_illum
    ).to(device)

    # 优化器（仅优化model_R、model_L和迭代参数）
    optimizer = optim.Adam([
        {'params': model.model_R.parameters()},
        {'params': model.model_L.parameters()},
        {'params': [model.gamma, model.lamda, model.Roffset, model.Loffset]}
    ], lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练过程
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loss_details = {"enhance": 0.0, "reflect": 0.0, "illum": 0.0}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            low_img = batch['low'].to(device)
            normal_img = batch['normal'].to(device)

            # 前向传播：获取优化后的R和L
            R, L = model(low_img)

            # 计算损失
            loss, details = criterion(low_img, normal_img, R, L)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            total_loss += loss.item() * low_img.size(0)
            for k in details:
                loss_details[k] += details[k] * low_img.size(0)

            pbar.set_postfix({"Loss": loss.item()})

        # 计算平均损失
        avg_loss = total_loss / len(dataset)
        for k in loss_details:
            loss_details[k] /= len(dataset)

        # 打印 epoch 结果
        print(f"Epoch {epoch} - 总损失: {avg_loss:.6f} | "
              f"增强损失: {loss_details['enhance']:.6f} | "
              f"反射一致性损失: {loss_details['reflect']:.6f} | "
              f"光照平滑损失: {loss_details['illum']:.6f}")

        # 保存最优模型（包含所有必要参数）
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dict = {
                # 迭代控制参数
                "gamma": model.gamma.item(),
                "lamda": model.lamda.item(),
                "Roffset": model.Roffset.item(),
                "Loffset": model.Loffset.item(),
                "round": args.round,
                "concat_L": args.concat_L,
                # 网络权重
                "model_R": model.model_R.state_dict(),
                "model_L": model.model_L.state_dict()
            }
            save_path = os.path.join(args.save_dir, "unfolding.pth")
            torch.save(save_dict, save_path)
            print(f"保存最优模型至 {save_path} (损失: {best_loss:.6f})")

        # 学习率衰减
        scheduler.step()

    print("迭代优化网络训练完成！")


# -------------------------- 6. 主函数 --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练迭代优化网络（生成unfolding.pth）")
    parser.add_argument("--img_size", type=int, default=256, help="输入图像尺寸")
    parser.add_argument("--batch_size", type=int, default=4, help="批量大小（建议小于分解网络）")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮次")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率（建议小于分解网络）")
    parser.add_argument("--round", type=int, default=5, help="迭代优化轮次")
    parser.add_argument("--concat_L", type=bool, default=True, help="是否拼接光照特征优化反射")
    parser.add_argument("--lambda_reflect", type=float, default=1.0, help="反射一致性损失权重")
    parser.add_argument("--lambda_illum", type=float, default=0.1, help="光照平滑损失权重")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU编号，-1表示CPU")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--save_dir", type=str, default="./ckpt", help="模型保存目录")

    args = parser.parse_args()
    train_unfolding_network(args)

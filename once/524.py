import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# 数据集路径设置
TEST_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high"
TRAIN_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low"
TRAIN_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high"
TEST_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"
CHECKPOINT_DIR = "./checkpoints"

# 创建检查点目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --------------------------
# 1. 自定义数据集类
# --------------------------
class LOLDataset(Dataset):
    def __init__(self, low_light_dir, normal_light_dir, transform=None):
        self.low_light_dir = low_light_dir
        self.normal_light_dir = normal_light_dir
        self.transform = transform

        # 获取所有图像文件名（假设低光和正常光图像文件名相同）
        self.image_names = [f for f in os.listdir(low_light_dir)
                            if os.path.isfile(os.path.join(low_light_dir, f))]

        # 确保低光和正常光图像数量匹配
        normal_names = set(os.listdir(normal_light_dir))
        self.image_names = [f for f in self.image_names if f in normal_names]
        print(f"加载数据集: {len(self.image_names)} 张图像")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # 加载低光和正常光图像
        low_light_path = os.path.join(self.low_light_dir, img_name)
        normal_light_path = os.path.join(self.normal_light_dir, img_name)

        low_light_img = Image.open(low_light_path).convert('RGB')
        normal_light_img = Image.open(normal_light_path).convert('RGB')

        # 应用变换
        if self.transform:
            low_light_img = self.transform(low_light_img)
            normal_light_img = self.transform(normal_light_img)

        return low_light_img, normal_light_img


# --------------------------
# 2. Retinex分解网络结构（编码器-解码器）
# --------------------------
class RetinexNet(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(RetinexNet, self).__init__()
        # 编码器（下采样）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features),  # 实例归一化
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1),  # 下采样
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, stride=2, padding=1),  # 下采样
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 解码器（上采样+跳跃连接）
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, kernel_size=4, stride=2, padding=1),  # 上采样
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),  # 上采样
            nn.Sigmoid()  # 输出范围[0,1]（照明层/反射层特性）
        )

    def forward(self, x):
        # 编码器输出（跳跃连接特征）
        e1 = self.encoder1(x)  # 跳跃连接特征1
        e2 = self.encoder2(e1)  # 最深层特征

        # 解码器（结合跳跃连接）
        d1 = self.decoder1(e2)
        d1 = torch.cat([d1, e1[:, :d1.size(1), :, :]], dim=1)  # 跳跃连接（拼接编码器特征）
        out = self.decoder2(d1)
        return out


# --------------------------
# 3. 正则项网络（用于估计反射层/照明层正则项）
# --------------------------
class RegularizationNet(nn.Module):
    """用于处理R(I)和R(R)的去噪正则项网络（简化实现）"""

    def __init__(self, in_channels=3):
        super(RegularizationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)  # 输出正则化后的结果


# --------------------------
# 4. 半二次分裂求解器
# --------------------------
class RetinexSolver:
    def __init__(self, lambda1=0.1, lambda2=0.1, mu1=0.5, mu2=0.5,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 初始化网络
        self.I_net = RetinexNet()  # 照明层估计网络
        self.R_net = RetinexNet()  # 反射层估计网络
        self.reg_I = RegularizationNet()  # 照明层正则网络
        self.reg_R = RegularizationNet()  # 反射层正则网络

        # 设备设置
        self.device = device
        self.I_net.to(self.device)
        self.R_net.to(self.device)
        self.reg_I.to(self.device)
        self.reg_R.to(self.device)

        # 超参数（可调正参数）
        self.lambda1 = lambda1  # 照明层正则系数
        self.lambda2 = lambda2  # 反射层正则系数
        self.mu1 = mu1  # 辅助变量约束系数
        self.mu2 = mu2  # 辅助变量约束系数

        # 优化器
        self.optimizer = optim.Adam(
            list(self.I_net.parameters()) + list(self.R_net.parameters()) +
            list(self.reg_I.parameters()) + list(self.reg_R.parameters()),
            lr=1e-4
        )

    def close_form_I(self, O, R, I_prev, z1_prev):
        """照明层I的闭式解（对应公式）"""
        numerator = O * R + self.mu1 * z1_prev
        denominator = R * R + self.mu1
        return numerator / (denominator + 1e-8)  # 避免除零

    def close_form_R(self, O, I, R_prev, z2_prev):
        """反射层R的闭式解（对应公式）"""
        numerator = O * I + self.mu2 * z2_prev
        denominator = I * I + self.mu2
        return numerator / (denominator + 1e-8)  # 避免除零

    def train_step(self, O):
        """单步训练（交替求解变量）"""
        self.optimizer.zero_grad()

        # 初始化变量（或上一步结果）
        I_est = self.I_net(O)  # 初始照明层估计
        R_est = self.R_net(O)  # 初始反射层估计
        z1 = I_est.detach().clone()  # 辅助变量z1（对应I的约束）
        z2 = R_est.detach().clone()  # 辅助变量z2（对应R的约束）

        # 交替更新：步骤1-更新I（闭式解）
        I_new = self.close_form_I(O, R_est, I_est, z1)

        # 交替更新：步骤2-更新R（闭式解）
        R_new = self.close_form_R(O, I_new, R_est, z2)

        # 交替更新：步骤3-更新辅助变量z1（正则项约束）
        z1_new = self.reg_I(I_new)

        # 交替更新：步骤4-更新辅助变量z2（正则项约束）
        z2_new = self.reg_R(R_new)

        # 计算损失（数据项+正则项+约束项）
        data_loss = F.mse_loss(O, I_new * R_new)  # 数据项||O - I⊗R||²
        reg_loss = self.lambda1 * F.mse_loss(self.reg_I(I_new), I_new) + \
                   self.lambda2 * F.mse_loss(self.reg_R(R_new), R_new)  # 正则项
        constraint_loss = self.mu1 * F.mse_loss(I_new, z1_new) + \
                          self.mu2 * F.mse_loss(R_new, z2_new)  # 约束项

        total_loss = data_loss + reg_loss + constraint_loss

        # 反向传播与优化
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'reg_loss': reg_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'I': I_new,
            'R': R_new,
            'reconstructed': I_new * R_new
        }

    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"retinex_epoch_{epoch}.pth")
        torch.save({
            'I_net_state_dict': self.I_net.state_dict(),
            'R_net_state_dict': self.R_net.state_dict(),
            'reg_I_state_dict': self.reg_I.state_dict(),
            'reg_R_state_dict': self.reg_R.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"检查点已保存至: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.I_net.load_state_dict(checkpoint['I_net_state_dict'])
        self.R_net.load_state_dict(checkpoint['R_net_state_dict'])
        self.reg_I.load_state_dict(checkpoint['reg_I_state_dict'])
        self.reg_R.load_state_dict(checkpoint['reg_R_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"已加载检查点: {checkpoint_path}")


# --------------------------
# 5. 训练和测试函数
# --------------------------
def train_model(num_epochs=1, batch_size=8, lr=1e-4):
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 创建数据集和数据加载器
    train_dataset = LOLDataset(
        low_light_dir=TRAIN_LOW_PATH,
        normal_light_dir=TRAIN_NORMAL_PATH,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # 初始化求解器
    solver = RetinexSolver()
    solver.optimizer.param_groups[0]['lr'] = lr

    # 训练循环
    for epoch in range(num_epochs):
        solver.I_net.train()
        solver.R_net.train()
        solver.reg_I.train()
        solver.reg_R.train()

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for low_light, normal_light in pbar:
            # 只使用低光图像作为输入（Retinex分解从单张低光图像恢复）
            O = low_light.to(solver.device)

            # 训练步骤
            result = solver.train_step(O)

            # 累计损失
            total_loss += result['total_loss']
            pbar.set_postfix({"Loss": f"{result['total_loss']:.6f}"})

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, 平均损失: {avg_loss:.6f}")

        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            solver.save_checkpoint(epoch + 1)

    return solver


def test_model(solver=None, checkpoint_path=None):
    # 如果没有提供求解器，则创建新的并加载检查点
    if solver is None:
        solver = RetinexSolver()

    # 如果提供了检查点路径，则加载
    if checkpoint_path:
        solver.load_checkpoint(checkpoint_path)

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 创建测试数据集和数据加载器
    test_dataset = LOLDataset(
        low_light_dir=TEST_LOW_PATH,
        normal_light_dir=TEST_NORMAL_PATH,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # 切换到评估模式
    solver.I_net.eval()
    solver.R_net.eval()
    solver.reg_I.eval()
    solver.reg_R.eval()

    # 测试循环
    results = []
    with torch.no_grad():
        for low_light, normal_light in tqdm(test_loader, desc="测试中"):
            O = low_light.to(solver.device)

            # 估计照明层和反射层
            I_est = solver.I_net(O)
            R_est = solver.R_net(O)
            reconstructed = I_est * R_est

            # 保存结果
            results.append({
                'low_light': low_light.cpu().numpy(),
                'normal_light': normal_light.cpu().numpy(),
                'illumination': I_est.cpu().numpy(),
                'reflectance': R_est.cpu().numpy(),
                'reconstructed': reconstructed.cpu().numpy()
            })

    print(f"测试完成，处理了 {len(results)} 张图像")
    return results


# --------------------------
# 6. 主函数
# --------------------------
if __name__ == "__main__":
    # 训练模型
    print("开始训练模型...")
    solver = train_model(num_epochs=50, batch_size=8)

    # 测试模型
    print("开始测试模型...")
    test_results = test_model(solver)

    print("所有任务完成!")

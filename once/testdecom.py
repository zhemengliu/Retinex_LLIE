import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------
# 路径设置（保持与训练时一致）
# --------------------------
TEST_NORMAL_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high"
TEST_LOW_PATH = "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low"
CHECKPOINT_DIR = "./checkpoints"  # 已保存50epoch模型的目录
VISUALIZATION_DIR = "./visualizations/test"  # 测试结果保存目录

# 创建可视化目录
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


# --------------------------
# 1. 自定义数据集类（仅用于测试）
# --------------------------
class LOLDataset(Dataset):
    def __init__(self, low_light_dir, normal_light_dir, transform=None):
        self.low_light_dir = low_light_dir
        self.normal_light_dir = normal_light_dir
        self.transform = transform

        # 获取所有图像文件名（匹配低光和正常光图像）
        self.image_names = [f for f in os.listdir(low_light_dir)
                            if os.path.isfile(os.path.join(low_light_dir, f))]
        normal_names = set(os.listdir(normal_light_dir))
        self.image_names = [f for f in self.image_names if f in normal_names]
        print(f"加载测试数据集: {len(self.image_names)} 张图像")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        # 加载图像
        low_light_path = os.path.join(self.low_light_dir, img_name)
        normal_light_path = os.path.join(self.normal_light_dir, img_name)
        low_light_img = Image.open(low_light_path).convert('RGB')
        normal_light_img = Image.open(normal_light_path).convert('RGB')
        # 应用变换
        if self.transform:
            low_light_img = self.transform(low_light_img)
            normal_light_img = self.transform(normal_light_img)
        return low_light_img, normal_light_img, img_name


# --------------------------
# 2. 可视化函数（测试专用）
# --------------------------
def visualize_test_result(low_light, illumination, reflectance, reconstructed,
                          normal_light, img_name):
    """将测试结果可视化并保存"""

    # 转换Tensor为可显示的numpy图像
    def tensor2img(tensor):
        img = tensor.detach().cpu().numpy()
        if img.ndim == 4:
            img = img[0]  # 去除batch维度
        return np.transpose(np.clip(img, 0, 1), (1, 2, 0))  # 调整维度并裁剪值范围

    # 转换所有图像
    low_light_img = tensor2img(low_light)
    illumination_img = tensor2img(illumination)
    reflectance_img = tensor2img(reflectance)
    reconstructed_img = tensor2img(reconstructed)
    normal_light_img = tensor2img(normal_light)

    # 创建画布并显示
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(low_light_img)
    axes[0].set_title("Low Light Input")
    axes[0].axis("off")

    axes[1].imshow(illumination_img)
    axes[1].set_title("Illumination Layer")
    axes[1].axis("off")

    axes[2].imshow(reflectance_img)
    axes[2].set_title("Reflectance Layer")
    axes[2].axis("off")

    axes[3].imshow(reconstructed_img)
    axes[3].set_title("Reconstructed Image")
    axes[3].axis("off")

    axes[4].imshow(normal_light_img)
    axes[4].set_title("Normal Light (GT)")
    axes[4].axis("off")

    # 保存图像
    save_path = os.path.join(VISUALIZATION_DIR, f"{img_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


# --------------------------
# 3. 网络结构（与训练时一致，必须保留）
# --------------------------
class RetinexNet(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(RetinexNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        d1 = self.decoder1(e2)
        d1 = torch.cat([d1, e1[:, :d1.size(1), :, :]], dim=1)
        return self.decoder2(d1)


class RegularizationNet(nn.Module):
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
        return self.net(x)


# --------------------------
# 4. 测试执行函数（核心）
# --------------------------
def test_with_pretrained(checkpoint_path):
    """加载预训练模型并执行测试"""
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 初始化网络
    I_net = RetinexNet().to(device)
    R_net = RetinexNet().to(device)
    reg_I = RegularizationNet().to(device)
    reg_R = RegularizationNet().to(device)

    # 加载50epoch的预训练权重
    print(f"加载预训练模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    I_net.load_state_dict(checkpoint['I_net_state_dict'])
    R_net.load_state_dict(checkpoint['R_net_state_dict'])
    reg_I.load_state_dict(checkpoint['reg_I_state_dict'])
    reg_R.load_state_dict(checkpoint['reg_R_state_dict'])

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
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

    # 开始测试
    I_net.eval()
    R_net.eval()
    reg_I.eval()
    reg_R.eval()
    results = []
    with torch.no_grad():  # 关闭梯度计算，加速测试
        for low_light, normal_light, img_name in tqdm(test_loader, desc="测试进度"):
            # 数据移至设备
            low_light = low_light.to(device)
            # 执行Retinex分解
            illumination = I_net(low_light)  # 照明层
            reflectance = R_net(low_light)  # 反射层
            reconstructed = illumination * reflectance  # 重建图像
            # 可视化并保存结果
            save_path = visualize_test_result(
                low_light=low_light,
                illumination=illumination,
                reflectance=reflectance,
                reconstructed=reconstructed,
                normal_light=normal_light,
                img_name=img_name[0]
            )
            # 保存结果数据
            results.append({
                "image_name": img_name[0],
                "visualization_path": save_path
            })
    print(f"测试完成！所有结果已保存至：{VISUALIZATION_DIR}")
    return results


# --------------------------
# 5. 主函数（直接运行测试）
# --------------------------
if __name__ == "__main__":
    # 50epoch模型的检查点路径（根据实际保存的文件名调整）
    checkpoint_50epoch = os.path.join(CHECKPOINT_DIR, "retinex_epoch_50.pth")
    # 执行测试
    test_results = test_with_pretrained(checkpoint_50epoch)
    print("所有测试任务完成！")
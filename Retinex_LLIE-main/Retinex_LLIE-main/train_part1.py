import os
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 假设你的模型定义在 Model.py 文件中
from model_Decom_fft import LLIE
from loss_DecomNet import DecomLoss


# =============================================================================
# --- 1. 数据集定义 (直接整合到脚本中) ---
# =============================================================================
class LowLightDataset(Dataset):
    """从文件夹读取低光图像的数据集"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


# =============================================================================
# --- 2. 损失函数定义 (直接整合到脚本中) ---
# =============================================================================

# =============================================================================
# --- 3. 辅助函数和主训练逻辑 ---
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = LowLightDataset(root_dir=args.data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Number of training data: {len(train_dataset)}")

    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)
        print(f"Created directory: {args.weights_dir}")

    # <-- 关键修改在这里：将 args 传递给模型
    model = LLIE(args).to(device)

    optimizer = optim.Adam(model.decom_net.parameters(), lr=args.lr)

    criterion = DecomLoss(device)

    for param in model.enhance_net.parameters():
        param.requires_grad = False
    print("EnhanceNet parameters frozen.")

    start_epoch = 1
    if args.resume_epoch > 0:
        resume_path = os.path.join(args.weights_dir, f'decom_epoch_{args.resume_epoch}.pth')
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            model.decom_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed training from epoch {start_epoch}.")
        else:
            print(f"Warning: Checkpoint file not found at {resume_path}. Starting from scratch.")

    print("Start Training DecomNet...")
    for epoch in range(start_epoch, args.epochs + 1):
        model.decom_net.train()

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]", leave=True)

        for i, low_img in enumerate(progress_bar):
            low_img = low_img.to(device)
            optimizer.zero_grad()

            low_R, low_L, gamma_R, gamma_L, x_gamma, _, _ = model(low_img)

            loss_dict = criterion(low_img, x_gamma, low_R, low_L, gamma_R, gamma_L)
            total_loss = loss_dict['total']

            if torch.isnan(total_loss):
                print(f"\n!!! NaN loss detected at epoch {epoch}, step {i}. Stopping training. !!!")
                return

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decom_net.parameters(), max_norm=1.0)
            optimizer.step()

            progress_bar.set_postfix(
                Loss=f"{total_loss.item():.4f}",
                Recon=f"{loss_dict['recon'].item():.4f}",
                Consist_R=f"{loss_dict['consistency_R'].item():.4f}",
                Smooth_L=f"{loss_dict['smooth_L'].item():.4f}"
            )

        print(f"--- Epoch [{epoch}/{args.epochs}] Finished, Last Batch Loss: {total_loss.item():.4f} ---")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.decom_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.weights_dir, f'decom_epoch_{epoch}.pth'))


# =============================================================================
# --- 4. 命令行参数解析和程序入口 ---
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DecomNet Training Script')

    parser.add_argument('--data_path', type=str, default='D:/low_light_image/Dataset/Dataset/LOLdataset/our485/low',
                        help='Path to the training data')
    parser.add_argument('--epochs', type=int, default=35, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for optimizer (already lowered)')
    parser.add_argument('--weights_dir', type=str, default='weights_decom', help='Directory to save model weights')
    parser.add_argument('--resume_epoch', type=int, default=25,
                        help='Epoch to resume')  # 改回0，从头开始
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    main(args)

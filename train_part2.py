import os
import argparse
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from PIL import Image
import matplotlib
from skimage import color # <--- 新增导入
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from torchvision.utils import save_image
from Model import LLIE


# ===================================================================
# 0. 数据集定义 (Dataset Definitions)
# ===================================================================

class MyDataset(Dataset):
    def __init__(self, folder, patch_size, num_patches_per_img):
        super(MyDataset, self).__init__()
        self.folder = folder
        self.patch_size = patch_size
        self.num_patches_per_img = num_patches_per_img
        self.image_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.length = len(self.image_list) * self.num_patches_per_img

    def __getitem__(self, index):
        img_path = random.choice(self.image_list)
        image = Image.open(img_path).convert('RGB')
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.patch_size, self.patch_size))
        patch = image.crop((j, i, j + w, i + h))
        tensor_patch = self.transform(patch)
        return tensor_patch, os.path.basename(img_path)

    def __len__(self):
        return self.length


class TestDataset(Dataset):
    def __init__(self, low_light_dir):
        self.low_light_dir = low_light_dir
        self.image_files = sorted([os.path.join(low_light_dir, f) for f in os.listdir(low_light_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.image_files[idx]


# ===================================================================
# 1. 定义自监督损失函数 (Self-Supervised Loss Functions)
# ===================================================================

class SpatialConsistencyLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(SpatialConsistencyLoss, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0).to(device)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0).to(device)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0).to(device)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0).to(device)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
        D_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)
        D_left = torch.abs(D_org_left - D_enhance_left)
        D_right = torch.abs(D_org_right - D_enhance_right)
        D_up = torch.abs(D_org_up - D_enhance_up)
        D_down = torch.abs(D_org_down - D_enhance_down)
        return torch.mean(D_left + D_right + D_up + D_down)


class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, enhance_img):
        x = torch.mean(enhance_img, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.pow(mean - self.mean_val, 2)
        return d.mean()


def total_variation_loss(image):
    B, C, H, W = image.shape
    h_tv = torch.pow((image[:, :, 1:, :] - image[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((image[:, :, :, 1:] - image[:, :, :, :-1]), 2).sum()
    return (h_tv + w_tv) / (B * C * H * W)

# --- 新增损失函数 ---
class IlluminationStructureLoss(nn.Module):
    def __init__(self):
        super(IlluminationStructureLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, enhance_L, low_L):
        return self.l1_loss(enhance_L, low_L)


# ===================================================================
# 2. 训练与测试/可视化函数 (Train and Test/Visualize Functions)
# ===================================================================

def train(args, model, train_loader, optimizer, scheduler, device, resume_epoch=0):
    print("\n--- Stage 2 Training (Self-Supervised): Training EnhanceNet Only ---")
    model.decom_net.eval()
    for param in model.decom_net.parameters():
        param.requires_grad = False
    model.enhance_net.train()
    print("DecomNet parameters have been frozen.")

    loss_spa = SpatialConsistencyLoss(device=device).to(device)
    loss_exp = ExposureControlLoss(patch_size=args.patch_size, mean_val=args.mean_val).to(device)
    loss_is = IlluminationStructureLoss().to(device) # <--- 新增

    # --- 调整后的损失权重 ---
    W_spa = 1.0
    W_exp = 10.0
    W_tv = 0.1
    W_is = 2.0  # <--- 新增权重

    loss_log_path = os.path.join(args.result_dir, "train_loss_log.txt")

    for epoch in range(resume_epoch, args.epochs):
        epoch_start = time.time()
        epoch_total_loss, epoch_loss_tv, epoch_loss_spa, epoch_loss_exp, epoch_loss_is = 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, (x_low, _) in enumerate(train_loader):
            x_low = x_low.to(device)

            with torch.no_grad():
                low_R, low_L, _, _, _, _, _ = model(x_low, decom_only=True)

            enhance_L, enhance_img = model(x_low=None, R_low=low_R, L_low=low_L, enhance_only=True)

            loss_tv_L = total_variation_loss(enhance_L)
            loss_spatial = loss_spa(x_low, enhance_img)
            loss_exposure = loss_exp(enhance_img)
            loss_illum_struct = loss_is(enhance_L, low_L) # <--- 计算新损失

            total_loss = W_tv * loss_tv_L + W_spa * loss_spatial + W_exp * loss_exposure + W_is * loss_illum_struct

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_loss_tv += loss_tv_L.item()
            epoch_loss_spa += loss_spatial.item()
            epoch_loss_exp += loss_exposure.item()
            epoch_loss_is += loss_illum_struct.item()

        avg_total = epoch_total_loss / len(train_loader)
        avg_tv = epoch_loss_tv / len(train_loader)
        avg_spa = epoch_loss_spa / len(train_loader)
        avg_exp = epoch_loss_exp / len(train_loader)
        avg_is = epoch_loss_is / len(train_loader)

        scheduler.step(avg_total)

        if (epoch + 1) % args.save_interval == 0:
            save_dict = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                         "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch + 1}
            save_path = os.path.join(args.ckpt_dir, f"enhance_epoch_{epoch + 1}.pth")
            torch.save(save_dict, save_path)
            print(f"Model saved to: {save_path}")

        epoch_time = time.time() - epoch_start
        log_str = (f"Epoch [{epoch + 1}/{args.epochs}] | Time: {epoch_time:.2f}s | "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f} | Total Loss: {avg_total:.4f} | "
                   f"TV: {avg_tv:.4f} | Spa: {avg_spa:.4f} | Exp: {avg_exp:.4f} | IS: {avg_is:.4f}")
        print(log_str)

        with open(loss_log_path, "a") as f:
            f.write(log_str + "\n")

        if (epoch + 1) % args.vis_interval == 0:
            test_and_visualize(args, model, TestDataset(args.test_dir), device, epoch + 1)


def test_and_visualize(args, model, test_dataset, device, epoch):
    print("\n" + "=" * 50 + f"\nTesting and Visualizing at Epoch {epoch}...\n" + "=" * 50)
    model.eval()
    epoch_result_dir = os.path.join(args.vis_dir, f"epoch_{epoch}")
    os.makedirs(epoch_result_dir, exist_ok=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    transform = transforms.ToTensor()
    with torch.no_grad():
        for i, img_path in enumerate(test_loader):
            img_path = img_path[0]
            img_name = os.path.basename(img_path)
            print(f" Processing {i + 1}/{len(test_loader)}: {img_name}")
            try:
                low_light_img = Image.open(img_path).convert('RGB')
                low_light_tensor = transform(low_light_img).unsqueeze(0).to(device)
                low_R, low_L, _, _, _, enhance_L, enhance_img = model(low_light_tensor)
                # --- 全新的、更稳定的色彩恢复步骤 (使用HSV空间) ---
                original_np_color = low_light_tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
                enhanced_np_gray = enhance_img.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
                original_hsv = color.rgb2hsv(original_np_color)

                H = original_hsv[:, :, 0]
                S = original_hsv[:, :, 1]

                # =================================================================
                #  vvv  这是我们修复的地方  vvv
                # =================================================================
                # 旧代码: V = enhanced_np_gray.squeeze(2)
                # 新代码: 计算RGB三个通道的平均值来得到单通道的明度V
                V = np.mean(enhanced_np_gray, axis=2)
                # =================================================================
                #  ^^^      修复结束      ^^^
                # =================================================================
                new_hsv = np.stack([H, S, V], axis=2)

                enhanced_with_color = color.hsv2rgb(new_hsv)

                def to_numpy(tensor):
                    return tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()

                original_np = np.clip(to_numpy(low_light_tensor), 0, 1)
                low_R_np = np.clip(to_numpy(low_R), 0, 1)
                low_L_np = np.clip(to_numpy(low_L), 0, 1)
                enhance_L_np = np.clip(to_numpy(enhance_L), 0, 1)
                enhanced_np = np.clip(enhanced_with_color, 0, 1)
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                titles = ['Original', 'Reflectance', 'Illumination', 'Enhanced Illum.', 'Final Result']
                images = [original_np, low_R_np, low_L_np, enhance_L_np, enhanced_np]
                for ax, title, img in zip(axes, titles, images):
                    # 这里对单通道图像的处理也稍微优化一下，更安全
                    if img.ndim == 3 and img.shape[2] == 1:
                        ax.imshow(img[:, :, 0], cmap='gray')
                    else:
                        ax.imshow(img)
                    ax.set_title(title)
                    ax.axis('off')
                plt.tight_layout()
                save_path = os.path.join(epoch_result_dir, f"compare_{img_name}")
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                gc.collect()
                final_img_save_path = os.path.join(epoch_result_dir, img_name)
                final_tensor_to_save = torch.from_numpy(enhanced_np).permute(2, 0, 1)
                save_image(final_tensor_to_save, final_img_save_path)
            except Exception as e:
                print(f"Visualization failed for {img_name}: {e}")
    print("Visualization complete.")
    model.train()


# ===================================================================
# 3. 主函数 (Main Function)
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Supervised Low-Light Enhancement - Stage 2: EnhanceNet")
    parser.add_argument("--train_dir", type=str, default="D:/low_light_image/Dataset/Dataset/LOLdataset/our485/low/", help="Path to training low-light images.")
    parser.add_argument("--test_dir", type=str, default="D:/low_light_image/Dataset/Dataset/LOLdataset/eval15/low/", help="Path to testing low-light images.")
    parser.add_argument("--decom_weights", type=str, default="D:/low_light_image/weights_decom/decom_epoch_33.pth", help="Path to pre-trained DecomNet weights.")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint to resume EnhanceNet training.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--num_patches", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--vis_interval", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--mean_val", type=float, default=0.6)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    args.result_dir = f"results_enhance_self_supervised_{timestamp}"
    args.ckpt_dir = os.path.join(args.result_dir, "ckpt")
    args.vis_dir = os.path.join(args.result_dir, "visualization")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    print(f"Results will be saved to: {args.result_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    print(f"Random seed fixed: {args.seed}")

    train_dataset = MyDataset(folder=args.train_dir, patch_size=args.crop_size, num_patches_per_img=args.num_patches)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataset = TestDataset(low_light_dir=args.test_dir)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    model = LLIE(args).to(device)

    if not os.path.exists(args.decom_weights):
        raise FileNotFoundError(f"DecomNet weights not found at: {args.decom_weights}")
    decom_checkpoint = torch.load(args.decom_weights, map_location=device)
    model.decom_net.load_state_dict(decom_checkpoint['model_state_dict'])
    print(f"Successfully loaded pre-trained DecomNet from {args.decom_weights}")

    optimizer = optim.Adam(model.enhance_net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    resume_epoch = 0
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        resume_epoch = checkpoint.get("epoch", 0)
        print(f"Resuming training from epoch {resume_epoch + 1}")
    elif args.resume_ckpt:
        print(f"Warning: Resume checkpoint not found at {args.resume_ckpt}. Starting from scratch.")

    train(args, model, train_loader, optimizer, scheduler, device, resume_epoch=resume_epoch)

    print("\n--- Final Testing and Visualization ---")
    test_and_visualize(args, model, test_dataset, device, epoch=args.epochs)

    print("\nAll Done!")


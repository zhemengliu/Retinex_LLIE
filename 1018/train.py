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
import datetime
import matplotlib
import torch
import torchvision.utils as vutils
import matplotlib
from torch.optim.lr_scheduler import SequentialLR, LinearLR

import matplotlib.pyplot as plt
torch.cuda.empty_cache()


from pytorch_msssim import ssim as ssim_torch
from Model import *  # 确保Model.py中LLIE类定义正确
from MyDataset import *  # 确保MyDataset、LOLDataset定义正确
from Loss import *  # 确保compute_total_loss函数定义正确


def calculate_metrics(x_enhanced, x_normal, lpips_model):
    """
    计算SSIM、PSNR、LPIPS。
    此版本接收一个已初始化的lpips_model，并内部处理设备一致性，保证高效和稳定。
    """
    # 1. 从输入张量自动获取设备，这是最可靠的方式
    device = x_enhanced.device

    # 2. 确保LPIPS模型在正确的设备上。
    # 这是一个轻量级操作，如果模型已在device上，则几乎不耗时。
    lpips_model.to(device)

    b = x_enhanced.size(0)
    ssim_sum, psnr_sum, lpips_sum = 0.0, 0.0, 0.0

    with torch.no_grad():
        for i in range(b):
            # 为了清晰，我们先准备好当前循环的图像切片
            enh_img_slice = x_enhanced[i].unsqueeze(0)
            norm_img_slice = x_normal[i].unsqueeze(0)

            # PyTorch SSIM (数据已在device上)
            ssim_val = ssim_torch(enh_img_slice, norm_img_slice, data_range=1.0, size_average=True)
            ssim_sum += ssim_val.item()

            # PSNR (需要移到CPU和NumPy)
            enh_np = (enh_img_slice.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            norm_np = (norm_img_slice.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            psnr_val = psnr(enh_np, norm_np, data_range=255)
            psnr_sum += psnr_val

            # LPIPS (范围标准化到[-1,1])
            enh_lpips = (enh_img_slice * 2) - 1
            norm_lpips = (norm_img_slice * 2) - 1

            # 关键：调用前，我们已经确保了lpips_model和输入张量都在同一个device上
            lpips_val = lpips_model(enh_lpips, norm_lpips).item()
            lpips_sum += lpips_val

    return (ssim_sum / b, psnr_sum / b, lpips_sum / b)


import torch
import os
import gc
import torch.nn.functional as F

# --- 这是新的可视化函数 ---
def visualize_modules(low_R, low_L, enhance_img, high_R, high_L, S, A, raw_image, filename):
    """
    一个简化的版本，只可视化核心的、确定为图像的张量，以避免维度错误。
    """
    try:
        def process_tensor(tensor):
            t = tensor.detach().cpu()[0:1, :, :, :] # 取第一张图
            if t.shape[1] == 1: # 如果是单通道
                t = t.repeat(1, 3, 1, 1) # 复制成三通道
            return torch.clamp(t, 0.0, 1.0)
        # 只选择我们确定是图像的张量进行可视化
        t_raw = process_tensor(raw_image)
        t_enhanced = process_tensor(enhance_img)
        t_low_R = process_tensor(low_R)
        t_low_L = process_tensor(low_L)
        # 将这些安全的张量组合成一个网格
        grid = vutils.make_grid(
            [t_raw, t_enhanced, t_low_R, t_low_L],
            nrow=4, # 每行4张图
            padding=2,
            normalize=False
        )
        vutils.save_image(grid, filename)
    except Exception as e:
        print(f"[CRITICAL ERROR] During visualization: {e}")
        import traceback
        traceback.print_exc()
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False


def train(args, model, train_loader, test_loader, optimizer, scheduler, device, resume_epoch=0):
    """端到端训练（支持断点续训）"""
    # 加载历史最佳损失（若续训，从之前的最佳损失开始）
    best_loss = float("inf")
    # 检查是否有历史损失记录（可选：可从日志文件读取）
    loss_log_path = os.path.join(args.vis_dir, "train_loss_log.txt")
    if os.path.exists(loss_log_path) and resume_epoch > 0:
        with open(loss_log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Best Loss" in line:
                    best_loss = float(line.strip().split(":")[-1])
        print(f"加载历史最佳损失: {best_loss:.6f}")



    # 调整epoch循环范围：从resume_epoch开始到args.epochs
    for epoch in range(resume_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss, decom_loss, enhance_loss = 0.0, 0.0, 0.0
        for batch_idx, (name, x_low, x_normal) in enumerate(train_loader):
            # 数据移至设备
            x_low = x_low.to(device)
            x_normal = x_normal.to(device)

            # 前向传播
            # 测试代码======================1111======================================

            low_R, low_L, R_gamma, L_gamma, x_gamma, enhance_L, enhance_img = model(x_low)
           # print(low_R)
            #============================================================================
            # 计算总损失
            loss, d_loss, e_loss = compute_total_loss(
                x_low, x_gamma, low_R, low_L, R_gamma, L_gamma, enhance_L, enhance_img
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # 累加损失
            batch_size = x_low.size(0)
            total_loss += loss.item() * batch_size
            decom_loss += d_loss.item() * batch_size
            enhance_loss += e_loss.item() * batch_size

        torch.cuda.empty_cache()

        # 平均损失
        avg_total = total_loss / len(train_loader.dataset)
        avg_d = decom_loss / len(train_loader.dataset)
        avg_e = enhance_loss / len(train_loader.dataset)

        # 学习率调度（基于当前epoch的损失）
        scheduler.step()

        # 保存最佳模型（含调度器状态，用于续训）
        if avg_total < best_loss:
            best_loss = avg_total
            os.makedirs(args.ckpt_dir, exist_ok=True)
            # 保存模型权重 + 调度器状态（关键：续训时恢复学习率）
            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器状态
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "last_epoch": epoch  # 记录最后训练的epoch
            }
            save_path = os.path.join(args.ckpt_dir, "best_model.pth")
            torch.save(save_dict, save_path)
            print(f"[续训] 最佳模型已保存至: {save_path} (当前epoch: {epoch + 1}, 最佳损失: {best_loss:.6f})")

        # 每隔10个epoch定期保存模型
        if (epoch + 1) % 10 == 0:  # epoch+1是1-based，确保第10、20、30...epoch保存
            os.makedirs(args.ckpt_dir, exist_ok=True)
            # 定期保存字典：包含模型、优化器、调度器状态（支持从该模型续训）
            periodic_save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器状态
                "scheduler_state_dict": scheduler.state_dict(),  # 保存学习率调度状态
                "current_epoch": epoch + 1,  # 记录当前完成的epoch（1-based）
                "current_total_loss": avg_total,  # 记录当前epoch损失
                "best_loss_so_far": best_loss  # 记录截至当前的最佳损失
            }
            # 文件名包含epoch号（避免覆盖），例如"model_epoch_10.pth"
            periodic_save_path = os.path.join(args.ckpt_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(periodic_save_dict, periodic_save_path)
            print(f"[定期保存] 第 {epoch + 1} 个epoch模型已保存至: {periodic_save_path}")

        # 打印日志
        epoch_time = time.time() - epoch_start
        log_str = f"[训练] Epoch [{epoch + 1}/{args.epochs}] | Time: {epoch_time:.2f}s | " \
                  f"Total Loss: {avg_total:.6f} | Decom Loss: {avg_d:.6f} | " \
                  f"Enhance Loss: {avg_e:.6f}"
       # print(log_str)

        # 写入训练日志（追加模式，避免覆盖）
        with open(loss_log_path, "a" if resume_epoch > 0 or epoch > 0 else "w") as f:
            f.write(log_str + "\n")

        # 每1个epoch验证
        if (epoch + 1) % 1 == 0:
            print(f"[验证] Evaluating / epoch {epoch + 1}...")
            model.eval()
            # --- 新增：在这里一次性初始化LPIPS模型 ---
            lpips_model = lpips.LPIPS(net='alex').to(device)
            lpips_model.eval()
            print("LPIPS model loaded for validation.")  # 提示信息，你会看到它只打印一次
            total_ssim, total_psnr, total_lpips = 0.0, 0.0, 0.0
            num_samples = len(test_loader.dataset)

            with torch.no_grad():
                for idx, (x_low, x_normal, filename) in enumerate(test_loader):
                    x_low, x_normal = x_low.to(device), x_normal.to(device)
                    #print(f" Processing {idx + 1}/{num_samples}: {filename[0]}")

                    # 前向传播
                    low_R, low_L, R_gamma, L_gamma, x_gamma, enhance_L, enhance_img = model(x_low)

                    # 计算指标
                    ssim_val, psnr_val, lpips_val = calculate_metrics(enhance_img, x_normal, lpips_model)
                    total_ssim += ssim_val
                    total_psnr += psnr_val
                    total_lpips += lpips_val

                    # 模块可视化
                    #visualize_modules(
                    #    low_R, low_L, R_gamma, L_gamma, enhance_L, enhance_img, x_low, x_normal,
                    #    filename[0],
                    #)

                    # 单样本日志
                    #print(
                     #   f" Sample {filename[0]} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.4f} | LPIPS: {lpips_val:.4f}")

            # 平均指标
            avg_ssim = total_ssim / num_samples
            avg_psnr = total_psnr / num_samples
            avg_lpips = total_lpips / num_samples

            # 保存验证结果（追加模式）
            eval_log_path = os.path.join(args.vis_dir, "eval_metrics.txt")
            with open(eval_log_path, "a" if resume_epoch > 0 or epoch > 0 else "w") as f:
                f.write(f"[验证] Epoch {epoch + 1} Test Metrics (Average over {num_samples} samples):\n")
                f.write(f"Average SSIM: {avg_ssim:.4f}\n")
                f.write(f"Average PSNR: {avg_psnr:.4f}\n")
                f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
                f.write("-" * 50 + "\n")

            # 验证日志
            print("\n" + "=" * 50)
            print(f"[验证] Epoch {epoch + 1} Test Results (Average):")
            print(f"SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.4f} | LPIPS: {avg_lpips:.4f}")
            print("=" * 50 + "\n")

            # 恢复训练模式
            model.train()


def extract_epoch_from_filename(filename):
    """从模型文件名中提取epoch数，如model_epoch_38.pth返回38"""
    import re
    # 匹配model_epoch_数字.pth格式
    match = re.search(r'model_epoch_(\d+)\.pth', filename)
    if match:
        return int(match.group(1))
    # 处理最佳模型
    if filename == 'best_model.pth':
        return None  # 最佳模型需要从文件内部读取epoch
    return None


if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description="Low Light Enhancement (Uretinex+Noise2noise+Zero-DCE)")
    # 通用参数
    parser.add_argument("--epochs", type=int, default=20, help="总训练epoch数")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-6, help="初始学习率")
    parser.add_argument("--crop_size", type=int, default=64, help="Crop size")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    # 新增：指定保存目录
    parser.add_argument("--save_dir", type=str, default="E:/大创/20251016", help="模型和日志保存的根目录，保存地址修改在这里")
    # 续训参数
    parser.add_argument("--resume", action="store_true", default=False, help="是否开启断点续训")
    parser.add_argument("--resume_ckpt", type=str, default="E:/大创/20251004/ckpt/model_epoch_1.pth", help="续训模型路径，如果要接着训，修改这里的路径")
    # Uretinex参数
    parser.add_argument("--unfolding_round", type=int, default=3, help="Uretinex迭代轮次")
    parser.add_argument("--gamma", type=float, default=0.1, help="P的正则化参数")
    parser.add_argument("--lamda", type=float, default=0.1, help="Q的正则化参数")
    parser.add_argument("--Roffset", type=float, default=0.05, help="gamma增量")
    parser.add_argument("--Loffset", type=float, default=0.05, help="lamda增量")
    parser.add_argument("--tv_weight", type=float, default=0.01, help="TV损失权重")
    parser.add_argument("--norm_layer", type=str, default="batch", help="归一化层类型")
    parser.add_argument("--concat_L", type=bool, default=False, help="是否拼接L到R")
    # Zero-DCE参数
    parser.add_argument("--patch_size", type=int, default=16, help="L_exp patch size")
    parser.add_argument("--mean_val", type=float, default=0.5, help="L_exp目标均值")
    # Noise2noise参数
    parser.add_argument("--noise_model", type=tuple, default=('gaussian', 50), help="噪声类型")
    parser.add_argument("--noise2noise_res_layers", type=int, default=16, help="SRResnet残差层数量")
    args = parser.parse_args()

    # 1. 设备配置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 设置保存目录
    # 从根目录创建子文件夹
    args.ckpt_dir = os.path.join(args.save_dir, "ckpt")  # 模型保存目录
    args.vis_dir = os.path.join(args.save_dir, "visualization")  # 可视化/日志目录
    # 创建目录（不存在则创建）
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    print(f"模型保存路径：{args.ckpt_dir}")
    print(f"可视化/日志路径：{args.vis_dir}")

    # 3. 固定随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"已固定随机种子: {seed}（确保数据一致性）")

    # 4. 加载数据
    train_folder = ["D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low\\",
                    "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high\\"]
    train_Data = []
    fixed_rand_modes = [np.random.randint(0, 7) for _ in range(10)]
    for patch_id in range(10):
        rand_mode = fixed_rand_modes[patch_id]
        train_data = MyDataset(rand_mode, patch_size=128, folder=train_folder)
        train_Data.extend(train_data)
    print(f"训练数据数量: {len(train_Data)}")

    # 训练加载器
    train_loader = DataLoader(
        dataset=train_Data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        drop_last=True
    )

    # 测试加载器
    eval_folder = ["D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low\\",
                   "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high\\"]
    test_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()])
    test_dataset = LOLDataset(eval_folder[0], eval_folder[1], test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pin_memory=True
    )
    print(f"训练样本数: {len(train_loader.dataset)}, 测试样本数: {len(test_loader.dataset)}")

    # 5. 模型初始化 + 续训加载
    model = LLIE(args).to(device)
    # 仍然使用 AdamW (Adam with weight decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)

    # --- 创建 Warmup + Cosine Decay 组合调度器 ---
    # 1. 主调度器：余弦退火，负责预热结束后的学习率下降
    main_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5, eta_min=1e-7)
    # 2. 预热调度器：线性增加学习率
    #    - warmup_epochs: 预热持续的 epoch 数量，5 是一个很好的起点
    #    - start_factor: 学习率从 lr * start_factor 开始，这里设为 0.1
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    # 3. 组合调度器 (关键)
    #    - schedulers: 按顺序执行的调度器列表
    #    - milestones: 切换调度器的 epoch 节点。在第 5 个 epoch 结束后，从 warmup_scheduler 切换到 main_scheduler
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # 续训起始epoch
    resume_epoch = 0
    best_loss = float("inf")

    # 若开启续训，加载模型权重和相关状态
    if args.resume:
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(f"续训模型不存在: {args.resume_ckpt}")

        # 从文件名提取epoch
        ckpt_filename = os.path.basename(args.resume_ckpt)
        epoch_from_filename = extract_epoch_from_filename(ckpt_filename)

        # 加载保存的字典
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 如果是定期保存的模型（含epoch信息）
        if epoch_from_filename is not None:
            resume_epoch = epoch_from_filename  # 从文件名提取的epoch开始
            print(f"从文件名提取到已完成epoch: {epoch_from_filename}")
        else:
            # 最佳模型，从文件内部读取
            if "last_epoch" in checkpoint:
                resume_epoch = checkpoint["last_epoch"] + 1  # 加1开始下一轮
                print(f"从最佳模型提取到最后训练epoch: {checkpoint['last_epoch']}")
            elif "current_epoch" in checkpoint:
                resume_epoch = checkpoint["current_epoch"]
                print(f"从模型文件提取到当前epoch: {checkpoint['current_epoch']}")

        # 加载优化器和调度器状态
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 加载最佳损失
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"成功加载模型: {args.resume_ckpt}")
        print(f"历史最佳损失: {best_loss:.6f}")
        print(f"起始epoch: {resume_epoch}（目标总epoch: {args.epochs}）")
        print(f"还需训练: {args.epochs - resume_epoch} 个epochs")

    print(
        f"模型初始化完成: {args.unfolding_round} Uretinex rounds, {args.noise2noise_res_layers} Noise2noise res layers")

    # 6. 开始训练
    print(f"\n从Epoch {resume_epoch} 开始训练...")
    train(args, model, train_loader, test_loader, optimizer, scheduler, device, resume_epoch=resume_epoch)

    print("\n训练完成!")
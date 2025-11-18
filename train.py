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
import os
import argparse
import time
import datetime
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
from pytorch_msssim import ssim as ssim_torch
from Model import *  # 确保Model.py中LLIE类定义正确
from MyDataset import *  # 确保MyDataset、LOLDataset定义正确
from Loss import *  # 确保compute_total_loss函数定义正确
from trans_loss import send_best_metrics_email

def calculate_metrics(x_enhanced, x_normal, device="cpu"):
    """计算SSIM、PSNR、LPIPS（使用PyTorch SSIM）"""
    # 为节省显存，指标计算放在CPU上（显存吃紧时更稳）
    device = 'cpu'
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    b = x_enhanced.size(0)
    ssim_sum, psnr_sum, lpips_sum = 0.0, 0.0, 0.0

    with torch.no_grad():
        for i in range(b):
            # PyTorch SSIM（直接在GPU上计算）
            ssim_val = ssim_torch(x_enhanced[i].unsqueeze(0).cpu(),
                                  x_normal[i].unsqueeze(0).cpu(),
                                  data_range=1.0, size_average=True)
            ssim_sum += ssim_val.item()

            # PSNR（转换为numpy计算，已优化内存）
            enh_np = (x_enhanced[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            norm_np = (x_normal[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            psnr_val = psnr(enh_np, norm_np, data_range=255)
            psnr_sum += psnr_val

            # LPIPS（范围标准化到[-1,1]）
            enh_lpips = ((x_enhanced[i].unsqueeze(0).cpu()) * 2) - 1
            norm_lpips = ((x_normal[i].unsqueeze(0).cpu()) * 2) - 1
            lpips_val = lpips_model(enh_lpips, norm_lpips).item()
            lpips_sum += lpips_val

    return (ssim_sum / b, psnr_sum / b, lpips_sum / b)


def visualize_modules(low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img, x_low, x_normal, filename,
                      save_dir="./visualization"):
    """优化版本：减少内存使用的模块级可视化"""
    import matplotlib.pyplot as plt
    import gc
    # 在tensor2np函数前加打印
    # print(f"low_R范围: {torch.min(low_R):.4f} ~ {torch.max(low_R):.4f}")
    # print(f"enhance_img范围: {torch.min(enhance_img):.4f} ~ {torch.max(enhance_img):.4f}")
    os.makedirs(save_dir, exist_ok=True)

    # tensor转numpy（0-1）- 添加内存优化
    def tensor2np(tensor):
        tensor_clamped = torch.clamp(tensor, 0.0, 1.0)
        np_array = tensor_clamped.squeeze(0).permute(1, 2, 0).cpu().numpy()
        del tensor_clamped
        return np_array

    def single2three(tensor):
        tensor_clamped = torch.clamp(tensor, 0.0, 1.0)
        img = tensor_clamped.squeeze(0).squeeze(0).cpu().numpy()
        np_array = np.clip(np.stack([img, img, img], axis=2), 0.0, 1.0)
        del tensor_clamped
        return np_array

    # 对单通道照明图进行对比度增强（百分位拉伸）
    def single2three_contrast(tensor, low_pct=2.0, high_pct=98.0):
        tensor_clamped = torch.clamp(tensor, 0.0, 1.0)
        img = tensor_clamped.squeeze(0).squeeze(0).cpu().numpy()
        # 百分位拉伸，增强灰度波动范围
        lo, hi = np.percentile(img, [low_pct, high_pct])
        if hi > lo:
            img = (img - lo) / (hi - lo)
        else:
            # 退化处理：使用min-max
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)
        img = np.clip(img, 0.0, 1.0)
        np_array = np.stack([img, img, img], axis=2)
        del tensor_clamped
        return np_array

    # 稳健法：与参考的“交集百分位窗口”线性拉伸，避免过度与分块
    def single2three_stretch_with_ref_intersection(tensor, ref_tensor, self_pct=(5.0, 95.0), ref_pct=(5.0, 95.0), min_width=0.1):
        t = torch.clamp(tensor, 0.0, 1.0)
        r = torch.clamp(ref_tensor, 0.0, 1.0)
        img = t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        ref = r.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        slo, shi = np.percentile(img, [self_pct[0], self_pct[1]])
        rlo, rhi = np.percentile(ref, [ref_pct[0], ref_pct[1]])
        lo = max(slo, rlo)
        hi = min(shi, rhi)
        if hi <= lo:
            # 退化处理：使用自身min-max
            lo, hi = img.min(), img.max()
        # 若交集窗口过窄，围绕中位数扩展至最小宽度，避免近似二值化显示
        if hi - lo < min_width:
            med = np.percentile(img, 50.0)
            lo = max(0.0, med - min_width / 2.0)
            hi = min(1.0, med + min_width / 2.0)
        if hi > lo:
            out = (img - lo) / (hi - lo)
        else:
            out = img.copy()
        out = np.clip(out, 0.0, 1.0)
        return np.stack([out, out, out], axis=2)

    # 彩色图的仅可视化白平衡 + 对比度增强（不影响模型输出）
    def color_vis_enhance(img_tensor, pct=(1.0, 99.0)):
        # img_tensor: [1,3,H,W] in [0,1]
        t = torch.clamp(img_tensor, 0.0, 1.0)
        img = t.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32)  # HWC
        # 灰世界白平衡
        means = img.reshape(-1,3).mean(axis=0) + 1e-6
        gray = means.mean()
        gains = gray / means
        img_wb = np.clip(img * gains, 0.0, 1.0)
        # 按通道百分位拉伸
        out = np.empty_like(img_wb)
        for c in range(3):
            lo, hi = np.percentile(img_wb[...,c], [pct[0], pct[1]])
            if hi > lo:
                out[...,c] = (img_wb[...,c] - lo) / (hi - lo)
            else:
                mn, mx = img_wb[...,c].min(), img_wb[...,c].max()
                if mx > mn:
                    out[...,c] = (img_wb[...,c] - mn) / (mx - mn)
                else:
                    out[...,c] = img_wb[...,c]
        return np.clip(out, 0.0, 1.0)

    try:
        # 提取中间结果并释放原始tensor
        low_R_np = tensor2np(low_R)
        # low_L 保持原样显示
        low_L_np = single2three(low_L)
        gamma_R_np = tensor2np(gamma_R)
        # gamma_L 保持原样显示
        gamma_L_np = single2three(gamma_L)
        x_gamma_np = tensor2np(x_gamma)
        # Enhanced Illumination：用与 gamma_L 的交集百分位做线性拉伸（更稳健）
        enhance_L_np = single2three_stretch_with_ref_intersection(enhance_L, gamma_L)
        # Enhanced Image：仅可视化做轻量白平衡 + 通道百分位拉伸，减轻偏色、提升对比
        enhance_img_np = color_vis_enhance(enhance_img)
        x_low_np = tensor2np(x_low)
        x_normal_np = tensor2np(x_normal)

        # 立即释放原始tensor引用
        del low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img, x_low, x_normal

        # 绘制子图（减小尺寸和DPI）
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))
        axes = axes.flatten()
        plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)

        # 1. 输入低光图
        axes[0].imshow(x_low_np)
        axes[0].set_title("Input Low-Light", fontsize=9)
        axes[0].axis("off")

        # 2. Uretinex分解反射层
        axes[1].imshow(low_R_np)
        axes[1].set_title("Reflectance Low", fontsize=9)
        axes[1].axis("off")
        axes[2].imshow(gamma_R_np)
        axes[2].set_title("Reflectance Gamma", fontsize=9)
        axes[2].axis("off")

        # 3. Uretinex分解照明层
        axes[3].imshow(low_L_np)
        axes[3].set_title("Illumination Low", fontsize=9)
        axes[3].axis("off")
        axes[4].imshow(gamma_L_np)
        axes[4].set_title("Illumination Gamma", fontsize=9)
        axes[4].axis("off")

        # 4. Noise2noise去噪反射层
        axes[5].imshow(x_gamma_np)
        axes[5].set_title("Gamma Correction", fontsize=9)
        axes[5].axis("off")

        # 5. Zero-DCE增强照明层
        axes[6].imshow(enhance_L_np)
        axes[6].set_title("Enhanced Illumination", fontsize=9)
        axes[6].axis("off")

        # 6. 最终增强图vs正常光图
        axes[7].imshow(enhance_img_np)
        axes[7].set_title("Enhanced Image", fontsize=9)
        axes[7].axis("off")
        axes[8].imshow(x_normal_np)
        axes[8].set_title("Normal Image", fontsize=9)
        axes[8].axis("off")

        # 保存（降低DPI）
        save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_modules.png")
        plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.1)

        # 释放numpy数组内存
        del low_R_np, low_L_np, gamma_R_np, gamma_L_np, x_gamma_np, enhance_L_np, enhance_img_np, x_low_np, x_normal_np

    except Exception as e:
        print(f"可视化失败 {filename}: {e}")
    finally:
        plt.close('all')
        gc.collect()


def train(args, model, train_loader, test_loader, optimizer, scheduler, device, resume_epoch=0):
    """端到端训练（支持断点续训）"""
    model.train()
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
    # 记录基础学习率，便于warmup阶段切换
    base_lr_other = None
    base_lr_cc = None
    if len(optimizer.param_groups) >= 2:
        base_lr_other = optimizer.param_groups[0]['lr']
        base_lr_cc = optimizer.param_groups[1]['lr']

    # AMP scaler（仅在CUDA且启用AMP时使用）
    use_amp = (device.type == 'cuda') and getattr(args, 'use_amp', True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    grad_accum_steps = max(1, getattr(args, 'grad_accum_steps', 1))

    # 基于验证的早停与最佳模型记录（按 SSIM-LPIPS 评分）
    best_eval_score = -1e9
    best_eval_epoch = -1
    # 早停耐心：<0 表示禁用早停
    eval_patience = int(getattr(args, 'early_stop_patience', -1))
    eval_bad_count = 0
    last_best_metrics = None  # 记录最佳验证指标明细

    for epoch in range(resume_epoch, args.epochs):
        epoch_start = time.time()
        total_loss, decom_loss, illum_loss = 0.0, 0.0, 0.0

        # ------- 三阶段训练：预热2epoch(仅颜色)->联合3epoch->仅主干 -------
        cc_params = list(model.cc_gap.parameters()) + list(model.cc_conv.parameters()) + list(model.cc_fc.parameters())
        cc_ids = set(id(p) for p in cc_params)
        other_params = [p for p in model.parameters() if id(p) not in cc_ids]

        warmup_epochs = 2
        joint_epochs = 3
        if epoch < warmup_epochs:
            # Phase A: 仅训练颜色头
            for p in other_params:
                p.requires_grad = False
            for p in cc_params:
                p.requires_grad = True
            if len(optimizer.param_groups) >= 2:
                optimizer.param_groups[0]['lr'] = 0.0  # other_params 学习率置零
                if base_lr_cc is not None:
                    optimizer.param_groups[1]['lr'] = base_lr_cc
            phase_note = f"[Phase A - Warmup Color] Epoch {epoch+1}: color head only"
        elif epoch < (warmup_epochs + joint_epochs):
            # Phase B: 联合训练颜色头与主干
            for p in model.parameters():
                p.requires_grad = True
            if len(optimizer.param_groups) >= 2:
                if base_lr_other is not None:
                    optimizer.param_groups[0]['lr'] = base_lr_other
                if base_lr_cc is not None:
                    optimizer.param_groups[1]['lr'] = base_lr_cc
            phase_note = f"[Phase B - Joint] Epoch {epoch+1}: color head + backbone"
        else:
            # Phase C: 仅训练主干，冻结颜色头
            for p in cc_params:
                p.requires_grad = False
            for p in other_params:
                p.requires_grad = True
            if len(optimizer.param_groups) >= 2:
                if base_lr_other is not None:
                    optimizer.param_groups[0]['lr'] = base_lr_other   # 训练主干
                optimizer.param_groups[1]['lr'] = 0.0                 # 冻结颜色头
            phase_note = f"[Phase C - Backbone Only] Epoch {epoch+1}: freeze color head"
        print(phase_note)

        # 组装损失配置（从 args 读取一次即可）
        loss_cfg = {
            'exp_patch': int(getattr(args, 'exp_patch', 16)),
            'exp_mean_enhL': float(getattr(args, 'exp_mean_enhL', 0.85)),
            'exp_mean_R': float(getattr(args, 'exp_mean_R', 0.4)),
            'bin_low_thr': float(getattr(args, 'bin_low_thr', 0.05)),
            'bin_high_thr': float(getattr(args, 'bin_high_thr', 0.98)),
            'bin_penalty_w': float(getattr(args, 'bin_penalty_w', 3.0)),
            'tv_enh_w': float(getattr(args, 'tv_enh_w', 0.05)),
            'align_gamma_w': float(getattr(args, 'align_gamma_w', 0.25)),
            'amp_loss_w': float(getattr(args, 'amp_loss_w', 1.2)),
            'grad_align_w': float(getattr(args, 'grad_align_w', 0.1)),
            'dark_v_thr': float(getattr(args, 'dark_v_thr', 0.10)),
            'dark_s_thr': float(getattr(args, 'dark_s_thr', 0.20)),
            'dark_lift_thr': float(getattr(args, 'dark_lift_thr', 0.20)),
            'dark_lift_w': float(getattr(args, 'dark_lift_w', 2.0)),
            'illum_loss_scale': float(getattr(args, 'illum_loss_scale', 0.05)),
            'r_chroma_dir_w': float(getattr(args, 'r_chroma_dir_w', 2.0)),
            'r_chroma_rg_w': float(getattr(args, 'r_chroma_rg_w', 1.0)),
            'r_dir_v_thr': float(getattr(args, 'r_dir_v_thr', 0.25)),
            # 传递当前epoch与配对监督阶段长度，用于在Loss中关停配对颜色项
            'epoch': int(epoch),
            'paired_color_until': int(getattr(args, 'paired_color_until', 5)),
        }

        for batch_idx, (name, x_low, x_normal) in enumerate(train_loader):
            # 数据移至设备
            x_low = x_low.to(device)
            x_normal = x_normal.to(device)

            # 前向传播
            with torch.amp.autocast('cuda', enabled=use_amp):
                low_R, low_L, low_k, gamma_R, gamma_L, gamma_k, x_gamma, enhance_L, enhance_img = model(x_low)
                # 计算总损失
                loss, d_loss, i_loss = compute_total_loss(
                    x_low, low_R, low_L, low_k, gamma_R, gamma_L, gamma_k, x_gamma, enhance_L, enhance_img, x_normal,
                    loss_cfg=loss_cfg
                )

            # 反向传播（支持梯度累积+AMP）
            loss = loss / grad_accum_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # 累加损失
            batch_size = x_low.size(0)
            total_loss += loss.item() * batch_size
            decom_loss += d_loss.item() * batch_size
            illum_loss += i_loss.item() * batch_size

        # 平均损失
        avg_total = total_loss / len(train_loader.dataset)
        avg_d = decom_loss / len(train_loader.dataset)
        avg_i = illum_loss / len(train_loader.dataset)

        # 学习率调度将放在验证后，基于 eval_score 进行

        # 保存最佳模型（含调度器状态，用于续训）
        if avg_total < best_loss:
            best_loss = avg_total
            os.makedirs(args.ckpt_dir, exist_ok=True)
            # 保存模型权重 + 调度器状态（关键：续训时恢复学习率）
            save_dict = {
                "model_state_dict": model.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "last_epoch": epoch  # 记录最后训练的epoch
            }
            save_path = os.path.join(args.ckpt_dir, "best_model.pth")
            torch.save(save_dict, save_path)
            print(f"[续训] 模型已保存至: {save_path} (当前epoch: {epoch + 1}, 最佳损失: {best_loss:.6f})")

        # # 打印日志（标注续训epoch）
        # epoch_time = time.time() - epoch_start
        # log_str = f"[续训] Epoch [{epoch + 1}/{args.epochs}] | Time: {epoch_time:.2f}s | " \
        #           f"Total Loss: {avg_total:.6f} | Decom Loss: {avg_d:.6f} | " \
        #           f"Denoise Loss: {avg_n:.6f} | Illum Loss: {avg_i:.6f} | " \
        #           f"Recon Loss: {avg_r:.6f} | Best Loss: {best_loss:.6f}"
        # print(log_str)

        # -------------------------- 2. 新增：每隔10个epoch定期保存模型 --------------------------
        if args.epochs > 10:
            k = 4
        else :
            k = 0
        if (epoch + 1) % (1+k) == 0:  # epoch+1是1-based，确保第10、20、30...epoch保存
            os.makedirs(args.ckpt_dir, exist_ok=True)
            # 定期保存字典：包含模型、优化器、调度器状态（支持从该模型续训）
            periodic_save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器状态
                "scheduler_state_dict": scheduler.state_dict(),  # 保存学习率调度状态
                "current_epoch": epoch + 1,  # 记录当前完成的epoch（1-based，用户易读）
                "current_total_loss": avg_total,  # 记录当前epoch损失
                "best_loss_so_far": best_loss  # 记录截至当前的最佳损失
            }
            # 文件名包含epoch号（避免覆盖），例如"model_epoch_10.pth"
            periodic_save_path = os.path.join(args.ckpt_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(periodic_save_dict, periodic_save_path)
            print(f"[定期保存] 第 {epoch + 1} 个epoch模型已保存至: {periodic_save_path}")

        # 打印日志（标注续训epoch）
        epoch_time = time.time() - epoch_start
        log_str = f"[续训] Epoch [{epoch + 1}/{args.epochs}] | Time: {epoch_time:.2f}s | " \
                  f"Total Loss: {avg_total:.6f} | Decom Loss: {avg_d:.6f} | Illum Loss: {avg_i:.6f}"
        print(log_str)

        # 写入训练日志（追加模式，避免覆盖）
        loss_log_path = os.path.join(args.vis_dir, "train_loss_log.txt")
        with open(loss_log_path, "a" if resume_epoch > 0 or epoch > 0 else "w") as f:
            f.write(log_str + "\n")

        # 每1个epoch验证（修复test_loader作用域：传入函数）
        if (epoch + 1) % 1 == 0:
            print(f"[续训] Evaluating / epoch {epoch + 1}...")
            model.eval()
            total_ssim, total_psnr, total_lpips = 0.0, 0.0, 0.0
            num_samples = len(test_loader.dataset)

            with torch.no_grad():
                for idx, (x_low, x_normal, filename) in enumerate(test_loader):
                    x_low, x_normal = x_low.to(device), x_normal.to(device)
                    print(f" Processing {idx + 1}/{num_samples}: {filename[0]}")

                    # 前向传播
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        low_R, low_L, low_k, gamma_R, gamma_L, gamma_k, x_gamma, enhance_L, enhance_img = model(x_low)

                    # 检查中间输出
                    # print(f"分解输出 - low_R: [{low_R.min():.4f}, {low_R.max():.4f}]")
                    # print(f"分解输出 - low_L: [{low_L.min():.4f}, {low_L.max():.4f}]")

                    # 如果发现NaN，立即停止
                    if torch.isnan(low_R).any():
                        print("检测到NaN在low_R中！")
                        break

                    # 计算指标
                    ssim_val, psnr_val, lpips_val = calculate_metrics(enhance_img, x_normal, device)
                    total_ssim += ssim_val
                    total_psnr += psnr_val
                    total_lpips += lpips_val

                    # 模块可视化（修复参数顺序：原代码多传了enhance_L1，此处修正）
                    visualize_modules(
                        low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img, x_low, x_normal,
                        filename[0], save_dir=args.vis_dir
                    )

                    # 单样本日志
                    print(
                        f" Sample {filename[0]} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.4f} | LPIPS: {lpips_val:.4f}")

            # 平均指标
            avg_ssim = total_ssim / num_samples
            avg_psnr = total_psnr / num_samples
            avg_lpips = total_lpips / num_samples

            # 保存验证结果（追加模式）
            eval_log_path = os.path.join(args.vis_dir, "eval_metrics.txt")
            with open(eval_log_path, "a" if resume_epoch > 0 or epoch > 0 else "w") as f:
                f.write(f"[续训] Epoch {epoch + 1} Test Metrics (Average over {num_samples} samples):\n")
                f.write(f"Average SSIM: {avg_ssim:.4f}\n")
                f.write(f"Average PSNR: {avg_psnr:.4f}\n")
                f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
                f.write("-" * 50 + "\n")

            # 验证日志
            print("\n" + "=" * 50)
            print(f"[续训] Epoch {epoch + 1} Test Results (Average):")
            print(f"SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.4f} | LPIPS: {avg_lpips:.4f}")
            print("=" * 50 + "\n")

            # 恢复训练模式
            model.train()

            # 基于验证指标的最佳保存与早停（自监督不变，仅改选择策略）
            eval_score = float(avg_ssim - avg_lpips)
            if eval_score > best_eval_score + 1e-6:
                best_eval_score = eval_score
                best_eval_epoch = epoch + 1
                eval_bad_count = 0
                os.makedirs(args.ckpt_dir, exist_ok=True)
                best_eval_path = os.path.join(args.ckpt_dir, "best_eval.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_eval_score": best_eval_score,
                    "metrics": {"ssim": avg_ssim, "psnr": avg_psnr, "lpips": avg_lpips}
                }, best_eval_path)
                print(f"[Eval-Best] Saved best_eval at epoch {epoch+1} with score {best_eval_score:.4f} -> {best_eval_path}")
                # 记录并发送邮件通知新的最佳验证
                last_best_metrics = {"ssim": float(avg_ssim), "psnr": float(avg_psnr), "lpips": float(avg_lpips), "score": float(eval_score)}
                try:
                    send_best_metrics_email(
                        args,
                        title=f"New best eval at epoch {epoch + 1}",
                        best_epoch=epoch + 1,
                        best_metrics=last_best_metrics,
                        best_loss=float(best_loss) if isinstance(best_loss, (int, float)) else None,
                        total_epochs=int(getattr(args, 'epochs', 0))
                    )
                except Exception as e:
                    print(f"[Email] send failed (best-update): {e}")
            else:
                eval_bad_count += 1
                # 仅当设置了非负耐心值时启用早停
                if eval_patience >= 0 and eval_bad_count >= eval_patience:
                    print(f"[EarlyStop] No eval improvement for {eval_patience} epochs. Best epoch: {best_eval_epoch} (score={best_eval_score:.4f}).")
                    # 早停前发送一次汇总邮件
                    try:
                        send_best_metrics_email(
                            args,
                            title="Training early-stopped",
                            best_epoch=int(best_eval_epoch) if best_eval_epoch != -1 else None,
                            best_metrics=last_best_metrics,
                            best_loss=float(best_loss) if isinstance(best_loss, (int, float)) else None,
                            total_epochs=int(getattr(args, 'epochs', 0))
                        )
                    except Exception as e:
                        print(f"[Email] send failed (early-stop): {e}")
                    break

            # 基于验证分数的学习率调度（越大越好）
            scheduler.step(eval_score)

        # 每训练8个epoch后休息8分钟，缓解过热
        if (epoch + 1) % 8 == 0:
            cooldown_minutes = 7
            cooldown_seconds = cooldown_minutes * 60
            print(f"\n[Cooling] Completed {epoch + 1} epochs. Sleeping for {cooldown_minutes} minutes to cool down...\n")
            time.sleep(cooldown_seconds)

    # 正常训练结束后的汇总邮件（未早停情况）
    try:
        send_best_metrics_email(
            args,
            title="Training finished",
            best_epoch=int(best_eval_epoch) if best_eval_epoch != -1 else None,
            best_metrics=last_best_metrics,
            best_loss=float(best_loss) if isinstance(best_loss, (int, float)) else None,
            total_epochs=int(getattr(args, 'epochs', 0))
        )
    except Exception as e:
        print(f"[Email] send failed (finished): {e}")


if __name__ == "__main__":
    # 命令行参数（新增续训相关参数）
    parser = argparse.ArgumentParser(description="Low Light Enhancement (Uretinex+Noise2noise+Zero-DCE)")
    # 通用参数
    parser.add_argument("--epochs", type=int, default=16, help="总训练epoch数（原默认100）")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size（原代码实际用4，此处统一）")
    parser.add_argument("--lr", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--crop_size", type=int, default=64, help="Crop size")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt", help="模型保存目录（需包含best_model.pth）")
    parser.add_argument("--vis_dir", type=str, default="./test_visualization", help="可视化目录")
    # 续训关键参数
    parser.add_argument("--resume", action="store_true", help="是否开启断点续训")
    # parser.add_argument("--resume_epoch", type=int, default=1, help="续训起始epoch（从28开始）")
    parser.add_argument("--resume_ckpt", type=str, default=None)  # "./ckpt/model_epoch_5.pth", help="续训模型路径")
    # 颜色头预热训练（仅训练颜色修正头的epoch数）
    parser.add_argument("--color_warmup_epochs", type=int, default=4, help="前多少个epoch仅训练颜色修正头")
    # 早停控制：<0 表示禁用早停；>=0 使用该耐心次数
    parser.add_argument("--early_stop_patience", type=int, default=-1, help="验证无提升的耐心次数（<0 禁用早停）")
    # 显存优化相关
    parser.add_argument("--use_amp", action="store_true", help="启用混合精度(AMP)以节省显存")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数，用于在小显存下模拟大batch")

    # Uretinex参数
    parser.add_argument("--unfolding_round", type=int, default=3, help="Uretinex迭代轮次")
    parser.add_argument("--gamma", type=float, default=1.0, help="P的正则化参数")
    parser.add_argument("--lamda", type=float, default=1.0, help="Q的正则化参数")
    parser.add_argument("--Roffset", type=float, default=1.05, help="gamma增量")
    parser.add_argument("--Loffset", type=float, default=1.05, help="lamda增量")
    parser.add_argument("--tv_weight", type=float, default=0.1, help="TV损失权重")
    parser.add_argument("--norm_layer", type=str, default="batch", help="归一化层类型")
    parser.add_argument("--concat_L", type=bool, default=False, help="是否拼接L到R")
    # Zero-DCE参数
    parser.add_argument("--patch_size", type=int, default=16, help="L_exp patch size")
    parser.add_argument("--mean_val", type=float, default=0.65, help="L_exp目标均值")
    # Noise2noise参数
    parser.add_argument("--noise_model", type=tuple, default=('gaussian', 50), help="噪声类型")
    parser.add_argument("--noise2noise_res_layers", type=int, default=16, help="SRResnet残差层数量")
    # 分解网络选择：0=DecomNet_RTV，1=IterativeUretinex
    parser.add_argument("--select", type=int, default=0, choices=[0, 1], help="分解网络选择：0=RTV, 1=IterativeUretinex")
    # 降噪开关：0 关闭，1 开启（与 select 类似的枚举型开关）
    parser.add_argument("--denoise", type=int, default=1, choices=[0, 1], help="是否开启降噪：0 关闭，1 开启")
    # 颜色校正强度控制（主要用于RTV路径）
    parser.add_argument("--cc_gain", type=float, default=5, help="颜色头输出相对增强：>1 放大颜色矫正幅度，1 为原始")
    parser.add_argument("--wb_alpha_cap", type=float, default=0.8, help="偏绿触发时 WB 融合上限，默认0.6，调大更强")
    # 统一降噪强度（绝对强度，越大整体降噪越强；模型内部按图像噪声自适应分配）
    parser.add_argument("--denoise_strength", type=float, default=0.6,
                        help="统一降噪强度系数[0,1]，0 关闭，1 最强（模型内部根据输入自适应强弱）")

    # Loss 可调参数（统一暴露到命令行）
    parser.add_argument("--exp_patch", type=int, default=16, help="曝光损失patch大小（enhance_L与R共享）")
    parser.add_argument("--exp_mean_enhL", type=float, default=0.85, help="enhance_L曝光目标均值")
    parser.add_argument("--exp_mean_R", type=float, default=0.4, help="R曝光目标均值")
    parser.add_argument("--bin_low_thr", type=float, default=0.05, help="照明二值化惩罚低阈值")
    parser.add_argument("--bin_high_thr", type=float, default=0.98, help="照明二值化惩罚高阈值")
    parser.add_argument("--bin_penalty_w", type=float, default=3.0, help="边界惩罚权重")
    parser.add_argument("--tv_enh_w", type=float, default=0.05, help="enhance_L TV权重")
    parser.add_argument("--align_gamma_w", type=float, default=0.25, help="与gamma_L对齐权重")
    parser.add_argument("--amp_loss_w", type=float, default=1.2, help="振幅差异约束权重")
    parser.add_argument("--grad_align_w", type=float, default=0.1, help="梯度对齐权重")
    parser.add_argument("--dark_v_thr", type=float, default=0.3, help="暗区拉升V阈值")
    parser.add_argument("--dark_s_thr", type=float, default=0.5, help="暗区拉升S阈值")
    parser.add_argument("--dark_lift_thr", type=float, default=0.20, help="暗区拉升目标下界")
    parser.add_argument("--dark_lift_w", type=float, default=2.5, help="暗区拉升损失权重")
    parser.add_argument("--illum_loss_scale", type=float, default=0.05, help="照明损失总权重缩放")
    args = parser.parse_args()

    # 1. 设备配置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    new_root_dir = f"20251116"  # 新根目录名称，可自定义前缀

    # 定义新的子文件夹路径（模型权重+可视化结果）
    new_ckpt_dir = os.path.join(new_root_dir, "ckpt")  # 新模型保存目录
    new_vis_dir = os.path.join(new_root_dir, "visualization")  # 新可视化/日志目录

    # 自动创建所有文件夹（不存在则创建，存在不报错）
    os.makedirs(new_ckpt_dir, exist_ok=True)
    os.makedirs(new_vis_dir, exist_ok=True)
    print(f"✅ 新结果目录已创建：{new_root_dir}")
    print(f"  - 模型权重路径：{new_ckpt_dir}")
    print(f"  - 可视化/日志路径：{new_vis_dir}")

    # 3. 覆盖原有args中的保存路径（关键：让代码所有保存操作指向新目录）
    args.ckpt_dir = new_ckpt_dir  # 模型保存目录指向新子文件夹
    args.vis_dir = new_vis_dir  # 可视化/日志目录指向新子文件夹

    # 2. 固定随机种子（关键：确保续训时数据和之前一致）
    seed = 42  # 可自定义，需与首次训练时一致
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[续训] 已固定随机种子: {seed}（确保数据一致性）")

    # 3. 加载数据（与原代码一致，但固定rand_mode避免随机性）
    train_folder = ["D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low\\",
                    "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high\\"]
    train_Data = []
    # 续训关键：固定rand_mode（首次训练时用的是随机，此处用固定值确保数据一致）
    fixed_rand_modes = [np.random.randint(0, 7) for _ in range(10)]  # 首次训练时的rand_mode列表，可从日志获取
    for patch_id in range(10):
        rand_mode = fixed_rand_modes[patch_id]  # 用固定的rand_mode
        train_data = MyDataset(rand_mode, patch_size=args.crop_size, folder=train_folder)
        train_Data.extend(train_data)
    print(f" Number of training data: {len(train_Data)}")

    # 训练加载器（与原代码一致）
    train_loader = DataLoader(
        dataset=train_Data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # 测试加载器（与原代码一致）
    eval_folder = ["D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\low\\",
                   "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\eval15\\high\\"]
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = LOLDataset(eval_folder[0], eval_folder[1], test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    print(f" Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # 4. 模型初始化 + 续训加载（核心步骤）
    model = LLIE(args).to(device)
    # 为颜色校正头设置更高学习率（加速颜色收敛）
    cc_params = list(model.cc_gap.parameters()) + list(model.cc_conv.parameters()) + list(model.cc_fc.parameters())
    cc_param_ids = set(id(p) for p in cc_params)
    other_params = [p for p in model.parameters() if id(p) not in cc_param_ids]
    optimizer = optim.Adam([
        {"params": other_params, "lr": args.lr},
        {"params": cc_params, "lr": max(1e-4, 10 * args.lr)}  # cc头用10倍学习率
    ])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 按验证指标“越大越好”调整
        patience=args.epochs // 4,
        factor=0.5,
        verbose=True
    )

    # 若开启续训，加载模型权重和调度器状态
    resume_epoch = 0
    best_loss = float("inf")  # 优先使用命令行传入的续训起始epoch
    if args.resume:
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(f"续训模型不存在: {args.resume_ckpt}")

        # 加载保存的字典（含模型、优化器、调度器、损失）
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型权重
        # 加载优化器状态（若参数组结构变化则跳过恢复，避免报错）
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 加载优化器状态
        except Exception as e:
            print(f"优化器状态加载失败，使用当前优化器设置: {e}")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # 加载学习率状态
        best_loss = checkpoint.get("best_loss", float("inf"))  # 兼容最佳模型和定期模型
        # 若定期模型包含current_epoch，可自动获取续训起始epoch（优先命令行参数）
        if "current_epoch" in checkpoint:
            resume_epoch = checkpoint["current_epoch"]  # 从定期模型的current_epoch继续
        else:
            # 兼容旧版本模型文件
            resume_epoch = checkpoint.get("last_epoch", 0) + 1
        print(f" 成功加载模型: {args.resume_ckpt}")
        print(f" 历史最佳损失: {best_loss:.6f}")
        print(f" 起始epoch: {resume_epoch}（目标总epoch: {args.epochs}）")
        print(f"还需训练: {args.epochs - resume_epoch} 个epochs")
    print(
        f"Model initialized: {args.unfolding_round} Uretinex rounds, {args.noise2noise_res_layers} Noise2noise res layers")

    # 5. 开始训练（传入test_loader，修复原代码作用域bug）
    print("\nStart Training from Epoch {}...".format(resume_epoch))
    train(args, model, train_loader, test_loader, optimizer, scheduler, device, resume_epoch=resume_epoch)

    print("\nAll Done!")

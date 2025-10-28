# train_decom.py

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# --- 从你的项目中导入必要的模块 ---
from Model import LLIE  # 导入你的主模型
from loss_DecomNet import DecomLoss  # 导入我们刚刚创建的DecomLoss


# 假设你的数据加载器在一个名为 data.py 的文件中
# from data import LowLightDataset

# --- 这是一个数据加载器的占位符，请替换为你自己的 ---
class PlaceholderDataset(torch.utils.data.Dataset):
    def __init__(self, length=100):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 返回成对的低光和高光图
        # 在你的模型中，高光图是通过gamma变换生成的，所以只需要低光图
        low_light_img = torch.rand(3, 256, 256)
        return {"low_light_image": low_light_img}


def main(args):
    # --- 1. 环境和设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 数据加载 ---
    # 请替换成你自己的 Dataset
    train_dataset = PlaceholderDataset(length=4850)  # 示例
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Number of training data: {len(train_dataset)}")

    # --- 3. 模型初始化 ---
    # 实例化你的主模型 LLIE
    model = LLIE(args).to(device)

    # !!! 关键步骤：冻结 EnhanceNet 的参数 !!!
    # 这样优化器就不会更新它的权重
    for param in model.enhance_net.parameters():
        param.requires_grad = False
    print("EnhanceNet parameters frozen.")

    # 确保 DecomNet 是可训练的 (默认就是，但明确写出来更清晰)
    for param in model.decom_net.parameters():
        param.requires_grad = True

    # --- 4. 定义优化器和损失函数 ---
    # !!! 关键步骤：优化器只包含 decom_net 的参数 !!!
    optimizer = optim.Adam(model.decom_net.parameters(), lr=args.lr)

    # 实例化我们的 DecomLoss
    criterion = DecomLoss(device)

    # --- 5. 开始训练 ---
    print("Start Training DecomNet...")
    for epoch in range(args.epochs):
        model.decom_net.train()  # 将 decom_net 设置为训练模式
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            low_img = batch['low_light_image'].to(device)

            optimizer.zero_grad()

            # --- 前向传播 ---
            # 你的模型前向传播会返回很多东西，我们只取 DecomNet 需要的
            # 使用 _ 来忽略 enhance_net 的输出
            low_R, low_L, gamma_R, gamma_L, x_gamma, _, _ = model(low_img)

            # --- 计算损失 ---
            loss_dict = criterion(low_img, x_gamma, low_R, low_L, gamma_R, gamma_L)
            total_loss = loss_dict["total"]

            # --- 反向传播和优化 ---
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decom_net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

            if (i + 1) % args.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {total_loss.item():.4f}")
                # 打印各项子损失，方便调试
                print(f"  > Recon: {loss_dict['recon']:.4f}, Consist_R: {loss_dict['consistency_R']:.4f}, "
                      f"Smooth_L: {loss_dict['smooth_L']:.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch [{epoch + 1}/{args.epochs}] Finished, Average Loss: {avg_epoch_loss:.4f} ---")

        # --- 6. 保存模型 ---
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'decom_net_epoch_{epoch + 1}.pth')
            torch.save(model.decom_net.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("DecomNet training finished!")
    # 保存最终模型
    torch.save(model.decom_net.state_dict(), os.path.join(args.save_dir, 'decom_net_final.pth'))


if __name__ == '__main__':
    # --- 7. 设置参数 ---
    parser = argparse.ArgumentParser(description="Train DecomNet Separately")

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_dir', type=str, default='decom_ckpt', help='directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=50, help='interval for logging')
    parser.add_argument('--save_interval', type=int, default=10, help='interval for saving model')

    # 模型所需的参数 (从你的 Model.py main 函数中复制)
    parser.add_argument("--unfolding_round", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument("--Roffset", type=float, default=0.05)
    parser.add_argument("--Loffset", type=float, default=0.05)
    parser.add_argument("--concat_L", type=bool, default=False)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main(args)


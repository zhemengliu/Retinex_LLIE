# test_fix.py
import torch
from Model import LLIE
from MyDataset import MyDataset
import argparse


def test_fix():
    train_folder = ["D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low\\",
                    "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high\\"]

    # 加载数据
    dataset = MyDataset(rand_mode=0, patch_size=128, folder=train_folder)
    name, x_low, x_normal = dataset[0]

    print(f"修复后数据检查:")
    print(f"输入 - low: [{x_low.min():.4f}, {x_low.max():.4f}], normal: [{x_normal.min():.4f}, {x_normal.max():.4f}]")

    # 模型参数
    args = argparse.Namespace(
        unfolding_round=3,
        gamma=0.1,
        lamda=0.1,
        Roffset=0.02,  # 减小增量
        Loffset=0.02,
        concat_L=False
    )

    model = LLIE(args)

    with torch.no_grad():
        outputs = model(x_low.unsqueeze(0))

    low_R, low_L, gamma_R, gamma_L, x_gamma, enhance_L, enhance_img = outputs

    print(f"\n修复后输出范围:")
    print(f"low_R: [{low_R.min():.4f}, {low_R.max():.4f}]")
    print(f"low_L: [{low_L.min():.4f}, {low_L.max():.4f}]")
    print(f"enhance_L: [{enhance_L.min():.4f}, {enhance_L.max():.4f}]")
    print(f"enhance_img: [{enhance_img.min():.4f}, {enhance_img.max():.4f}]")

    # 检查增强效果
    enhancement_ratio = enhance_img.mean() / x_low.mean()
    print(f"增强倍数: {enhancement_ratio:.2f}x")


if __name__ == "__main__":
    test_fix()
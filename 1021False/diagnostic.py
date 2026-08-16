# diagnostic.py
import torch
import argparse
from Model import LLIE
from MyDataset import MyDataset


def diagnostic_test():
    # 使用小批量数据进行测试
    train_folder = ["D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\low\\",
                    "D:\\low_light_image\\Dataset\\Dataset\\LOLdataset\\our485\\high\\"]

    # 加载单张图像测试
    dataset = MyDataset(rand_mode=0, patch_size=128, folder=train_folder)
    name, x_low, x_normal = dataset[0]  # 第一张图像

    print(f"测试图像: {name}")
    print(
        f"输入数据范围: low [{x_low.min():.4f}, {x_low.max():.4f}], normal [{x_normal.min():.4f}, {x_normal.max():.4f}]")

    # 模型参数
    args = argparse.Namespace(
        unfolding_round=3,
        gamma=0.1,
        lamda=0.1,
        Roffset=0.05,
        Loffset=0.05,
        concat_L=False
    )

    # 初始化模型
    model = LLIE(args)

    # 单张图像前向传播
    with torch.no_grad():
        outputs = model(x_low.unsqueeze(0))  # 添加batch维度

    print("\n诊断完成!")
    return outputs


if __name__ == "__main__":
    diagnostic_test()
import torch
import argparse
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Decom_Model import IterativeUretinex, Decom
from Illum_Model import enhance_net_nopool
from Model import LLIE


def create_test_args():
    """创建测试参数"""
    args = argparse.Namespace()
    args.unfolding_round = 3
    args.gamma = 0.1
    args.lamda = 0.1
    args.Roffset = 0.05
    args.Loffset = 0.05
    args.concat_L = False
    args.tv_weight = 0.01
    args.norm_layer = "batch"
    return args


def test_decom_model():
    """测试分解模型"""
    print("=== 测试分解模型 ===")
    args = create_test_args()

    # 测试基础分解
    print("1. 测试基础分解网络...")
    decom = Decom().cuda()
    x = torch.rand(1, 3, 128, 128).cuda()

    with torch.no_grad():
        R, L = decom(x)
        print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
        print(f"反射层R: [{R.min():.4f}, {R.max():.4f}] shape: {R.shape}")
        print(f"光照层L: [{L.min():.4f}, {L.max():.4f}] shape: {L.shape}")

        # 检查重建
        recon = R * L
        recon_error = torch.mean(torch.abs(x - recon))
        print(f"重建误差: {recon_error:.6f}")

    # 测试迭代分解
    print("\n2. 测试迭代分解网络...")
    model = IterativeUretinex(args).cuda()

    with torch.no_grad():
        R, L = model(x)
        print(f"迭代分解 - 反射层: [{R.min():.4f}, {R.max():.4f}]")
        print(f"迭代分解 - 光照层: [{L.min():.4f}, {L.max():.4f}]")

        recon = R * L
        recon_error = torch.mean(torch.abs(x - recon))
        print(f"迭代重建误差: {recon_error:.6f}")


def test_illum_model():
    """测试照明增强模型"""
    print("\n=== 测试照明增强模型 ===")
    model = enhance_net_nopool().cuda()

    # 测试单通道输入（光照层）
    x = torch.rand(1, 1, 128, 128).cuda()

    with torch.no_grad():
        enhance_L1, enhance_L, r = model(x)
        print(f"输入照明: [{x.min():.4f}, {x.max():.4f}] shape: {x.shape}")
        print(f"中间增强: [{enhance_L1.min():.4f}, {enhance_L1.max():.4f}] shape: {enhance_L1.shape}")
        print(f"最终增强: [{enhance_L.min():.4f}, {enhance_L.max():.4f}] shape: {enhance_L.shape}")
        print(f"增强系数: [{r.min():.4f}, {r.max():.4f}] shape: {r.shape}")


def test_full_model():
    """测试完整模型"""
    print("\n=== 测试完整模型 ===")
    args = create_test_args()

    model = LLIE(args).cuda()
    x = torch.rand(1, 3, 128, 128).cuda()

    print("模型结构:")
    print(f"  - 分解网络: {type(model.decom_net).__name__}")
    print(f"  - 增强网络: {type(model.enhance_net).__name__}")

    with torch.no_grad():
        outputs = model(x)
        names = ['low_R', 'low_L', 'gamma_R', 'gamma_L', 'enhance_L1', 'enhance_L', 'x_gamma', 'enhance_img']

        print("\n各输出层信息:")
        for name, output in zip(names, outputs):
            print(f"  {name:15} range: [{output.min():.4f}, {output.max():.4f}] shape: {output.shape}")

        # 检查最终增强结果
        enhance_img = outputs[-1]
        print(f"\n最终增强图像:")
        print(f"  像素范围: [{enhance_img.min():.4f}, {enhance_img.max():.4f}]")
        print(f"  均值: {enhance_img.mean():.4f}")
        print(f"  标准差: {enhance_img.std():.4f}")


def check_parameter_requirements():
    """检查模型参数要求"""
    print("\n=== 检查模型参数要求 ===")

    # 检查分解网络输入要求
    decom = Decom()
    print(f"分解网络输入要求: 3通道RGB图像")
    print(f"分解网络输出: 反射层(3通道) + 光照层(1通道)")

    # 检查增强网络输入要求
    enhance = enhance_net_nopool()
    print(f"增强网络输入要求: 1通道光照层")
    print(f"增强网络输出: 增强光照层(1通道) + 增强系数(8通道)")


if __name__ == "__main__":
    print("开始模型诊断...")

    try:
        test_decom_model()
        test_illum_model()
        test_full_model()
        check_parameter_requirements()
        print("\n✅ 所有模型测试通过!")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTransformer(nn.Module):
    """多尺度Transformer用于学习S型曲线参数"""

    def __init__(self, in_channels=1, base_channels=32):
        super(MultiScaleTransformer, self).__init__()

        # 多尺度特征提取
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, base_channels, 5, 1, 2)
        self.conv3 = nn.Conv2d(in_channels, base_channels, 7, 1, 3)

        # 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels * 3, base_channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 3, 1, 1, 0),
            nn.Sigmoid()
        )

        # 全局特征提取
        self.global_conv = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, 3, 1, 1, 0)  # 输出a, k, b三个参数
        )

    def forward(self, x):
        # 多尺度特征
        f1 = F.leaky_relu(self.conv1(x), 0.2)
        f2 = F.leaky_relu(self.conv2(x), 0.2)
        f3 = F.leaky_relu(self.conv3(x), 0.2)

        # 特征融合
        features = torch.cat([f1, f2, f3], dim=1)

        # 通道注意力
        attention = self.se(features)
        features = features * attention

        # 全局参数预测
        params = self.global_conv(features)
        params = params.squeeze(-1).squeeze(-1)  # [B, 3]

        # 确保参数在合理范围
        a = torch.sigmoid(params[:, 0:1]) * 2 + 0.5  # [0.5, 2.5]
        k = torch.sigmoid(params[:, 1:2]) * 10 + 1  # [1, 11]
        b = torch.sigmoid(params[:, 2:3]) * 0.4 + 0.3  # [0.3, 0.7]

        return a, k, b


class Illum_Model(nn.Module):
    def __init__(self):
        super(Illum_Model, self).__init__()
        self.param_net = MultiScaleTransformer()

    def forward(self, I_low):
        """
        I_low: 低光照明层 [B, 1, H, W]
        返回增强后的照明层
        """
        batch_size, _, height, width = I_low.shape

        # 预测S型曲线参数
        a, k, b = self.param_net(I_low)  # [B,1], [B,1], [B,1]

        # 扩展参数维度以匹配输入
        a = a.view(batch_size, 1, 1, 1).expand(-1, -1, height, width)
        k = k.view(batch_size, 1, 1, 1).expand(-1, -1, height, width)
        b = b.view(batch_size, 1, 1, 1).expand(-1, -1, height, width)

        # S型曲线增强
        I_enhanced = a * torch.sigmoid(k * (I_low - b))

        return I_enhanced, a, k, b
import torch
import torch.nn as nn
import torch.fft


# =============================================================================
# --- 1. 新增：傅里葉变换辅助模块 ---
# =============================================================================

class FourierLayer(nn.Module):
    """
    将输入从空间域转换到频域。
    处理从 (B, C, H, W) 的实数张量到 (B, 2*C, H, W) 的实数张量的转换，
    其中通道维度翻倍以分别表示复数的实部和虚部。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对最后两个维度（H, W）进行2D傅里葉變換
        # norm='ortho' 是一种标准化的方法，有助于保持数值稳定性
        freq_domain = torch.fft.fft2(x, norm='ortho')

        # 将复数张量的实部和虚部沿着通道维度拼接起来
        # 神经网络的标准卷积层只能处理实数张量
        freq_domain_real_imag = torch.cat([freq_domain.real, freq_domain.imag], dim=1)

        return freq_domain_real_imag


class InverseFourierLayer(nn.Module):
    """
    将频域表示转换回空间域。
    处理从 (B, 2*C, H, W) 的实数张量回到 (B, C, H, W) 的实数张量的转换。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将拼接的实部和虚部分开
        real, imag = torch.chunk(x, 2, dim=1)

        # 重新组合成复数张量
        freq_domain = torch.complex(real, imag)

        # 进行2D逆傅里葉變換
        spatial_domain = torch.fft.ifft2(freq_domain, norm='ortho')

        # 取其实部作为最终结果。由于原始输入是实数，逆变换后的虚部理论上应接近于零。
        return spatial_domain.real


# =============================================================================
# --- 2. 核心修改：使用傅里葉变换的分解网络 ---
# =============================================================================

class DecomNet_Fourier(nn.Module):
    """
    一个混合模型，结合了空间域的局部特征提取和频域的全局信息处理。
    """

    def __init__(self, in_channels=3, mid_channels=64, out_channels_R=3, out_channels_L=1):
        super(DecomNet_Fourier, self).__init__()

        # 阶段1: 初始空间域卷积，用于提取局部特征
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # 阶段2: 傅里葉变换块，用于全局信息处理
        self.to_freq = FourierLayer()
        # 在频域中进行学习。输入通道数因实部和虚部分离而翻倍。
        # 使用1x1卷积在频域中高效地混合通道信息，这相当于在每个频率点上应用一个全连接层。
        self.freq_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=1, bias=True),
        )
        self.from_freq = InverseFourierLayer()

        # 阶段3: 后续空间域卷积，用于融合和细化特征
        # 使用残差连接来融合傅里葉块前后的特征
        self.final_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # 阶段4: 输出层，生成反射分量 R 和光照分量 L
        self.conv_R = nn.Conv2d(mid_channels, out_channels_R, kernel_size=3, padding=1, bias=True)
        self.conv_L = nn.Conv2d(mid_channels, out_channels_L, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # 1. 提取初始局部特征
        x1 = self.initial_conv(x)

        # 2. 进入频域进行全局处理
        freq_features = self.to_freq(x1)
        processed_freq_features = self.freq_conv(freq_features)
        x2 = self.from_freq(processed_freq_features)

        # 3. 融合局部和全局特征 (使用残差连接)
        # 这使得模型可以选择性地使用频域信息，增强了训练的稳定性
        x_fused = x1 + x2

        # 4. 最终特征细化
        x_final = self.final_conv(x_fused)

        # 5. 生成输出
        # 使用 Sigmoid 激活函数将输出限制在 [0, 1] 范围内，这符合 R 和 L 的物理意义
        R = torch.sigmoid(self.conv_R(x_final))
        L = torch.sigmoid(self.conv_L(x_final))

        return R, L


# =============================================================================
# --- 3. 保持不变：增强网络 ---
# =============================================================================

class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(64, 3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, L, R):
        x = torch.cat([L, R], dim=1)
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        x_r = self.conv7(torch.cat([x1, x6], dim=1))
        return x_r


# =============================================================================
# --- 4. 主模型：更新以使用新的 DecomNet_Fourier ---
# =============================================================================

class LLIE(nn.Module):
    def __init__(self, args=None):  # args 设为可选，方便单独实例化
        super(LLIE, self).__init__()

        # --- 关键修改：使用新的基于傅里葉的分解网络 ---
        self.decom_net = DecomNet_Fourier()

        self.enhance_net = EnhanceNet()
        self.relu = nn.ReLU(inplace=True)
        self.gamma = 2.2

    def forward(self, low_light_image):
        # 生成伪高光图像 x_gamma
        x_gamma = low_light_image ** (1 / self.gamma)

        # 使用 DecomNet 分解低光图像和伪高光图像
        R_low, L_low = self.decom_net(low_light_image)
        R_gamma, L_gamma = self.decom_net(x_gamma)

        # 使用 EnhanceNet 增强光照图
        L_enhance_low = self.enhance_net(L_low, R_low)
        L_enhance_gamma = self.enhance_net(L_gamma, R_gamma)

        # 得到最终增强后的图像
        I_low_3 = L_enhance_low * R_low
        I_gamma_3 = L_enhance_gamma * R_gamma

        return R_low, L_low, R_gamma, L_gamma, x_gamma, I_low_3, I_gamma_3

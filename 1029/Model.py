import torch
import torch.nn as nn
from Decom_Model import Decom_Model
from Denoise_Model import Denoise_Model
from Illum_Model import Illum_Model


class RetinexEnhanceModel(nn.Module):
    def __init__(self):
        super(RetinexEnhanceModel, self).__init__()
        self.decom_net = Decom_Model()
        self.denoise_net = Denoise_Model()
        self.illum_net = Illum_Model()

    def forward(self, low_img, gamma=2.2):
        batch_size = low_img.shape[0]

        # 1. Retinex分解
        R_low, L_low = self.decom_net(low_img)

        # 2. 生成gamma校正图像 (无监督训练的关键)
        gamma_img = torch.clamp(low_img ** (1 / gamma), 0, 1)
        R_gamma, L_gamma = self.decom_net(gamma_img)

        # 3. 反射层去噪
        # 为自监督训练生成两个噪声版本
        noise1 = torch.randn_like(R_low) * 0.05
        noise2 = torch.randn_like(R_low) * 0.05
        R_noisy1 = R_low + noise1
        R_noisy2 = R_low + noise2

        R_denoised = self.denoise_net(R_noisy1)

        # 4. 照明增强
        L_enhanced, a, k, b = self.illum_net(L_low)

        # 5. 图像重建
        enhanced_img = R_denoised * L_enhanced

        return {
            'R_low': R_low,
            'L_low': L_low,
            'R_gamma': R_gamma,
            'L_gamma': L_gamma,
            'R_denoised': R_denoised,
            'L_enhanced': L_enhanced,
            'enhanced_img': enhanced_img,
            'gamma_img': gamma_img,
            'R_noisy1': R_noisy1,
            'R_noisy2': R_noisy2,
            'params': (a, k, b)
        }
# loss_decom.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# 从你原有的 Loss.py 中复制 L_spa 和 L_TV 类
# 这两个类是 DecomNet 损失的重要组成部分

# 在 loss_decom.py 文件中

class L_spa(nn.Module):
    """空间一致性损失 (用于反射分量) - 已修正"""

    def __init__(self):
        super(L_spa, self).__init__()
        # 这些卷积核都是为单通道图像设计的
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        # --- 关键修正：在这里将3通道输入转为1通道灰度图 ---
        # torch.mean(..., dim=1, ...) 会在通道维度上求平均值
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        # ---------------------------------------------------

        # 现在 org_pool 和 enhance_pool 都是单通道的了
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

        D_left = torch.pow(D_org_left - D_enhance_left, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = torch.mean(D_left + D_right + D_up + D_down)
        return E


class L_TV(nn.Module):
    """照明平滑度损失 (Total Variation Loss)"""

    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


# --- 核心：DecomNet 的总损失函数 ---
class DecomLoss(nn.Module):
    def __init__(self, device):
        super(DecomLoss, self).__init__()
        self.mse_loss = nn.MSELoss().to(device)
        self.spa_loss = L_spa().to(device)
        self.tv_loss = L_TV().to(device)

        # 定义各项损失的权重 (你可以根据需要调整)
        self.lambda_recon = 1.0
        self.lambda_consistency = 10.0
        self.lambda_spa = 1.0
        self.lambda_tv = 1.0
        self.lambda_l_con = 0.1

    def forward(self, low_img, x_gamma, low_R, low_L, gamma_R, gamma_L):
        """
        计算 DecomNet 的总损失
        """
        # 1. 重建损失 (Reconstruction Loss)
        # 分解后的 R 和 L 相乘，应该能还原出原始图像
        loss_recon_low = self.mse_loss(low_R * low_L, low_img)
        loss_recon_gamma = self.mse_loss(gamma_R * gamma_L, x_gamma)
        total_recon_loss = loss_recon_low + loss_recon_gamma

        # 2. 反射一致性损失 (Reflectance Consistency Loss)
        # 低光和伪高光的反射图 R 应该是一致的
        loss_consistency = self.mse_loss(low_R, gamma_R)

        # 3. 反射空间损失 (Reflectance Spatial Loss)
        # 反射图 R 应该保留原始图像的结构
        loss_spa = self.spa_loss(low_R, low_img) + self.spa_loss(gamma_R, x_gamma)

        # 4. 光照平滑损失 (Illumination Smoothness Loss)
        # 光照图 L 应该是平滑的
        loss_tv = self.tv_loss(low_L) + self.tv_loss(gamma_L)

        # 5. 光照初始一致性损失 (Illumination Consistency)
        # 你的代码中有一个隐式的损失，即 L 应该接近于图像的最大通道值
        low_L0 = torch.max(low_img, dim=1, keepdim=True)[0]
        gamma_L0 = torch.max(x_gamma, dim=1, keepdim=True)[0]
        loss_l_con = self.mse_loss(low_L, low_L0) + self.mse_loss(gamma_L, gamma_L0)

        # --- 计算加权总损失 ---
        total_loss = (self.lambda_recon * total_recon_loss +
                      self.lambda_consistency * loss_consistency +
                      self.lambda_spa * loss_spa +
                      self.lambda_tv * loss_tv +
                      self.lambda_l_con * loss_l_con)

        # 返回总损失和各项子损失，方便监控
        loss_dict = {
            "total": total_loss,
            "recon": total_recon_loss,
            "consistency_R": loss_consistency,
            "spa_R": loss_spa,
            "smooth_L": loss_tv,
            "consistency_L": loss_l_con
        }
        return loss_dict

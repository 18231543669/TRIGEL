"""
损失函数模块
包含对比学习损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """对比学习损失函数 - 简化高效版本"""

    def __init__(self, temperature: float = 0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失 - 简化版本
        Args:
            z1: 原始数据的嵌入 [N, D]
            z2: 增强数据的嵌入 [N, D]
        """
        # 归一化嵌入向量
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2N, D]

        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]

        # 创建标签：前N个对应后N个
        labels = torch.arange(N).to(z1.device)
        labels = torch.cat([labels + N, labels])  # [0->N, 1->N+1, ..., N-1->2N-1, 0->N, 1->N+1, ...]

        # 移除自连接
        mask = torch.eye(2 * N, device=z1.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))

        # 计算损失
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

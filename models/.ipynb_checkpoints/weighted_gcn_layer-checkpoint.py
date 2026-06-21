"""
支持边权重的GCN层
在标准GCN基础上添加边权重支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedGCNConv(nn.Module):
    """支持边权重的GCN卷积层"""
    
    def __init__(self, in_features: int, out_features: int):
        super(WeightedGCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [num_nodes, in_features]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges]，如果为None则所有边权重为1
        """
        # 线性变换
        x = self.linear(x)
        
        # 聚合邻居信息
        num_nodes = x.size(0)
        
        # 计算度（考虑边权重）
        row, col = edge_index[0], edge_index[1]
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        
        # 计算归一化系数
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 归一化边权重
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # 消息传递
        out = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            out[dst] += norm[i] * x[src]
        
        return out


def weighted_gcn_conv(x: torch.Tensor, edge_index: torch.Tensor, 
                     weight: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    函数式接口：支持边权重的GCN卷积
    
    Args:
        x: 节点特征 [num_nodes, in_features]
        edge_index: 边索引 [2, num_edges]
        weight: 权重矩阵 [in_features, out_features]
        edge_weight: 边权重 [num_edges]
    """
    # 线性变换
    x = torch.matmul(x, weight)
    
    # 聚合
    num_nodes = x.size(0)
    row, col = edge_index[0], edge_index[1]
    
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=x.device)
    
    # 度归一化
    deg = torch.zeros(num_nodes, device=x.device)
    deg.scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # 归一化
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    # 聚合
    out = torch.zeros_like(x)
    out.index_add_(0, col, norm.unsqueeze(1) * x[row])
    
    return out

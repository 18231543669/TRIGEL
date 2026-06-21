"""
训练和评估模块
包含模型训练和评估相关函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from models import GCNContrastiveModel


def train_with_contrastive_learning(model: GCNContrastiveModel,
                                   x: torch.Tensor, x_aug: torch.Tensor,
                                   edge_indices: List[torch.Tensor],
                                   edge_indices_aug: List[torch.Tensor],
                                   labels: torch.Tensor, train_idx: torch.Tensor,
                                   optimizer: torch.optim.Optimizer,
                                   classification_criterion: nn.Module,
                                   edge_weights: Optional[List[torch.Tensor]] = None,
                                   edge_weights_aug: Optional[List[torch.Tensor]] = None,
                                   cb_weight: float = 0.6, cl_weight: float = 0.4) -> Dict:
    """
    带对比学习的训练步骤
    
    Args:
        model: GCN对比学习模型
        x: 原始特征
        x_aug: 增强特征
        edge_indices: 原始边索引列表
        edge_indices_aug: 增强边索引列表
        labels: 标签
        train_idx: 训练索引
        optimizer: 优化器
        classification_criterion: 分类损失函数
        edge_weights: 原始边权重列表（可选）
        edge_weights_aug: 增强边权重列表（可选）
        cb_weight: 分类损失权重
        cl_weight: 对比学习损失权重
    """
    model.train()
    optimizer.zero_grad()
    
    # 创建训练掩码
    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
    train_mask[train_idx] = True
    
    # 前向传播 - 分类（使用边权重）
    logits = model(x_aug, edge_indices_aug, edge_weights_aug)
    classification_loss = classification_criterion(logits[train_idx], labels[train_idx])
    
    # 计算对比学习损失
    contrastive_loss = model.compute_contrastive_loss(
        x, x_aug, edge_indices, edge_indices_aug,
        edge_weights, edge_weights_aug, labels, train_mask
    )
    
    # 组合损失
    total_loss = cb_weight * classification_loss + cl_weight * contrastive_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    
    # 计算训练准确率
    train_pred = logits[train_idx].argmax(dim=1)
    train_acc = (train_pred == labels[train_idx]).float().mean().item()
    
    return {
        'total_loss': total_loss.item(),
        'classification_loss': classification_loss.item(),
        'contrastive_loss': contrastive_loss.item(),
        'train_acc': train_acc
    }


def evaluate_model(model: nn.Module, x: torch.Tensor, edge_indices: List[torch.Tensor],
                  eval_idx: torch.Tensor, 
                  edge_weights: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    评估模型
    
    Args:
        model: 要评估的模型
        x: 输入特征
        edge_indices: 边索引列表
        eval_idx: 评估样本的索引
        edge_weights: 边权重列表（可选）
    
    Returns:
        predictions: 预测标签
        probabilities: 预测概率
    """
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_indices, edge_weights)
        predictions = logits[eval_idx].argmax(dim=1)
        probabilities = F.softmax(logits[eval_idx], dim=1)
    
    return predictions, probabilities
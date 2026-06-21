"""
GCN模型模块
包含基础GCN和对比学习GCN的实现

改进点：
1. 融合模块从简单加权 → 节点级动态注意力
2. 支持融合模块的超参数配置
3. 保持完整的向后兼容性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import List, Optional, Dict

from .graph_fusion import AttentionFusion


class MultiOmicsGCN(nn.Module):
    """多组学图卷积网络（基础版本）
    
    该模型为每个组学数据构建独立的GCN，然后通过Attention融合得到融合特征，
    最后拼接所有组学特征和融合特征进行分类。
    支持边权重。
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_omics: int,
                 num_classes: int, gcn_layers: int = 2,
                 fusion_num_heads: int = 4, fusion_dropout: float = 0.3):
        """初始化多组学GCN模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_omics: 组学数量
            num_classes: 类别数量
            gcn_layers: GCN层数
            fusion_num_heads: 融合模块的注意力头数（新增）
            fusion_dropout: 融合模块的dropout率（新增）
        """
        super().__init__()
        self.num_omics = num_omics
        self.gcn_layers = gcn_layers
        self.hidden_dim = hidden_dim

        # 为每个组学创建独立的GCN层
        self.gcn_list = nn.ModuleList()
        for _ in range(num_omics):
            layers = nn.ModuleList()
            for layer_idx in range(gcn_layers):
                if layer_idx == 0:
                    # 第一层：input_dim -> hidden_dim
                    layers.append(GCNConv(input_dim, hidden_dim))
                else:
                    # 后续层：hidden_dim -> hidden_dim
                    layers.append(GCNConv(hidden_dim, hidden_dim))
            self.gcn_list.append(layers)

        # ✨ 改进：使用节点级动态注意力融合（替换原来的简单加权融合）
        self.fusion_attention = AttentionFusion(
            num_omics=num_omics,
            hidden_dim=hidden_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout
        )

        # 分类器（输入维度为 num_omics 个组学特征 + 1 个融合特征）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (num_omics + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, edge_indices: List[torch.Tensor],
                edge_weights: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征 [num_nodes, input_dim]
            edge_indices: 边索引列表，每个组学一个 [2, num_edges]
            edge_weights: 边权重列表（可选），每个组学一个 [num_edges]
            
        Returns:
            logits: 分类logits [num_nodes, num_classes]
        """
        omics_features = []

        for i in range(self.num_omics):
            edge_index = edge_indices[i]
            
            # 安全获取边权重
            edge_weight = None
            if edge_weights is not None and i < len(edge_weights):
                edge_weight = edge_weights[i]

            h = x
            for layer_idx, gcn_layer in enumerate(self.gcn_list[i]):
                h = gcn_layer(h, edge_index, edge_weight=edge_weight)
                if layer_idx < self.gcn_layers - 1:  # 最后一层不加ReLU
                    h = F.relu(h)

            omics_features.append(h)

        # ✨ 使用节点级注意力融合（每个节点独立计算组学权重）
        fusion_feature = self.fusion_attention(omics_features)

        # 拼接所有组学特征和融合特征：[组学1, 组学2, 组学3, 融合]
        final_features = torch.cat(omics_features + [fusion_feature], dim=1)
        logits = self.classifier(final_features)

        return logits
    
    def get_omics_attention_weights(self, x: torch.Tensor, edge_indices: List[torch.Tensor],
                                    edge_weights: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """获取每个节点对各组学的注意力权重（用于模型解释）
        
        Returns:
            attention_weights: [num_nodes, num_omics]
        """
        omics_features = []
        
        for i in range(self.num_omics):
            edge_index = edge_indices[i]
            edge_weight = None
            if edge_weights is not None and i < len(edge_weights):
                edge_weight = edge_weights[i]

            h = x
            for layer_idx, gcn_layer in enumerate(self.gcn_list[i]):
                h = gcn_layer(h, edge_index, edge_weight=edge_weight)
                if layer_idx < self.gcn_layers - 1:
                    h = F.relu(h)

            omics_features.append(h)
        
        return self.fusion_attention.get_attention_weights(omics_features)


class GCNContrastiveModel(nn.Module):
    """带对比学习的多组学GCN模型
    
    在基础GCN上增加对比学习模块，通过对比增强前后的特征来学习更鲁棒的表示。
    使用Attention融合多个组学特征，最后拼接所有组学特征和融合特征进行分类。
    支持边权重。
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_omics: int,
                 num_classes: int, gcn_layers: int = 2,
                 embedding_dim: int = 128, temperature: float = 0.5,
                 fusion_num_heads: int = 4, fusion_dropout: float = 0.3):
        """初始化对比学习GCN模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_omics: 组学数量
            num_classes: 类别数量
            gcn_layers: GCN层数
            embedding_dim: 对比学习嵌入维度
            temperature: 对比学习温度参数
            fusion_num_heads: 融合模块的注意力头数（新增）
            fusion_dropout: 融合模块的dropout率（新增）
        """
        super().__init__()
        self.num_omics = num_omics
        self.gcn_layers = gcn_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # 为每个组学创建独立的GCN层
        self.gcn_list = nn.ModuleList()
        for _ in range(num_omics):
            layers = nn.ModuleList()
            for layer_idx in range(gcn_layers):
                if layer_idx == 0:
                    layers.append(GCNConv(input_dim, hidden_dim))
                else:
                    layers.append(GCNConv(hidden_dim, hidden_dim))
            self.gcn_list.append(layers)

        # ✨ 改进：使用节点级动态注意力融合
        self.fusion_attention = AttentionFusion(
            num_omics=num_omics,
            hidden_dim=hidden_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout
        )

        # 投影头（用于对比学习，输入包含融合特征）
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim * (num_omics + 1), embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 分类器（输入维度为 num_omics 个组学特征 + 1 个融合特征）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (num_omics + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, edge_indices: List[torch.Tensor],
                edge_weights: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """前向传播（用于分类）
        
        Args:
            x: 输入特征 [num_nodes, input_dim]
            edge_indices: 边索引列表
            edge_weights: 边权重列表（可选）
            
        Returns:
            logits: 分类logits [num_nodes, num_classes]
        """
        omics_features = []

        for i in range(self.num_omics):
            edge_index = edge_indices[i]
            
            # 安全获取边权重
            edge_weight = None
            if edge_weights is not None and i < len(edge_weights):
                edge_weight = edge_weights[i]

            h = x
            for layer_idx, gcn_layer in enumerate(self.gcn_list[i]):
                h = gcn_layer(h, edge_index, edge_weight=edge_weight)
                if layer_idx < self.gcn_layers - 1:
                    h = F.relu(h)

            omics_features.append(h)

        # ✨ 使用节点级注意力融合
        fusion_feature = self.fusion_attention(omics_features)

        # 拼接所有组学特征和融合特征：[组学1, 组学2, 组学3, 融合]
        final_features = torch.cat(omics_features + [fusion_feature], dim=1)
        logits = self.classifier(final_features)

        return logits

    def get_embeddings(self, x: torch.Tensor, edge_indices: List[torch.Tensor],
                       edge_weights: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """获取对比学习嵌入
        
        Args:
            x: 输入特征
            edge_indices: 边索引列表
            edge_weights: 边权重列表（可选）
            
        Returns:
            embeddings: 归一化的嵌入向量 [num_nodes, embedding_dim]
        """
        omics_features = []

        for i in range(self.num_omics):
            edge_index = edge_indices[i]
            
            # 安全获取边权重
            edge_weight = None
            if edge_weights is not None and i < len(edge_weights):
                edge_weight = edge_weights[i]

            h = x
            for layer_idx, gcn_layer in enumerate(self.gcn_list[i]):
                h = gcn_layer(h, edge_index, edge_weight=edge_weight)
                if layer_idx < self.gcn_layers - 1:
                    h = F.relu(h)

            omics_features.append(h)

        # Attention融合得到融合特征
        fusion_feature = self.fusion_attention(omics_features)

        # 拼接所有组学特征和融合特征
        final_features = torch.cat(omics_features + [fusion_feature], dim=1)
        
        # 投影到嵌入空间
        embeddings = self.projection_head(final_features)

        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def compute_contrastive_loss(self, x: torch.Tensor, x_aug: torch.Tensor,
                                edge_indices: List[torch.Tensor],
                                edge_indices_aug: List[torch.Tensor],
                                edge_weights: Optional[List[torch.Tensor]] = None,
                                edge_weights_aug: Optional[List[torch.Tensor]] = None,
                                labels: Optional[torch.Tensor] = None,
                                train_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算对比学习损失
        
        使用归一化温度缩放的余弦相似度 (NT-Xent) 损失。
        
        Args:
            x: 原始特征
            x_aug: 增强特征
            edge_indices: 原始边索引列表
            edge_indices_aug: 增强边索引列表
            edge_weights: 原始边权重列表（可选）
            edge_weights_aug: 增强边权重列表（可选）
            labels: 标签（可选，用于类内对比）
            train_mask: 训练掩码（可选，只计算训练节点的损失）
            
        Returns:
            contrastive_loss: 对比学习损失值
        """
        # 获取原始和增强的嵌入
        z1 = self.get_embeddings(x, edge_indices, edge_weights)
        z2 = self.get_embeddings(x_aug, edge_indices_aug, edge_weights_aug)

        # 如果提供了train_mask，只使用训练节点
        if train_mask is not None:
            z1 = z1[train_mask]
            z2 = z2[train_mask]
            if labels is not None:
                labels = labels[train_mask]

        batch_size = z1.shape[0]

        # 计算相似度矩阵
        # z1和z2已经归一化，所以直接矩阵乘法就是余弦相似度
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature

        # 创建标签：对角线为正样本对
        contrastive_labels = torch.arange(batch_size, device=z1.device)

        # 计算对比损失（双向）
        loss_1_to_2 = F.cross_entropy(similarity_matrix, contrastive_labels)
        loss_2_to_1 = F.cross_entropy(similarity_matrix.T, contrastive_labels)

        contrastive_loss = (loss_1_to_2 + loss_2_to_1) / 2

        return contrastive_loss
    
    def get_omics_attention_weights(self, x: torch.Tensor, edge_indices: List[torch.Tensor],
                                    edge_weights: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """获取每个节点对各组学的注意力权重（用于模型解释）
        
        Returns:
            attention_weights: [num_nodes, num_omics]
        """
        omics_features = []
        
        for i in range(self.num_omics):
            edge_index = edge_indices[i]
            edge_weight = None
            if edge_weights is not None and i < len(edge_weights):
                edge_weight = edge_weights[i]

            h = x
            for layer_idx, gcn_layer in enumerate(self.gcn_list[i]):
                h = gcn_layer(h, edge_index, edge_weight=edge_weight)
                if layer_idx < self.gcn_layers - 1:
                    h = F.relu(h)

            omics_features.append(h)
        
        return self.fusion_attention.get_attention_weights(omics_features)


class ModelFactory:
    """模型工厂类，用于创建不同类型的模型"""

    @staticmethod
    def create_model(model_type: str, **kwargs):
        """创建模型实例
        
        Args:
            model_type: 模型类型 ('gcn' 或 'gcn_contrastive')
            **kwargs: 模型参数
            
        Returns:
            model: 创建的模型实例
        """
        # 提取融合模块参数（带默认值）
        fusion_num_heads = kwargs.pop('fusion_num_heads', 4)
        fusion_dropout = kwargs.pop('fusion_dropout', 0.3)
        
        if model_type == 'gcn':
            return MultiOmicsGCN(
                input_dim=kwargs['input_dim'],
                hidden_dim=kwargs['hidden_dim'],
                num_omics=kwargs['num_omics'],
                num_classes=kwargs['num_classes'],
                gcn_layers=kwargs.get('gcn_layers', 2),
                fusion_num_heads=fusion_num_heads,
                fusion_dropout=fusion_dropout
            )
        elif model_type == 'gcn_contrastive':
            return GCNContrastiveModel(
                input_dim=kwargs['input_dim'],
                hidden_dim=kwargs['hidden_dim'],
                num_omics=kwargs['num_omics'],
                num_classes=kwargs['num_classes'],
                gcn_layers=kwargs.get('gcn_layers', 2),
                embedding_dim=kwargs.get('embedding_dim', 128),
                temperature=kwargs.get('temperature', 0.5),
                fusion_num_heads=fusion_num_heads,
                fusion_dropout=fusion_dropout
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
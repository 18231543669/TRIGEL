"""
图融合模块 - 节点级动态注意力融合
参考: MOGAT (2024), MOGONET (2021), CrossAttOmics (2025)

改进点：
1. 从3个参数 → ~600+参数，大幅提升表达能力
2. 每个节点独立计算组学权重（原来所有节点共享）
3. 支持多头注意力机制
4. 可学习温度参数
5. 残差连接 + LayerNorm 提升训练稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class NodeLevelAttentionFusion(nn.Module):
    """节点级动态注意力融合
    
    为每个节点独立计算组学权重，而不是全局共享权重。
    支持多头注意力和可学习温度参数。
    
    参数量估算（hidden_dim=64, num_omics=3, num_heads=4）:
    - query_proj: 64*64 = 4096
    - key_proj: 64*64 = 4096
    - omics_transforms: 3 * 64*64 = 12288
    - output_proj: 64*64 = 4096
    - 总计: ~24000+ 参数（vs 原来的3个参数）
    """
    
    def __init__(self, num_omics: int, hidden_dim: int, 
                 num_heads: int = 4, dropout: float = 0.3):
        """
        Args:
            num_omics: 组学数量
            hidden_dim: 每个组学的嵌入维度
            num_heads: 注意力头数（推荐4或8）
            dropout: 注意力dropout率
        """
        super().__init__()
        self.num_omics = num_omics
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 每个头的维度
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim({hidden_dim})必须能被num_heads({num_heads})整除"
        
        # Query, Key 投影层（用于计算注意力分数）
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 组学特定的变换层（增强表达能力，捕获组学特异性）
        self.omics_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_omics)
        ])
        
        # 可学习的温度参数（控制注意力分布的锐度）
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 融合后的投影层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization（提升训练稳定性）
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化权重"""
        for module in [self.query_proj, self.key_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        
        for transform in self.omics_transforms:
            nn.init.xavier_uniform_(transform.weight)
            nn.init.zeros_(transform.bias)
    
    def forward(self, omics_features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播 - 节点级动态融合
        
        Args:
            omics_features: 组学特征列表，每个 [num_nodes, hidden_dim]
            
        Returns:
            fused_feature: 融合后的特征 [num_nodes, hidden_dim]
        """
        num_nodes = omics_features[0].shape[0]
        device = omics_features[0].device
        
        # 1. 对每个组学应用特定变换（捕获组学特异性模式）
        transformed_features = []
        for i, feat in enumerate(omics_features):
            transformed = self.omics_transforms[i](feat)
            transformed_features.append(transformed)
        
        # 2. 堆叠所有组学特征: [num_nodes, num_omics, hidden_dim]
        stacked = torch.stack(transformed_features, dim=1)
        
        # 3. 计算全局查询向量（所有组学特征的均值作为查询基准）
        global_query = stacked.mean(dim=1)  # [num_nodes, hidden_dim]
        
        # 4. 计算Query和Key
        Q = self.query_proj(global_query)  # [num_nodes, hidden_dim]
        K = self.key_proj(stacked.view(-1, self.hidden_dim))  # [num_nodes * num_omics, hidden_dim]
        K = K.view(num_nodes, self.num_omics, self.hidden_dim)  # [num_nodes, num_omics, hidden_dim]
        
        # 5. 多头注意力计算
        # 重塑为多头格式
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)  # [N, heads, head_dim]
        K = K.view(num_nodes, self.num_omics, self.num_heads, self.head_dim)  # [N, omics, heads, head_dim]
        K = K.permute(0, 2, 1, 3)  # [N, heads, omics, head_dim]
        
        # 计算注意力分数
        # Q: [N, heads, head_dim] -> [N, heads, 1, head_dim]
        Q = Q.unsqueeze(2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [N, heads, 1, omics]
        attn_scores = attn_scores.squeeze(2)  # [N, heads, omics]
        
        # 温度缩放（防止数值不稳定，最小值0.1）
        temperature = torch.clamp(self.temperature, min=0.1)
        attn_scores = attn_scores / (temperature * (self.head_dim ** 0.5))
        
        # Softmax得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, heads, omics]
        attn_weights = self.dropout(attn_weights)
        
        # 6. 加权融合
        # 对多头取平均得到最终权重
        final_weights = attn_weights.mean(dim=1)  # [N, omics]
        
        # 应用权重进行加权求和
        # stacked: [N, omics, hidden_dim]
        # final_weights: [N, omics] -> [N, omics, 1]
        weighted = stacked * final_weights.unsqueeze(-1)
        fused = weighted.sum(dim=1)  # [N, hidden_dim]
        
        # 7. 输出投影 + 残差连接 + LayerNorm
        output = self.output_proj(fused)
        output = self.layer_norm(output + global_query)  # 残差连接提升梯度流动
        
        return output
    
    def get_attention_weights(self, omics_features: List[torch.Tensor]) -> torch.Tensor:
        """
        获取注意力权重（用于可视化和模型解释）
        
        Args:
            omics_features: 组学特征列表
            
        Returns:
            attention_weights: [num_nodes, num_omics] 每个节点对每个组学的注意力权重
        """
        with torch.no_grad():
            num_nodes = omics_features[0].shape[0]
            
            transformed_features = []
            for i, feat in enumerate(omics_features):
                transformed = self.omics_transforms[i](feat)
                transformed_features.append(transformed)
            
            stacked = torch.stack(transformed_features, dim=1)
            global_query = stacked.mean(dim=1)
            
            Q = self.query_proj(global_query)
            K = self.key_proj(stacked.view(-1, self.hidden_dim))
            K = K.view(num_nodes, self.num_omics, self.hidden_dim)
            
            Q = Q.view(num_nodes, self.num_heads, self.head_dim).unsqueeze(2)
            K = K.view(num_nodes, self.num_omics, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)).squeeze(2)
            temperature = torch.clamp(self.temperature, min=0.1)
            attn_scores = attn_scores / (temperature * (self.head_dim ** 0.5))
            attn_weights = F.softmax(attn_scores, dim=-1).mean(dim=1)
            
            return attn_weights


class SimpleAttentionFusion(nn.Module):
    """简化版节点级注意力融合（参数量较少，适合小数据集）
    
    相比NodeLevelAttentionFusion:
    - 无多头机制
    - 无残差连接
    - 参数量约为1/4
    """
    
    def __init__(self, num_omics: int, hidden_dim: int, dropout: float = 0.3):
        """
        Args:
            num_omics: 组学数量
            hidden_dim: 嵌入维度
            dropout: dropout率
        """
        super().__init__()
        self.num_omics = num_omics
        self.hidden_dim = hidden_dim
        
        # 注意力计算层
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim * num_omics, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_omics)
        )
        
        # 可学习温度
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, omics_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            omics_features: 组学特征列表，每个 [num_nodes, hidden_dim]
            
        Returns:
            fused_feature: [num_nodes, hidden_dim]
        """
        # 拼接所有组学特征
        concat = torch.cat(omics_features, dim=1)  # [N, hidden_dim * num_omics]
        
        # 计算每个节点的注意力权重
        attn_logits = self.attention_fc(concat)  # [N, num_omics]
        temperature = torch.clamp(self.temperature, min=0.1)
        attn_weights = F.softmax(attn_logits / temperature, dim=1)  # [N, num_omics]
        
        # 堆叠并加权融合
        stacked = torch.stack(omics_features, dim=2)  # [N, hidden_dim, num_omics]
        fused = (stacked * attn_weights.unsqueeze(1)).sum(dim=2)  # [N, hidden_dim]
        
        return fused
    
    def get_attention_weights(self, omics_features: List[torch.Tensor]) -> torch.Tensor:
        """获取注意力权重"""
        with torch.no_grad():
            concat = torch.cat(omics_features, dim=1)
            attn_logits = self.attention_fc(concat)
            temperature = torch.clamp(self.temperature, min=0.1)
            return F.softmax(attn_logits / temperature, dim=1)


# ==================== 向后兼容性 ====================

class AttentionFusion(NodeLevelAttentionFusion):
    """向后兼容的别名 - 使用节点级注意力融合
    
    替换原来的简单加权融合，保持接口兼容
    """
    def __init__(self, num_omics: int, hidden_dim: int = 64, 
                 num_heads: int = 4, dropout: float = 0.3):
        """
        Args:
            num_omics: 组学数量
            hidden_dim: 隐藏维度（默认64，与GCN模型保持一致）
            num_heads: 注意力头数
            dropout: dropout率
        """
        super().__init__(num_omics, hidden_dim, num_heads, dropout)


# ==================== 原始简单融合（保留用于对比实验） ====================

class LegacyAttentionFusion(nn.Module):
    """原始的简单加权融合（仅用于对比实验）
    
    警告：此实现仅有num_omics个参数，表达能力极其有限
    """
    
    def __init__(self, num_omics: int):
        super().__init__()
        self.num_omics = num_omics
        self.attention_weights = nn.Parameter(torch.ones(num_omics) / num_omics)
    
    def forward(self, omics_features: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.attention_weights, dim=0)
        fused = weights[0] * omics_features[0]
        for i in range(1, self.num_omics):
            fused = fused + weights[i] * omics_features[i]
        return fused
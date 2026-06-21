"""
数据处理工具模块
包含数据加载、预处理、图构建和分割等功能
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import time
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional


def set_random_seed(seed: Optional[int]) -> None:
    """设置随机种子"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def construct_adjacency(data: np.ndarray, k: int = 5) -> torch.Tensor:
    """使用余弦相似度的KNN方法构建邻接矩阵"""
    n_samples = data.shape[0]
    print(f"Using cosine similarity with KNN (k={k}) for graph construction...")
    start_time = time.time()

    # 高效计算余弦相似度
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # 避免除零错误
    normalized = data / norms
    cos_sim = normalized @ normalized.T  # 余弦相似度矩阵

    # 初始化邻接矩阵
    adj = np.zeros((n_samples, n_samples))

    # 设置对角线为1（自环）
    np.fill_diagonal(adj, 1)

    # 对每个节点，选择相似度最高的k个其他节点
    for i in range(n_samples):
        # 获取当前节点的相似度（排除自身）
        similarities = cos_sim[i].copy()
        similarities[i] = -np.inf  # 排除自身

        # 获取top k索引
        top_k_indices = np.argpartition(similarities, -k)[-k:]

        # 设置边（确保对称）
        adj[i, top_k_indices] = 1
        adj[top_k_indices, i] = 1

    print(f"Cosine KNN graph construction completed in {time.time() - start_time:.2f} seconds")
    print(f"Graph density: {np.mean(adj):.4f}")

    # 转换为COO格式并返回边索引
    adj_coo = coo_matrix(adj)
    edge_index = np.vstack([adj_coo.row, adj_coo.col])
    return torch.tensor(edge_index, dtype=torch.long)


def load_dataset(data_dir: str, use_preprocessing: bool = False,
                preprocessing_config: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, int, int, List, List, int]:
    """
    加载多组学数据集（✓ 集成预处理）

    Args:
        data_dir: 数据目录路径
        use_preprocessing: 是否使用预处理（强烈推荐True）
        preprocessing_config: 预处理配置字典
            - scaler_type: 'standard' 或 'robust'
            - variance_threshold: 方差阈值
            - clip_outliers: 是否裁剪异常值
            - outlier_std: 异常值定义
            
    Returns:
        X_full, y_full, total_features, num_omics, train_data, test_data, num_classes
    """
    print("\nLoading dataset...")
    # 获取组学数量
    omics_files = glob.glob(os.path.join(data_dir, '*_featname.csv'))
    num_omics = len(omics_files)
    print(f"Found {num_omics} omics datasets")

    # 加载标签
    labels_tr = pd.read_csv(os.path.join(data_dir, 'labels_tr.csv'), header=None).values.flatten()
    labels_te = pd.read_csv(os.path.join(data_dir, 'labels_te.csv'), header=None).values.flatten()

    # 初始化数据结构
    train_data, test_data = [], []
    total_features_before = 0

    # 加载每个组学数据
    for i in range(1, num_omics + 1):
        # 加载训练和测试数据
        tr_data = pd.read_csv(os.path.join(data_dir, f'{i}_tr.csv'), header=None).values
        te_data = pd.read_csv(os.path.join(data_dir, f'{i}_te.csv'), header=None).values

        train_data.append(tr_data)
        test_data.append(te_data)
        total_features_before += tr_data.shape[1]
        print(f"Omics {i}: {tr_data.shape[1]} features, {tr_data.shape[0]} train samples, {te_data.shape[0]} test samples")

    # ✓ 预处理（每个组学独立标准化）
    if use_preprocessing:
        from preprocessing import preprocess_multi_omics_data, RECOMMENDED_CONFIGS
        
        # 使用配置（默认为conservative）
        if preprocessing_config is None:
            preprocessing_config = RECOMMENDED_CONFIGS['conservative']
        
        train_data, test_data, preprocess_stats = preprocess_multi_omics_data(
            train_data, test_data, config=preprocessing_config
        )
    else:
        print("\n⚠️  警告: 跳过预处理！强烈建议启用预处理以获得更好的性能。")

    # 合并所有组学的特征
    X_train = np.concatenate(train_data, axis=1)
    X_test = np.concatenate(test_data, axis=1)

    # 合并所有样本
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([labels_tr, labels_te])
    
    # 更新总特征数
    total_features = X_train.shape[1]

    # 分析数据集
    unique_classes, class_counts = np.unique(y_full, return_counts=True)
    num_classes = len(unique_classes)

    print(f"\n✓ 数据加载完成")
    print(f"  总样本: {X_full.shape[0]} (训练: {len(labels_tr)}, 测试: {len(labels_te)})")
    print(f"  总特征: {total_features} {'(预处理后)' if use_preprocessing else '(原始)'}")
    if use_preprocessing:
        print(f"  特征变化: {total_features_before} → {total_features} (移除 {total_features_before - total_features})")
    print(f"  类别数: {num_classes}")
    print(f"  类别分布: {dict(zip(unique_classes, class_counts))}")

    return X_full, y_full, total_features, num_omics, train_data, test_data, num_classes


def stratified_split(indices: np.ndarray, y: np.ndarray, test_size: float = 0.5,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """分层抽样，保持类别比例"""
    indices = np.asarray(indices)
    y = np.asarray(y)

    unique_classes = np.unique(y)
    train_idx = []
    test_idx = []

    for cls in unique_classes:
        # 找到当前类别对应的索引
        cls_mask = y == cls
        cls_indices = indices[cls_mask]
        cls_tr, cls_te = train_test_split(
            cls_indices,
            test_size=test_size,
            random_state=random_state
        )
        train_idx.extend(cls_tr)
        test_idx.extend(cls_te)

    return np.array(train_idx), np.array(test_idx)
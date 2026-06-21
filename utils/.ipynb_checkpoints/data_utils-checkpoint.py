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


# 在 data_utils.py 顶部添加
from .graph_cache import GraphCacheManager

# 修改 construct_adjacency 函数
def construct_adjacency(data: np.ndarray, k: int = 5, use_cache: bool = True) -> torch.Tensor:
    """使用余弦相似度的KNN方法构建邻接矩阵（带缓存）"""
    n_samples = data.shape[0]
    
    # 如果启用缓存，先检查缓存
    if use_cache:
        cache_manager = GraphCacheManager()
        cached_result = cache_manager.get(data, k)
        if cached_result is not None:
            print(f"✓ 从缓存加载KNN图 (k={k})")
            return cached_result
    
    print(f"Using cosine similarity with KNN (k={k}) for graph construction...")
    start_time = time.time()

    # 原始的计算逻辑（保持不变）
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = data / norms
    cos_sim = normalized @ normalized.T

    adj = np.zeros((n_samples, n_samples))
    np.fill_diagonal(adj, 1)

    for i in range(n_samples):
        similarities = cos_sim[i].copy()
        similarities[i] = -np.inf
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        adj[i, top_k_indices] = 1
        adj[top_k_indices, i] = 1

    print(f"Cosine KNN graph construction completed in {time.time() - start_time:.2f} seconds")
    print(f"Graph density: {np.mean(adj):.4f}")

    adj_coo = coo_matrix(adj)
    edge_index = np.vstack([adj_coo.row, adj_coo.col])
    result = torch.tensor(edge_index, dtype=torch.long)
    
    # 保存到缓存
    if use_cache:
        cache_key = cache_manager.set(data, k, result)
        print(f"✓ KNN图已缓存 (key: {cache_key[:8]}...)")
    
    return result


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, int, int, List, List, int]:
    """
    加载多组学数据集

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
    total_features = 0

    # 加载每个组学数据
    for i in range(1, num_omics + 1):
        # 加载训练和测试数据
        tr_data = pd.read_csv(os.path.join(data_dir, f'{i}_tr.csv'), header=None).values
        te_data = pd.read_csv(os.path.join(data_dir, f'{i}_te.csv'), header=None).values

        train_data.append(tr_data)
        test_data.append(te_data)
        total_features += tr_data.shape[1]
        print(f"Omics {i}: {tr_data.shape[1]} features, {tr_data.shape[0]} train samples, {te_data.shape[0]} test samples")

    # 合并所有组学的特征
    X_train = np.concatenate(train_data, axis=1)
    X_test = np.concatenate(test_data, axis=1)

    # 合并所有样本
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([labels_tr, labels_te])

    # 分析数据集
    unique_classes, class_counts = np.unique(y_full, return_counts=True)
    num_classes = len(unique_classes)

    print(f"\nFull dataset: {X_full.shape[0]} samples, {total_features} features")
    print(f"Train labels: {len(labels_tr)}, Test labels: {len(labels_te)}")
    print(f"Number of classes: {num_classes}")
    print("Class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")

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
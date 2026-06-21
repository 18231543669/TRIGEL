"""
Step 2: 将 KRCCC.mat 转换为模型所需的 CSV 文件
运行方式: python step2_convert_mat.py KRCCC.mat --output_dir ./processed_data --k 5 --train_ratio 0.7

输出文件:
  1_tr.csv, 1_te.csv, 1_featname.csv   (组学1: 329维)
  2_tr.csv, 2_te.csv, 2_featname.csv   (组学2: 17899维)
  3_tr.csv, 3_te.csv, 3_featname.csv   (组学3: 24960维)
  adj1.csv, adj2.csv, adj3.csv          (每个组学的KNN邻接矩阵)
  labels_tr.csv, labels_te.csv          (标签, 从0开始)
"""

import argparse
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def build_knn_graph_cosine(X, k=5):
    """
    使用余弦相似度构建对称 KNN 邻接矩阵（与 data_utils.py 一致）
    """
    n_samples = X.shape[0]
    print(f"    构建余弦KNN图 (n={n_samples}, k={k}) ...")

    # 归一化
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = X / norms

    # 余弦相似度矩阵
    cos_sim = normalized @ normalized.T

    # 构建邻接矩阵
    adj = np.zeros((n_samples, n_samples), dtype=int)
    np.fill_diagonal(adj, 1)  # 自环

    for i in range(n_samples):
        similarities = cos_sim[i].copy()
        similarities[i] = -np.inf  # 排除自身
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        adj[i, top_k_indices] = 1
        adj[top_k_indices, i] = 1  # 对称化

    density = np.sum(adj) / (n_samples * n_samples)
    print(f"    图密度: {density:.4f}")
    return adj


def main():
    parser = argparse.ArgumentParser(description="MAT -> CSV 转换工具")
    parser.add_argument("mat_file", type=str, help=".mat 文件路径")
    parser.add_argument("--output_dir", type=str, default="./processed_data", help="输出目录")
    parser.add_argument("--k", type=int, default=5, help="KNN 近邻数")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # ========== 1. 加载 .mat ==========
    print(f"\n加载 {args.mat_file} ...")
    mat = sio.loadmat(args.mat_file)

    X_cell = mat['X'].flatten()  # shape (3,), 每个元素是 (features, samples)
    Y = mat['Y'].flatten()       # shape (122,)

    n_views = len(X_cell)
    n_samples = len(Y)
    print(f"组学数: {n_views}, 样本数: {n_samples}")

    # ========== 2. 标签处理：转为从 0 开始 ==========
    # 原始标签: 1,2,3,4 -> 转为 0,1,2,3
    Y_zero = Y - Y.min()
    unique, counts = np.unique(Y_zero, return_counts=True)
    print(f"\n标签分布 (转换后从0开始):")
    for cls, cnt in zip(unique, counts):
        print(f"  类别 {cls}: {cnt} 个样本")

    # ========== 3. 分层抽样划分训练/测试集 ==========
    print(f"\n分层抽样: 训练集 {args.train_ratio}, 测试集 {1 - args.train_ratio}")
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        train_size=args.train_ratio,
        stratify=Y_zero,
        random_state=args.seed
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    print(f"训练集: {len(train_idx)} 样本, 测试集: {len(test_idx)} 样本")

    # 验证分层效果
    print("训练集类别分布:")
    for cls in unique:
        cnt = np.sum(Y_zero[train_idx] == cls)
        print(f"  类别 {cls}: {cnt}")
    print("测试集类别分布:")
    for cls in unique:
        cnt = np.sum(Y_zero[test_idx] == cls)
        print(f"  类别 {cls}: {cnt}")

    # ========== 4. 保存标签 ==========
    pd.DataFrame(Y_zero[train_idx]).to_csv(
        os.path.join(args.output_dir, "labels_tr.csv"), header=False, index=False
    )
    pd.DataFrame(Y_zero[test_idx]).to_csv(
        os.path.join(args.output_dir, "labels_te.csv"), header=False, index=False
    )
    print("\n✓ labels_tr.csv, labels_te.csv 已保存")

    # ========== 5. 处理每个组学 ==========
    for i in range(n_views):
        print(f"\n{'='*50}")
        print(f"处理组学 {i+1}/{n_views}")

        # 注意: .mat cell array 中数据形状是 (features, samples)，需要转置
        raw = X_cell[i]
        if raw.shape[1] == n_samples:
            # shape (features, samples) -> (samples, features)
            data = raw.T
        elif raw.shape[0] == n_samples:
            # shape 已经是 (samples, features)
            data = raw
        else:
            raise ValueError(
                f"组学 {i+1} 的形状 {raw.shape} 与样本数 {n_samples} 不匹配！"
            )

        n_feat = data.shape[1]
        print(f"  形状: ({n_samples}, {n_feat})")

        # --- 5a. 保存特征名 ---
        feat_names = [f"omics{i+1}_feat{j}" for j in range(n_feat)]
        pd.DataFrame(feat_names).to_csv(
            os.path.join(args.output_dir, f"{i+1}_featname.csv"),
            header=False, index=False
        )

        # --- 5b. 划分并保存训练/测试数据 ---
        train_data = data[train_idx]
        test_data = data[test_idx]

        pd.DataFrame(train_data).to_csv(
            os.path.join(args.output_dir, f"{i+1}_tr.csv"),
            header=False, index=False
        )
        pd.DataFrame(test_data).to_csv(
            os.path.join(args.output_dir, f"{i+1}_te.csv"),
            header=False, index=False
        )
        print(f"  ✓ {i+1}_tr.csv ({train_data.shape}), {i+1}_te.csv ({test_data.shape})")

        # --- 5c. 用全部样本构建 KNN 邻接矩阵 ---
        adj = build_knn_graph_cosine(data, k=args.k)
        pd.DataFrame(adj).to_csv(
            os.path.join(args.output_dir, f"adj{i+1}.csv"),
            header=False, index=False
        )
        print(f"  ✓ adj{i+1}.csv ({adj.shape})")

    # ========== 6. 总结 ==========
    print(f"\n{'='*50}")
    print(f"全部完成！输出目录: {args.output_dir}")
    print(f"生成文件:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f:25s} {size:>10.1f} KB")


if __name__ == "__main__":
    main()

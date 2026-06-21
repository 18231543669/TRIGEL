import os
import os.path as osp
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ==================== 配置参数 ====================
path = './'               # 特征矩阵所在目录
n_views = 3                    # 视图数量
K = 10                         # 近邻个数
feature_files = [f'feature_{i+1}.txt' for i in range(n_views)]  # 特征文件名为 view0.txt, view1.txt, view2.txt
# 如果实际分隔符不是空白，请修改下面的读取参数（例如 sep=','）
sep = None                     # 使用 delim_whitespace=True 时，sep 无需指定

# ==================== KNN 图构建函数 ====================
def build_knn_graph(X, k=10):
    """
    构建对称 KNN 邻接矩阵（二值）
    参数:
        X: 特征矩阵，形状 (n_samples, n_features)
        k: 近邻个数
    返回:
        adj: 邻接矩阵，形状 (n_samples, n_samples)，数据类型 int
    """
    n_samples = X.shape[0]
    # 使用欧氏距离，设置 n_neighbors = k+1 以包含自身
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # 初始化邻接矩阵
    adj = np.zeros((n_samples, n_samples), dtype=int)

    # 为每个样本标记其 k 个最近邻（排除自身）
    for i in range(n_samples):
        neighbors = indices[i, 1:]   # 跳过自身（索引0）
        adj[i, neighbors] = 1

    # 对称化：如果 i 是 j 的近邻 或 j 是 i 的近邻，则边存在
    adj = np.maximum(adj, adj.T)
    return adj

# ==================== 主处理循环 ====================
for i, fname in enumerate(feature_files):
    file_path = osp.join(path, fname)
    print(f"正在处理视图 {i}，读取文件：{file_path}")

    # 读取 TXT 特征矩阵（空白分隔，无列名）
    # 使用 delim_whitespace=True 可自动处理任意空白字符（空格/制表符）
    X = pd.read_csv(file_path, header=None, delim_whitespace=True).values

    # 可选：对特征进行标准化（根据需求取消注释）
    # from sklearn.preprocessing import StandardScaler
    # X = StandardScaler().fit_transform(X)

    # 构建 KNN 邻接矩阵
    adj = build_knn_graph(X, k=K)

    # 保存为 CSV（无表头、无索引）
    out_path = osp.join(path, f'adj{i+1}.csv')
    pd.DataFrame(adj).to_csv(out_path, header=False, index=False)
    print(f"邻接矩阵已保存至：{out_path}")

print("所有视图处理完成！")
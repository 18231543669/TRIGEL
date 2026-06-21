"""
离线生成邻接矩阵文件（adj1.csv, adj2.csv, ...）。

用法示例：
python generate_adjacency.py --data_dir ./datasets/KRCCC --method mogonet_threshold --k 5
python generate_adjacency.py --data_dir ./datasets/KRCCC --output_dir ./datasets/KRCCC --method knn_binary --k 5
"""

import argparse
import os
import numpy as np

from utils.data_utils import load_dataset, construct_adjacency, save_adjacency_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线生成adj邻接矩阵")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录（含 *_tr.csv, *_te.csv）")
    parser.add_argument("--output_dir", type=str, default="", help="adj输出目录，默认同 data_dir")
    parser.add_argument(
        "--method",
        type=str,
        default="mogonet_threshold",
        choices=["knn_binary", "mogonet_threshold"],
        help="构图方式"
    )
    parser.add_argument("--k", type=int, default=5, help="构图超参数k")
    parser.add_argument("--no_cache", action="store_true", help="禁用图缓存")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载数据: {args.data_dir}")
    X_full, y_full, _total_features, num_omics, train_data, test_data, _num_classes = load_dataset(args.data_dir)
    num_nodes = X_full.shape[0]
    print(f"样本数: {num_nodes}, 组学数: {num_omics}")
    print(f"开始生成adj (method={args.method}, k={args.k})")

    for i in range(num_omics):
        omics_data = train_data[i]
        omics_test = test_data[i]
        omics_full = np.vstack([omics_data, omics_test])

        edge_index = construct_adjacency(
            omics_full,
            k=args.k,
            use_cache=not args.no_cache,
            method=args.method
        )

        save_path = os.path.join(output_dir, f"adj{i + 1}.csv")
        save_adjacency_csv(edge_index, num_nodes=num_nodes, save_path=save_path)
        print(f"  已生成: {save_path}")

    print("完成：模型后续将直接读取adj文件，不再在线构图。")


if __name__ == "__main__":
    main()

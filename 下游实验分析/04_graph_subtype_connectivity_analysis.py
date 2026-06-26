#!/usr/bin/env python
"""样本相似图中的亚型连接结构分析。

这个脚本读取 ``adj1.csv``、``adj2.csv``、``adj3.csv`` 和样本标签，分析每个
组学图中的连边是否更倾向于连接同一亚型样本。

图的含义：
    1. subtype edge counts：不同亚型之间的边数量；
    2. subtype edge density：考虑类别大小后的边密度；
    3. intra-subtype edge ratio：所有边中同亚型连边所占比例。

下游分析意义：
    图神经网络依赖样本图传播信息。如果原始图中同亚型连边比例较高，说明图结构
    与生物亚型有一定一致性；如果后续做“图增强前后”对比，则同亚型连边比例提升
    可以直接支持图增强策略的有效性。

注意：
    当前脚本分析的是已有原始邻接图。若要证明 TRIGEL 的图增强创新点，应进一步
    对比增强前和增强后的邻接图，尤其关注少数类样本的同类连边比例。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: matplotlib.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BRCA"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "04_graph_subtype_connectivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze subtype aggregation in graphs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--graphs", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="使用边权重；若不指定，则把所有非零邻接项二值化为边。",
    )
    parser.add_argument(
        "--subtype-names",
        nargs="*",
        default=None,
        help="按数字标签顺序提供亚型名称。",
    )
    return parser.parse_args()


def create_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_labels(data_dir: Path, labels_path: Path | None) -> np.ndarray:
    if labels_path is not None:
        return pd.read_csv(labels_path, header=None).iloc[:, 0].to_numpy()
    train = pd.read_csv(data_dir / "labels_tr.csv", header=None).iloc[:, 0]
    test = pd.read_csv(data_dir / "labels_te.csv", header=None).iloc[:, 0]
    return pd.concat([train, test], ignore_index=True).to_numpy()


def subtype_display(value: object, subtype_names: list[str] | None) -> str:
    idx = int(float(value))
    if subtype_names and 0 <= idx < len(subtype_names):
        return subtype_names[idx]
    return f"Subtype {idx}"


def load_adjacency(data_dir: Path, graph_id: int, weighted: bool) -> np.ndarray:
    """读取某个组学的邻接矩阵。

    默认把非零项看作存在边；如果指定 --weighted，则保留原始边权重。
    对角线置零，避免样本自己连自己影响统计。
    """
    adj = pd.read_csv(data_dir / f"adj{graph_id}.csv", header=None).to_numpy(dtype=float)
    np.fill_diagonal(adj, 0.0)
    if not weighted:
        adj = (adj != 0).astype(float)
    return adj


def edge_count_matrix(
    adj: np.ndarray, labels: np.ndarray, subtype_names: list[str] | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """统计不同亚型之间的边数量和理论可连接样本对数量。

    counts 表示实际边数量；possible 表示两个亚型之间最多可能有多少样本对。
    用 counts / possible 可以得到边密度，避免大类因为样本多而天然有更多边。
    """
    classes = np.unique(labels)
    names = [subtype_display(cls, subtype_names) for cls in classes]
    counts = pd.DataFrame(0.0, index=names, columns=names)
    possible = pd.DataFrame(0.0, index=names, columns=names)

    upper = np.triu_indices_from(adj, k=1)
    edge_values = adj[upper]
    nonzero = edge_values != 0
    i_idx = upper[0][nonzero]
    j_idx = upper[1][nonzero]
    values = edge_values[nonzero]
    class_to_name = {cls: subtype_display(cls, subtype_names) for cls in classes}

    for i, j, value in zip(i_idx, j_idx, values):
        a = class_to_name[labels[i]]
        b = class_to_name[labels[j]]
        counts.loc[a, b] += value
        if a != b:
            counts.loc[b, a] += value

    for i, cls_i in enumerate(classes):
        name_i = class_to_name[cls_i]
        n_i = int(np.sum(labels == cls_i))
        for cls_j in classes[i:]:
            name_j = class_to_name[cls_j]
            n_j = int(np.sum(labels == cls_j))
            if cls_i == cls_j:
                total = n_i * (n_i - 1) / 2
            else:
                total = n_i * n_j
            possible.loc[name_i, name_j] = total
            possible.loc[name_j, name_i] = total
    return counts, possible


def plot_matrix(matrix: pd.DataFrame, title: str, output_stem: Path) -> None:
    """绘制亚型-亚型边数量或边密度矩阵。"""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(matrix.to_numpy(), cmap="magma", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            text = f"{value:.2f}" if value < 10 else f"{value:.0f}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def graph_summary(
    graph_name: str, adj: np.ndarray, labels: np.ndarray, counts: pd.DataFrame, possible: pd.DataFrame
) -> tuple[dict[str, float], pd.DataFrame]:
    """汇总单个图的连边结构。

    intra_subtype_edge_ratio 是核心指标：值越高，说明图中更多边连接同一亚型样本。
    若用于图增强分析，增强后该值上升通常表示图结构更符合亚型关系。
    """
    upper = np.triu_indices_from(adj, k=1)
    values = adj[upper]
    nonzero = values != 0
    same = labels[upper[0]] == labels[upper[1]]
    total_edge_weight = float(values[nonzero].sum())
    intra_edge_weight = float(values[nonzero & same].sum())
    summary = {
        "graph": graph_name,
        "num_nodes": int(adj.shape[0]),
        "num_edges": int(nonzero.sum()),
        "total_edge_weight": total_edge_weight,
        "intra_subtype_edge_weight": intra_edge_weight,
        "intra_subtype_edge_ratio": intra_edge_weight / (total_edge_weight + 1e-8),
        "graph_density": float(nonzero.sum() / (len(values) + 1e-8)),
    }
    density = counts / possible.replace(0, np.nan)
    density = density.fillna(0.0)
    return summary, density


def plot_ratio_bar(summary: pd.DataFrame, output_dir: Path) -> None:
    """绘制每个组学图的同亚型连边比例柱状图。"""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    bars = ax.bar(
        summary["graph"],
        summary["intra_subtype_edge_ratio"],
        width=0.38,
        color="#3B6EA8",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Intra-subtype edge ratio")
    ax.set_title("Subtype aggregation in sample graphs")
    ax.grid(axis="y", alpha=0.20, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"intra_subtype_edge_ratio.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    """运行所有指定组学图的亚型连接结构分析。"""
    args = parse_args()
    output_dir = create_run_dir(args.output_dir)
    labels = load_labels(args.data_dir, args.labels)
    summary_rows = []

    for graph_id in args.graphs:
        graph_name = f"Graph {graph_id}"
        adj = load_adjacency(args.data_dir, graph_id, args.weighted)
        if adj.shape[0] != len(labels) or adj.shape[1] != len(labels):
            raise ValueError(
                f"adj{graph_id}.csv shape {adj.shape} does not match "
                f"{len(labels)} labels."
            )
        counts, possible = edge_count_matrix(adj, labels, args.subtype_names)
        summary, density = graph_summary(graph_name, adj, labels, counts, possible)
        summary_rows.append(summary)

        counts.to_csv(output_dir / f"graph{graph_id}_subtype_edge_counts.csv")
        density.to_csv(output_dir / f"graph{graph_id}_subtype_edge_density.csv")
        plot_matrix(
            counts,
            f"{graph_name}: subtype-subtype edge counts",
            output_dir / f"graph{graph_id}_subtype_edge_counts",
        )
        plot_matrix(
            density,
            f"{graph_name}: subtype-subtype edge density",
            output_dir / f"graph{graph_id}_subtype_edge_density",
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "graph_subtype_connectivity_summary.csv", index=False)
    plot_ratio_bar(summary, output_dir)
    print(f"Graph subtype connectivity analysis completed: {output_dir}")


if __name__ == "__main__":
    main()

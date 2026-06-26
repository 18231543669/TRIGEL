#!/usr/bin/env python
"""比较单组学、简单拼接和 TRIGEL 的表示空间。

这张图不是为了单纯展示 PCA 散点图，而是服务于“不平衡数据”这一创新点：
在类别数量差异较大的 BRCA 亚型任务中，少数类样本容易被多数类样本的邻域结构
淹没。如果 TRIGEL 的不平衡图表示学习是有效的，那么它的表示空间应该满足：

1. 少数类样本的近邻中，同类样本比例更高；
2. 全部样本的同类近邻比例更高；
3. 不同亚型之间的整体分离度更好。

因此，本脚本上排画不同表示空间的二维 PCA 可视化；下排用定量指标验证这种
可视化观察是否可靠。定量指标在原始高维表示空间中计算，而不是在二维 PCA
坐标上计算，避免“图看起来分开但高维结构并不好”的问题。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc.name}.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BRCA"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "08_representation_space_comparison"
DEFAULT_EARLY_FUSION = DEFAULT_DATA_DIR / "BRCA875_fixed_early_fusion.npy"
DEFAULT_TRIGEL_EMBEDDING = DEFAULT_DATA_DIR / "BRCA875_TRIGEL_embeddings.npy"

# 标签序号与亚型名称的对应关系：
#   0 -> Normal-like
#   1 -> Basal-like
#   2 -> HER2-enriched
#   3 -> Luminal A
#   4 -> Luminal B
#
# 注意：这个映射不是直接从原始临床注释表读取的，而是根据当前数据中的类别数量
# 和已知亚型相关 marker/signature 审计后得到的更合理对应关系。若后续拿到原始
# PAM50 临床注释表，应优先以原始注释表为准。
SUBTYPE_NAMES = ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]
SUBTYPE_COLORS = ["#8E6C8A", "#009E73", "#D55E00", "#4C78A8", "#72B7B2"]

# 当前映射下，label=2 是样本量最少的 HER2-enriched；label=3 是样本量最多的
# Luminal A。下面的少数类近邻指标专门观察 HER2-enriched 是否被多数类邻域淹没。
MINORITY_LABEL = 2
MAJORITY_LABEL = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw representation-space comparison for imbalance analysis."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--early-fusion", type=Path, default=DEFAULT_EARLY_FUSION)
    parser.add_argument("--trigel-embedding", type=Path, default=DEFAULT_TRIGEL_EMBEDDING)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=2026)
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


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def load_labels(data_dir: Path) -> np.ndarray:
    train = pd.read_csv(data_dir / "labels_tr.csv", header=None).iloc[:, 0]
    test = pd.read_csv(data_dir / "labels_te.csv", header=None).iloc[:, 0]
    return pd.concat([train, test], ignore_index=True).to_numpy(dtype=int)


def load_omics(data_dir: Path, omics_id: int) -> np.ndarray:
    train = pd.read_csv(data_dir / f"{omics_id}_tr.csv", header=None)
    test = pd.read_csv(data_dir / f"{omics_id}_te.csv", header=None)
    return pd.concat([train, test], ignore_index=True).to_numpy(dtype=float)


def robust_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    return StandardScaler().fit_transform(x)


def projection_2d(x: np.ndarray, seed: int) -> np.ndarray:
    """用 PCA 得到二维坐标，仅用于可视化展示。

    PCA 坐标不参与最终指标计算。论文/报告里解释时应把上排散点图作为“直观展示”，
    把下排柱状图作为主要证据。
    """
    scaled = robust_scale(x)
    return PCA(n_components=2, random_state=seed).fit_transform(scaled)


def knn_metrics(x: np.ndarray, labels: np.ndarray, k: int) -> dict[str, float]:
    """计算近邻结构指标。

    overall_neighbor_purity：
        对所有样本计算 k 个近邻中同亚型样本的比例，反映整体亚型邻域结构。

    minority_neighbor_purity：
        只看 HER2-enriched 少数类样本，计算其 k 个近邻中同类样本比例。
        这个指标最贴合“不平衡”创新点：如果少数类被多数类淹没，该值会偏低；
        如果 TRIGEL 保住了少数类局部结构，该值应明显升高。

    minority_to_majority_neighbor_ratio：
        只看 HER2-enriched 少数类样本，计算其近邻中多数类 Luminal A 的比例。
        这个指标越低越好，表示少数类不再大量靠近多数类样本。
    """
    scaled = robust_scale(x)
    # 多组学特征维度高、尺度差异大；标准化后用 cosine 距离衡量邻域更稳定。
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(scaled)
    neighbors = nn.kneighbors(scaled, return_distance=False)[:, 1:]
    same = labels[neighbors] == labels[:, None]

    minority = labels == MINORITY_LABEL
    majority_neighbors = labels[neighbors] == MAJORITY_LABEL
    return {
        "overall_neighbor_purity": float(np.mean(same)),
        "minority_neighbor_purity": float(np.mean(same[minority])),
        "minority_to_majority_neighbor_ratio": float(np.mean(majority_neighbors[minority])),
    }


def representation_metrics(x: np.ndarray, labels: np.ndarray, k: int) -> dict[str, float]:
    """计算每一种表示的高维空间质量指标。

    silhouette_cosine 是经典的类内紧凑/类间分离指标。这里用 cosine 距离计算，
    与上面的近邻指标保持一致。数值越高，说明同亚型样本越靠近、不同亚型越分开。
    """
    scaled = robust_scale(x)
    metrics = knn_metrics(x, labels, k)
    metrics["silhouette_cosine"] = float(silhouette_score(scaled, labels, metric="cosine"))
    return metrics


def load_representations(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """载入需要比较的表示。

    mRNA、DNA methylation、miRNA 是三个单组学原始输入；
    Early fusion 是简单拼接后的固定表示；
    TRIGEL 是模型输出的融合表示。

    如果 TRIGEL 的指标显著优于单组学和 Early fusion，才能说明模型不是简单
    依赖某一个组学或简单拼接，而是学到了更适合不平衡亚型任务的表示空间。
    """
    return {
        "mRNA": load_omics(args.data_dir, 1),
        "DNA methylation": load_omics(args.data_dir, 2),
        "miRNA": load_omics(args.data_dir, 3),
        "Early fusion": np.load(args.early_fusion),
        "TRIGEL": np.load(args.trigel_embedding),
    }


def save_source_tables(
    output_dir: Path,
    projections: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    metrics.to_csv(output_dir / "representation_metrics.csv", index=False)
    rows = []
    for name, coords in projections.items():
        for i, (x, y) in enumerate(coords):
            rows.append(
                {
                    "representation": name,
                    "sample_index": i,
                    "projection_1": x,
                    "projection_2": y,
                    "label": int(labels[i]),
                    "subtype": SUBTYPE_NAMES[int(labels[i])],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "representation_projection_source_data.csv", index=False)


def plot_figure(
    projections: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    """绘制最终多面板图。

    上排五个散点图：
        展示单组学、简单拼接和 TRIGEL 在二维 PCA 空间中的样本分布。
        每个点是一个样本，颜色表示亚型标签。

    下排三个柱状图：
        1. 少数类同类近邻纯度：核心创新点证据，看 HER2-enriched 是否被保住；
        2. 整体同类近邻纯度：看所有亚型的局部结构是否更清晰；
        3. Cosine silhouette：看整体亚型分离度是否提高。

    图中不额外标注标签序号，以免画面拥挤；标签序号映射见文件顶部注释。
    """
    names = list(projections)
    fig = plt.figure(figsize=(12.4, 6.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.72], hspace=0.46)
    top_gs = outer[0].subgridspec(1, 5, wspace=0.28)
    bottom_gs = outer[1].subgridspec(1, 3, wspace=0.32)

    legend_handles = None
    for idx, name in enumerate(names):
        ax = fig.add_subplot(top_gs[0, idx])
        coords = projections[name]
        handles = []
        for label_id, subtype in enumerate(SUBTYPE_NAMES):
            mask = labels == label_id
            h = ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=10 if name != "TRIGEL" else 13,
                color=SUBTYPE_COLORS[label_id],
                alpha=0.72,
                linewidths=0,
                label=subtype,
            )
            handles.append(h)
        row = metrics[metrics["representation"] == name].iloc[0]
        ax.set_title(
            f"{name}\nSil={row['silhouette_cosine']:.3f}, "
            f"HER2-kNN={row['minority_neighbor_purity']:.2f}",
            fontsize=8.5,
            fontweight="bold" if name == "TRIGEL" else "normal",
        )
        ax.set_xlabel("PC1")
        if idx == 0:
            ax.set_ylabel("PC2")
        else:
            ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        if legend_handles is None:
            legend_handles = handles

    ax_bar1 = fig.add_subplot(bottom_gs[0, 0])
    ax_bar2 = fig.add_subplot(bottom_gs[0, 1])
    ax_bar3 = fig.add_subplot(bottom_gs[0, 2])

    x = np.arange(len(names))
    palette = ["#9AA5B1", "#9AA5B1", "#9AA5B1", "#6E7F99", "#D55E00"]

    bar_width = 0.46
    ax_bar1.bar(x, metrics["minority_neighbor_purity"], color=palette, width=bar_width)
    ax_bar1.set_title("Minority same-subtype neighbor purity")
    ax_bar1.set_ylabel("HER2-enriched kNN purity")
    ax_bar1.set_ylim(0, 1)
    ax_bar1.set_xticks(x)
    ax_bar1.set_xticklabels(names, rotation=28, ha="right")
    ax_bar1.grid(axis="y", color="#B0B0B0", alpha=0.22)

    ax_bar2.bar(x, metrics["overall_neighbor_purity"], color=palette, width=bar_width)
    ax_bar2.set_title("Overall same-subtype neighbor purity")
    ax_bar2.set_ylabel("All-sample kNN purity")
    ax_bar2.set_ylim(0, 1)
    ax_bar2.set_xticks(x)
    ax_bar2.set_xticklabels(names, rotation=28, ha="right")
    ax_bar2.grid(axis="y", color="#B0B0B0", alpha=0.22)

    ax_bar3.bar(x, metrics["silhouette_cosine"], color=palette, width=bar_width)
    ax_bar3.axhline(0, color="#555555", linewidth=0.7)
    ax_bar3.set_title("Subtype separation")
    ax_bar3.set_ylabel("Cosine silhouette")
    ax_bar3.set_xticks(x)
    ax_bar3.set_xticklabels(names, rotation=45, ha="right")
    ax_bar3.grid(axis="y", color="#B0B0B0", alpha=0.22)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            SUBTYPE_NAMES,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            columnspacing=1.2,
            handletextpad=0.3,
        )

    fig.suptitle(
        "Representation-space comparison under imbalanced BRCA subtypes",
        y=1.08,
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.86)

    stem = output_dir / "representation_space_comparison"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()
    output_dir = create_run_dir(args.output_dir)

    labels = load_labels(args.data_dir)
    representations = load_representations(args)
    for name, matrix in representations.items():
        if matrix.shape[0] != len(labels):
            raise ValueError(
                f"{name} has {matrix.shape[0]} samples, but labels have {len(labels)}."
            )

    projections = {
        name: projection_2d(matrix, args.random_state)
        for name, matrix in representations.items()
    }
    metrics = pd.DataFrame(
        [
            {"representation": name, **representation_metrics(matrix, labels, args.knn_k)}
            for name, matrix in representations.items()
        ]
    )

    save_source_tables(output_dir, projections, metrics, labels)
    plot_figure(projections, metrics, labels, output_dir)
    print(f"Representation-space comparison completed: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Subtype aggregation analysis on sample-similarity graphs.

This downstream analysis uses adj1.csv, adj2.csv, adj3.csv and subtype labels.
It asks whether graph edges preferentially connect samples from the same cancer
subtype and which subtype pairs are frequently connected.
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
        help="Use edge weights instead of binarizing nonzero adjacency entries.",
    )
    parser.add_argument(
        "--subtype-names",
        nargs="*",
        default=None,
        help="Optional subtype names in numeric-label order.",
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
    adj = pd.read_csv(data_dir / f"adj{graph_id}.csv", header=None).to_numpy(dtype=float)
    np.fill_diagonal(adj, 0.0)
    if not weighted:
        adj = (adj != 0).astype(float)
    return adj


def edge_count_matrix(
    adj: np.ndarray, labels: np.ndarray, subtype_names: list[str] | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar(summary["graph"], summary["intra_subtype_edge_ratio"], color="#4C78A8")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Intra-subtype edge ratio")
    ax.set_title("Subtype aggregation in sample graphs")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"intra_subtype_edge_ratio.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
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

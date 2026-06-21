#!/usr/bin/env python
"""Multi-omics subtype similarity-structure analysis.

This script compares sample-similarity structure in each omics layer and in
TRIGEL embeddings. It measures whether samples from the same subtype are more
similar than samples from different subtypes.
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
DEFAULT_TRIGEL_EMBEDDING = (
    PROJECT_ROOT
    / "results"
    / "BRCA_hyperparameter_search_20260118_202538"
    / "best_result"
    / "brca875_umap"
    / "BRCA875_TRIGEL_embeddings.npy"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "03_multiomics_similarity_structure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze subtype similarity structure.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--omics", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--trigel-embedding", type=Path, default=DEFAULT_TRIGEL_EMBEDDING)
    parser.add_argument(
        "--max-heatmap-samples",
        type=int,
        default=500,
        help="Maximum samples shown in heatmaps; statistics still use all samples.",
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


def load_omics(data_dir: Path, omics_id: int) -> np.ndarray:
    train = pd.read_csv(data_dir / f"{omics_id}_tr.csv", header=None)
    test = pd.read_csv(data_dir / f"{omics_id}_te.csv", header=None)
    return pd.concat([train, test], ignore_index=True).to_numpy(dtype=float)


def subtype_display(value: object, subtype_names: list[str] | None) -> str:
    idx = int(float(value))
    if subtype_names and 0 <= idx < len(subtype_names):
        return subtype_names[idx]
    return f"Subtype {idx}"


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / (norms + 1e-8)
    return np.clip(x @ x.T, -1.0, 1.0)


def sample_order(labels: np.ndarray, max_samples: int) -> np.ndarray:
    full_order = np.argsort(labels, kind="stable")
    if len(full_order) <= max_samples:
        return full_order
    selected = []
    classes = np.unique(labels)
    per_class = max(1, max_samples // len(classes))
    for cls in classes:
        cls_idx = full_order[labels[full_order] == cls]
        if len(cls_idx) > per_class:
            take = np.linspace(0, len(cls_idx) - 1, per_class).astype(int)
            cls_idx = cls_idx[take]
        selected.extend(cls_idx.tolist())
    return np.asarray(selected[:max_samples], dtype=int)


def plot_similarity_heatmap(
    sim: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_stem: Path,
    max_samples: int,
    subtype_names: list[str] | None,
) -> None:
    order = sample_order(labels, max_samples)
    labels_sorted = labels[order]
    values = sim[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    im = ax.imshow(values, cmap="viridis", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    start = 0
    tick_positions = []
    tick_labels = []
    for cls in np.unique(labels_sorted):
        count = int(np.sum(labels_sorted == cls))
        end = start + count
        ax.axhline(end - 0.5, color="white", linewidth=0.5)
        ax.axvline(end - 0.5, color="white", linewidth=0.5)
        tick_positions.append((start + end - 1) / 2)
        tick_labels.append(subtype_display(cls, subtype_names))
        start = end
    secx = ax.secondary_xaxis("top")
    secx.set_xticks(tick_positions)
    secx.set_xticklabels(tick_labels, rotation=30, ha="left", fontsize=8)
    secy = ax.secondary_yaxis("right")
    secy.set_yticks(tick_positions)
    secy.set_yticklabels(tick_labels, fontsize=8)
    fig.colorbar(im, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def similarity_stats(sim: np.ndarray, labels: np.ndarray, name: str) -> dict[str, float]:
    upper = np.triu_indices_from(sim, k=1)
    same = labels[upper[0]] == labels[upper[1]]
    within = sim[upper][same]
    between = sim[upper][~same]
    return {
        "representation": name,
        "within_mean": float(np.mean(within)),
        "within_median": float(np.median(within)),
        "between_mean": float(np.mean(between)),
        "between_median": float(np.median(between)),
        "separation_mean": float(np.mean(within) - np.mean(between)),
        "within_pairs": int(len(within)),
        "between_pairs": int(len(between)),
    }


def plot_within_between(summary_long: pd.DataFrame, output_dir: Path) -> None:
    reps = summary_long["representation"].unique().tolist()
    x = np.arange(len(reps))
    width = 0.36
    within = summary_long.set_index("representation").loc[reps, "within_mean"]
    between = summary_long.set_index("representation").loc[reps, "between_mean"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, within, width, label="Within subtype")
    ax.bar(x + width / 2, between, width, label="Between subtype")
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=30, ha="right")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Within-subtype vs between-subtype similarity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"within_between_similarity.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = create_run_dir(args.output_dir)
    labels = load_labels(args.data_dir, args.labels)
    rows = []

    for omics_id in args.omics:
        x = load_omics(args.data_dir, omics_id)
        if x.shape[0] != len(labels):
            raise ValueError(f"Omics {omics_id} sample count does not match labels.")
        sim = cosine_similarity_matrix(x)
        name = f"Omics {omics_id}"
        rows.append(similarity_stats(sim, labels, name))
        plot_similarity_heatmap(
            sim,
            labels,
            f"{name} sample similarity",
            output_dir / f"omics{omics_id}_similarity_heatmap",
            args.max_heatmap_samples,
            args.subtype_names,
        )

    if args.trigel_embedding.exists():
        emb = np.load(args.trigel_embedding)
        if emb.shape[0] == len(labels):
            sim = cosine_similarity_matrix(emb)
            rows.append(similarity_stats(sim, labels, "TRIGEL"))
            plot_similarity_heatmap(
                sim,
                labels,
                "TRIGEL representation similarity",
                output_dir / "trigel_similarity_heatmap",
                args.max_heatmap_samples,
                args.subtype_names,
            )
        else:
            print(
                f"[skip] TRIGEL embedding has {emb.shape[0]} rows, labels have "
                f"{len(labels)} rows."
            )
    else:
        print(f"[skip] TRIGEL embedding not found: {args.trigel_embedding}")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "similarity_structure_summary.csv", index=False)
    plot_within_between(summary, output_dir)
    print(f"Multi-omics similarity-structure analysis completed: {output_dir}")


if __name__ == "__main__":
    main()

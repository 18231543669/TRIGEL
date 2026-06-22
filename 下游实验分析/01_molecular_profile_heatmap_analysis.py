#!/usr/bin/env python
"""Subtype-level multi-omics molecular profile heatmap analysis.

This downstream analysis uses the current feature matrices and subtype labels
only. It selects subtype-discriminative features in each omics layer and draws
heatmaps sorted by subtype.
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "01_molecular_profile_heatmap"
DEFAULT_SUBTYPE_NAMES = [
    "Luminal A",
    "Luminal B",
    "HER2-enriched",
    "Basal-like",
    "Normal-like",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw multi-omics subtype molecular profile heatmaps."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Optional sample-level subtype labels. If omitted, labels_tr.csv "
            "and labels_te.csv are concatenated."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top subtype-discriminative features shown in each heatmap.",
    )
    parser.add_argument("--omics", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--subtype-names",
        nargs="*",
        default=DEFAULT_SUBTYPE_NAMES,
        help=(
            "Optional subtype names in numeric-label order. Defaults to PAM50 "
            "subtype names: Luminal A, Luminal B, HER2-enriched, Basal-like, "
            "Normal-like."
        ),
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


def load_omics(data_dir: Path, omics_id: int) -> tuple[np.ndarray, list[str]]:
    train = pd.read_csv(data_dir / f"{omics_id}_tr.csv", header=None)
    test = pd.read_csv(data_dir / f"{omics_id}_te.csv", header=None)
    matrix = pd.concat([train, test], ignore_index=True).to_numpy(dtype=float)
    names = pd.read_csv(data_dir / f"{omics_id}_featname.csv", header=None).iloc[:, 0]
    feature_names = [str(x) for x in names.tolist()]
    if len(feature_names) != matrix.shape[1]:
        feature_names = [f"omics{omics_id}_feature_{i}" for i in range(matrix.shape[1])]
    return matrix, feature_names


def subtype_display(value: object, subtype_names: list[str] | None) -> str:
    if subtype_names is None:
        return f"Subtype {int(float(value))}"
    idx = int(float(value))
    return subtype_names[idx] if 0 <= idx < len(subtype_names) else f"Subtype {idx}"


def feature_scores_by_subtype(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Compute an ANOVA-like between/within subtype score for each feature."""
    classes = np.unique(y)
    grand = np.nanmean(x, axis=0)
    between = np.zeros(x.shape[1], dtype=float)
    within = np.zeros(x.shape[1], dtype=float)
    for cls in classes:
        subset = x[y == cls]
        if subset.size == 0:
            continue
        mean = np.nanmean(subset, axis=0)
        between += subset.shape[0] * (mean - grand) ** 2
        within += np.nansum((subset - mean) ** 2, axis=0)
    score = between / (within + 1e-8)
    return pd.DataFrame({"feature_index": np.arange(x.shape[1]), "score": score})


def zscore_rows(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    return np.clip((values - mean) / (std + 1e-8), -2.5, 2.5)


def plot_heatmap(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    selected: np.ndarray,
    omics_id: int,
    output_dir: Path,
    subtype_names: list[str] | None,
) -> None:
    order = np.argsort(labels, kind="stable")
    labels_sorted = labels[order]
    values = zscore_rows(matrix[order][:, selected].T)
    names = [feature_names[i] for i in selected]

    fig_height = max(5.2, min(9.0, 0.32 * len(selected) + 2.8))
    fig, ax = plt.subplots(figsize=(11.0, fig_height))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_title(f"Omics {omics_id}: top PAM50 subtype-discriminative features")
    ax.set_xlabel("Samples sorted by subtype")
    ax.set_ylabel("Features")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks([])

    boundaries = []
    midpoints = []
    tick_labels = []
    start = 0
    for cls in np.unique(labels_sorted):
        count = int(np.sum(labels_sorted == cls))
        end = start + count
        boundaries.append(end - 0.5)
        midpoints.append((start + end - 1) / 2)
        tick_labels.append(subtype_display(cls, subtype_names))
        start = end
    for boundary in boundaries[:-1]:
        ax.axvline(boundary, color="black", linewidth=0.6)
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(midpoints)
    secax.set_xticklabels(tick_labels, rotation=30, ha="left", fontsize=8)
    fig.colorbar(im, ax=ax, label="Row z-score", fraction=0.025, pad=0.02)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"omics{omics_id}_profile_heatmap.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = create_run_dir(args.output_dir)
    labels = load_labels(args.data_dir, args.labels)
    summary_rows = []

    for omics_id in args.omics:
        matrix, feature_names = load_omics(args.data_dir, omics_id)
        if matrix.shape[0] != len(labels):
            raise ValueError(
                f"Omics {omics_id} has {matrix.shape[0]} samples but labels have "
                f"{len(labels)} samples."
            )
        scores = feature_scores_by_subtype(matrix, labels)
        scores["feature_name"] = scores["feature_index"].map(lambda i: feature_names[i])
        scores = scores.sort_values("score", ascending=False)
        scores.to_csv(output_dir / f"omics{omics_id}_feature_scores.csv", index=False)
        selected = scores.head(args.top_n)["feature_index"].to_numpy(dtype=int)
        plot_heatmap(
            matrix,
            labels,
            feature_names,
            selected,
            omics_id,
            output_dir,
            args.subtype_names,
        )
        summary_rows.append(
            {
                "omics": omics_id,
                "num_samples": matrix.shape[0],
                "num_features": matrix.shape[1],
                "top_n": len(selected),
                "top_feature": scores.iloc[0]["feature_name"],
                "top_score": scores.iloc[0]["score"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "analysis_summary.csv", index=False)
    print(f"Molecular profile heatmap analysis completed: {output_dir}")


if __name__ == "__main__":
    main()

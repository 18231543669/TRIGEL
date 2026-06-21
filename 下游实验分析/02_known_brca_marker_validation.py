#!/usr/bin/env python
"""Known BRCA subtype-marker consistency validation.

This analysis does not claim novel biomarker discovery. It checks whether
classic breast-cancer subtype markers show expected subtype-level differences
in the current multi-omics matrices.
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "02_known_brca_marker_validation"

DEFAULT_MARKERS = {
    "Luminal_ER_positive": ["ESR1", "PGR", "FOXA1", "GATA3", "BCL2"],
    "HER2_enriched": ["ERBB2", "GRB7", "PGAP3"],
    "Basal_like": ["KRT5", "KRT14", "KRT17", "EGFR", "KRT6A"],
    "Proliferation": ["MKI67", "TOP2A", "AURKA", "BIRC5", "CCNB1"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate known BRCA markers.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--omics", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--markers",
        nargs="*",
        default=None,
        help="Optional marker symbols. If omitted, classic BRCA markers are used.",
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


def load_omics(data_dir: Path, omics_id: int) -> tuple[np.ndarray, list[str], list[str]]:
    train = pd.read_csv(data_dir / f"{omics_id}_tr.csv", header=None)
    test = pd.read_csv(data_dir / f"{omics_id}_te.csv", header=None)
    matrix = pd.concat([train, test], ignore_index=True).to_numpy(dtype=float)
    raw_names = pd.read_csv(data_dir / f"{omics_id}_featname.csv", header=None).iloc[:, 0]
    feature_names = [str(x) for x in raw_names.tolist()]
    symbols = [name.split("|")[0].upper() for name in feature_names]
    return matrix, feature_names, symbols


def subtype_display(value: object, subtype_names: list[str] | None) -> str:
    idx = int(float(value))
    if subtype_names and 0 <= idx < len(subtype_names):
        return subtype_names[idx]
    return f"Subtype {idx}"


def marker_list(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.markers:
        return [("Custom", marker.upper()) for marker in args.markers]
    rows = []
    for group, markers in DEFAULT_MARKERS.items():
        rows.extend((group, marker.upper()) for marker in markers)
    return rows


def find_markers(
    data_dir: Path, omics_ids: list[int], markers: list[tuple[str, str]]
) -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, list[str], list[str]]]]:
    omics_data = {}
    found_rows = []
    marker_symbols = [marker for _, marker in markers]
    for omics_id in omics_ids:
        matrix, feature_names, symbols = load_omics(data_dir, omics_id)
        omics_data[omics_id] = (matrix, feature_names, symbols)
        symbol_to_indices: dict[str, list[int]] = {}
        for idx, symbol in enumerate(symbols):
            symbol_to_indices.setdefault(symbol, []).append(idx)
        for group, marker in markers:
            for idx in symbol_to_indices.get(marker, []):
                found_rows.append(
                    {
                        "marker_group": group,
                        "marker": marker,
                        "omics": omics_id,
                        "feature_index": idx,
                        "feature_name": feature_names[idx],
                    }
                )
    return pd.DataFrame(found_rows), omics_data


def zscore_rows(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    return np.clip((values - mean) / (std + 1e-8), -2.5, 2.5)


def plot_marker_heatmap(
    marker_matrix: np.ndarray,
    marker_labels: list[str],
    labels: np.ndarray,
    output_dir: Path,
    subtype_names: list[str] | None,
) -> None:
    order = np.argsort(labels, kind="stable")
    labels_sorted = labels[order]
    values = zscore_rows(marker_matrix[:, order])

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.32 * len(marker_labels) + 2.5)))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_title("Known BRCA subtype-marker patterns")
    ax.set_xlabel("Samples sorted by subtype")
    ax.set_ylabel("Known markers")
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(marker_labels)))
    ax.set_yticklabels(marker_labels, fontsize=8)

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
        fig.savefig(output_dir / f"known_brca_marker_heatmap.{ext}", dpi=300)
    plt.close(fig)


def plot_marker_boxplots(
    marker_matrix: np.ndarray,
    marker_labels: list[str],
    labels: np.ndarray,
    output_dir: Path,
    subtype_names: list[str] | None,
) -> None:
    classes = np.unique(labels)
    chosen = np.arange(min(len(marker_labels), 12))
    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 3.0 * nrows)))
    axes = np.asarray(axes).reshape(-1)
    rng = np.random.default_rng(2026)
    for ax, marker_idx in zip(axes, chosen):
        values_by_class = [marker_matrix[marker_idx, labels == cls] for cls in classes]
        ax.boxplot(values_by_class, showfliers=False, patch_artist=True)
        for pos, values in enumerate(values_by_class, start=1):
            x = rng.normal(pos, 0.035, size=len(values))
            ax.scatter(x, values, s=8, alpha=0.35)
        ax.set_title(marker_labels[marker_idx], fontsize=9)
        ax.set_xticks(np.arange(1, len(classes) + 1))
        ax.set_xticklabels(
            [subtype_display(cls, subtype_names) for cls in classes],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(chosen) :]:
        ax.axis("off")
    fig.suptitle("Known marker distributions across subtypes", y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"known_brca_marker_boxplots.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = create_run_dir(args.output_dir)
    labels = load_labels(args.data_dir, args.labels)
    markers = marker_list(args)
    found, omics_data = find_markers(args.data_dir, args.omics, markers)
    found.to_csv(output_dir / "matched_known_markers.csv", index=False)

    if found.empty:
        raise ValueError(
            "None of the requested markers were found in the selected feature names."
        )

    rows = []
    marker_values = []
    marker_labels = []
    for _, row in found.iterrows():
        matrix, _, _ = omics_data[int(row["omics"])]
        values = matrix[:, int(row["feature_index"])]
        marker_values.append(values)
        marker_labels.append(f"{row['marker']} | omics{row['omics']}")
        for cls in np.unique(labels):
            subset = values[labels == cls]
            rows.append(
                {
                    "marker": row["marker"],
                    "marker_group": row["marker_group"],
                    "omics": row["omics"],
                    "subtype": subtype_display(cls, args.subtype_names),
                    "n": len(subset),
                    "mean": float(np.nanmean(subset)),
                    "median": float(np.nanmedian(subset)),
                    "std": float(np.nanstd(subset)),
                }
            )

    marker_matrix = np.vstack(marker_values)
    pd.DataFrame(rows).to_csv(output_dir / "marker_by_subtype_summary.csv", index=False)
    plot_marker_heatmap(marker_matrix, marker_labels, labels, output_dir, args.subtype_names)
    plot_marker_boxplots(marker_matrix, marker_labels, labels, output_dir, args.subtype_names)
    print(f"Known BRCA marker validation completed: {output_dir}")


if __name__ == "__main__":
    main()

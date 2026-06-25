#!/usr/bin/env python
"""Subtype sample distribution, TGS grouping, and BRCA marker annotation.

This figure is intended to show that the BRCA task is imbalanced and that the
imbalanced labels correspond to biologically meaningful PAM50 subtypes.
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
    from matplotlib.patches import Patch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: matplotlib.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BRCA"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "05_subtype_distribution_tgs_marker_annotation"

SUBTYPE_NAMES = {
    0: "Normal-like",
    1: "Basal-like",
    2: "HER2-enriched",
    3: "Luminal A",
    4: "Luminal B",
}

TGS_GROUPS = {
    0: "Intermediate",
    1: "Intermediate",
    2: "Minority",
    3: "Majority",
    4: "Intermediate",
}

MARKER_ANNOTATIONS = {
    0: "Normal-like/stromal context\nCNN1, MYH11, LMOD1-like signals",
    1: "Basal/epithelial program\nFOXC1, KRT5/14/17, EGFR-related axis",
    2: "HER2-amplified program\nERBB2, GRB7, PGAP3",
    3: "ER-positive luminal program\nESR1, PGR, FOXA1, GATA3, MLPH",
    4: "Luminal/proliferative program\nER axis with cell-cycle activity",
}

GROUP_COLORS = {
    "Minority": "#D55E00",
    "Intermediate": "#4C78A8",
    "Majority": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot BRCA subtype sample distribution with TGS grouping."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional label file. If omitted, labels_tr.csv and labels_te.csv are combined.",
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
        return pd.read_csv(labels_path, header=None).iloc[:, 0].to_numpy(dtype=int)
    train = pd.read_csv(data_dir / "labels_tr.csv", header=None).iloc[:, 0]
    test = pd.read_csv(data_dir / "labels_te.csv", header=None).iloc[:, 0]
    return pd.concat([train, test], ignore_index=True).to_numpy(dtype=int)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def build_summary(labels: np.ndarray) -> pd.DataFrame:
    rows = []
    total = len(labels)
    for subtype_id in sorted(SUBTYPE_NAMES):
        count = int(np.sum(labels == subtype_id))
        rows.append(
            {
                "subtype_id": subtype_id,
                "subtype": SUBTYPE_NAMES[subtype_id],
                "tgs_group": TGS_GROUPS[subtype_id],
                "count": count,
                "fraction": count / total,
                "marker_annotation": MARKER_ANNOTATIONS[subtype_id].replace("\n", "; "),
            }
        )
    return pd.DataFrame(rows)


def plot_distribution(summary: pd.DataFrame, output_dir: Path) -> None:
    fig = plt.figure(figsize=(8.8, 5.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    ax_note = fig.add_subplot(gs[0, 1])
    ax_note.axis("off")

    x = np.arange(len(summary))
    colors = [GROUP_COLORS[group] for group in summary["tgs_group"]]
    bars = ax.bar(x, summary["count"], color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("PAM50 subtype")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["subtype"], rotation=28, ha="right")
    ax.set_title("Subtype sample distribution and TGS grouping", pad=10)
    ax.grid(axis="y", color="#B0B0B0", alpha=0.22, linewidth=0.8)

    total = int(summary["count"].sum())
    for bar, count, fraction in zip(bars, summary["count"], summary["fraction"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(summary["count"]) * 0.015,
            f"n={int(count)}\n{fraction:.1%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, max(summary["count"]) * 1.22)

    handles = [Patch(facecolor=color, label=group) for group, color in GROUP_COLORS.items()]
    ax.legend(handles=handles, title="TGS group", loc="upper left", bbox_to_anchor=(0.02, 0.98))

    ax_note.set_title("Biological subtype annotation", loc="left", pad=10)
    y_positions = [0.90, 0.72, 0.54, 0.36, 0.18]
    for y, (_, row) in zip(y_positions, summary.iterrows()):
        color = GROUP_COLORS[row["tgs_group"]]
        ax_note.text(0.00, y, row["subtype"], color=color, fontweight="bold", transform=ax_note.transAxes)
        ax_note.text(
            0.00,
            y - 0.075,
            MARKER_ANNOTATIONS[int(row["subtype_id"])],
            transform=ax_note.transAxes,
            fontsize=7.1,
            va="top",
            linespacing=1.25,
        )
    ax_note.text(
        0.00,
        -0.03,
        f"Total samples: {total}. TGS groups are assigned from class frequency.",
        transform=ax_note.transAxes,
        fontsize=7,
        color="#4A4A4A",
    )

    stem = output_dir / "subtype_distribution_tgs_marker_annotation"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()
    output_dir = create_run_dir(args.output_dir)
    labels = load_labels(args.data_dir, args.labels)
    summary = build_summary(labels)
    summary.to_csv(output_dir / "subtype_distribution_tgs_marker_annotation.csv", index=False)
    plot_distribution(summary, output_dir)
    print(f"Subtype distribution figure completed: {output_dir}")


if __name__ == "__main__":
    main()

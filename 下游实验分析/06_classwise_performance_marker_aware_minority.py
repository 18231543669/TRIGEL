#!/usr/bin/env python
"""Class-wise TRIGEL performance and minority-subtype biological context.

The performance panel is calculated from saved confusion matrices in the best
TRIGEL result JSON, so this script does not load torch or a trained model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: matplotlib.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BRCA"
DEFAULT_RESULT_JSON = (
    PROJECT_ROOT
    / "results"
    / "BRCA_hyperparameter_search_20260118_202538"
    / "best_result"
    / "best_hyperparameter_result.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "06_classwise_performance_marker_aware_minority"

SUBTYPE_NAMES = ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]
MINORITY_SUBTYPE_INDEX = 2

CANONICAL_MARKER_SETS = {
    "Luminal canonical markers": ["ESR1", "PGR", "FOXA1", "GATA3", "BCL2", "MLPH"],
    "Basal/proliferation canonical markers": ["FOXC1", "KRT17", "AURKA", "BIRC5", "CCNB1"],
}
HER2_CANONICAL_MARKERS = ["ERBB2", "GRB7", "PGAP3", "MIEN1", "STARD3"]
SIGNATURE_TARGETS = {
    "Normal-like signature": [0],
    "Basal-like signature": [1],
    "Luminal subtype signature": [3, 4],
    "HER2-enriched subtype signature": [2],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot class-wise TRIGEL recall/F1 and minority-subtype marker context."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-signature-genes", type=int, default=8)
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
            "font.size": 8,
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


def load_omics(data_dir: Path, omics_id: int) -> tuple[np.ndarray, list[str]]:
    train = pd.read_csv(data_dir / f"{omics_id}_tr.csv", header=None)
    test = pd.read_csv(data_dir / f"{omics_id}_te.csv", header=None)
    matrix = pd.concat([train, test], ignore_index=True).to_numpy(dtype=float)
    raw_names = pd.read_csv(data_dir / f"{omics_id}_featname.csv", header=None).iloc[:, 0]
    symbols = [str(name).split("|")[0].upper() for name in raw_names.tolist()]
    return matrix, symbols


def zscore_columns(matrix: np.ndarray) -> np.ndarray:
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    std = np.nanstd(matrix, axis=0, keepdims=True)
    return (matrix - mean) / (std + 1e-8)


def collect_confusion_matrices(result_json: Path) -> list[np.ndarray]:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    runs = data.get("full_result", {}).get("all_runs_details", [])
    matrices = []
    for run in runs:
        cm = (
            run.get("original_result", {})
            .get("test_metrics", {})
            .get("conf_matrix")
        )
        if cm is not None:
            matrices.append(np.asarray(cm, dtype=float))
    if not matrices:
        raise ValueError(f"No confusion matrices found in {result_json}")
    return matrices


def metrics_from_confusion(cm: np.ndarray) -> pd.DataFrame:
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    recall = tp / np.maximum(support, 1)
    precision = tp / np.maximum(predicted, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return pd.DataFrame(
        {
            "subtype_id": np.arange(len(tp)),
            "subtype": SUBTYPE_NAMES,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def summarize_classwise_metrics(matrices: list[np.ndarray]) -> pd.DataFrame:
    frames = []
    for run_id, cm in enumerate(matrices, start=1):
        frame = metrics_from_confusion(cm)
        frame.insert(0, "run_id", run_id)
        frames.append(frame)
    all_metrics = pd.concat(frames, ignore_index=True)
    summary = (
        all_metrics.groupby(["subtype_id", "subtype"], as_index=False)
        .agg(
            support_mean=("support", "mean"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
        )
    )
    summary["imbalance_group"] = np.where(
        summary["subtype_id"] == MINORITY_SUBTYPE_INDEX,
        "Minority",
        np.where(summary["subtype"] == "Luminal A", "Majority", "Intermediate"),
    )
    return all_metrics, summary


def marker_score(
    matrix: np.ndarray, symbols: list[str], markers: list[str]
) -> tuple[np.ndarray | None, list[str]]:
    symbol_to_indices: dict[str, list[int]] = {}
    for idx, symbol in enumerate(symbols):
        symbol_to_indices.setdefault(symbol, []).append(idx)
    indices = []
    matched = []
    for marker in markers:
        found = symbol_to_indices.get(marker.upper(), [])
        if found:
            indices.extend(found)
            matched.append(marker.upper())
    if not indices:
        return None, matched
    z = zscore_columns(matrix[:, indices])
    return np.nanmean(z, axis=1), matched


def subtype_discriminative_signature(
    matrix: np.ndarray,
    symbols: list[str],
    labels: np.ndarray,
    target_subtypes: list[int],
    signature_name: str,
    top_n: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    z = zscore_columns(matrix)
    target = np.isin(labels, target_subtypes)
    target_mean = np.nanmean(z[target], axis=0)
    other_mean = np.nanmean(z[~target], axis=0)
    effect = target_mean - other_mean
    order = np.argsort(effect)[::-1][:top_n]
    score = np.nanmean(z[:, order], axis=1)
    signature = pd.DataFrame(
        {
            "signature_name": signature_name,
            "target_subtypes": ", ".join(SUBTYPE_NAMES[i] for i in target_subtypes),
            "rank": np.arange(1, len(order) + 1),
            "feature_index": order,
            "gene_symbol": [symbols[i] for i in order],
            "target_minus_others_z": effect[order],
            "note": "Data-derived subtype-discriminative signature from the current mRNA feature table.",
        }
    )
    return score, signature


def build_marker_context(
    data_dir: Path, labels: np.ndarray, top_signature_genes: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix, symbols = load_omics(data_dir, 1)
    if matrix.shape[0] != len(labels):
        raise ValueError("Omics 1 and labels have different sample counts.")

    score_rows = []
    availability_rows = []
    for score_name, markers in CANONICAL_MARKER_SETS.items():
        _, matched = marker_score(matrix, symbols, markers)
        availability_rows.append(
            {
                "score_name": score_name,
                "requested_markers": ", ".join(markers),
                "matched_markers": ", ".join(matched),
                "missing_markers": ", ".join([m for m in markers if m.upper() not in matched]),
            }
        )

    _, canonical_matched = marker_score(matrix, symbols, HER2_CANONICAL_MARKERS)
    availability_rows.append(
        {
            "score_name": "HER2 canonical marker score",
            "requested_markers": ", ".join(HER2_CANONICAL_MARKERS),
            "matched_markers": ", ".join(canonical_matched),
            "missing_markers": ", ".join(
                [m for m in HER2_CANONICAL_MARKERS if m.upper() not in canonical_matched]
            ),
        }
    )

    signature_frames = []
    for signature_name, target_subtypes in SIGNATURE_TARGETS.items():
        score, signature = subtype_discriminative_signature(
            matrix,
            symbols,
            labels,
            target_subtypes,
            signature_name,
            top_signature_genes,
        )
        signature_frames.append(signature)
        for subtype_id, subtype in enumerate(SUBTYPE_NAMES):
            values = score[labels == subtype_id]
            score_rows.append(
                {
                    "score_name": signature_name,
                    "subtype_id": subtype_id,
                    "subtype": subtype,
                    "mean_score": float(np.nanmean(values)),
                    "median_score": float(np.nanmedian(values)),
                    "n": int(len(values)),
                }
            )

    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(availability_rows),
        pd.concat(signature_frames, ignore_index=True),
    )


def plot_figure(summary: pd.DataFrame, marker_context: pd.DataFrame, output_dir: Path) -> None:
    fig = plt.figure(figsize=(10.2, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 0.95], wspace=0.56)
    ax = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])

    x = np.arange(len(summary))
    width = 0.32
    colors = ["#D55E00" if sid == MINORITY_SUBTYPE_INDEX else "#4C78A8" for sid in summary["subtype_id"]]
    ax.bar(
        x - width / 2,
        summary["recall_mean"],
        width,
        yerr=summary["recall_std"].fillna(0),
        capsize=2,
        label="Recall",
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        summary["f1_mean"],
        width,
        yerr=summary["f1_std"].fillna(0),
        capsize=2,
        label="F1",
        color=["#E6A57E" if sid == MINORITY_SUBTYPE_INDEX else "#A8BED6" for sid in summary["subtype_id"]],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0.8, color="#666666", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Test metric")
    ax.set_xlabel("PAM50 subtype")
    ax.set_title("Class-wise TRIGEL performance", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["subtype"], rotation=28, ha="right")
    ax.grid(axis="y", color="#B0B0B0", alpha=0.22, linewidth=0.8)
    ax.legend(loc="lower right")
    for i, row in summary.iterrows():
        ax.text(
            i,
            1.005,
            f"n~{int(round(row['support_mean']))}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#555555",
        )

    matrix = marker_context.pivot(index="score_name", columns="subtype", values="mean_score")
    matrix = matrix.reindex(columns=SUBTYPE_NAMES)
    values = matrix.to_numpy(dtype=float)
    im = ax_hm.imshow(values, cmap="coolwarm", vmin=-1.4, vmax=1.4, aspect="auto")
    ax_hm.set_title("Subtype-discriminative signature context", pad=10)
    ax_hm.set_xticks(np.arange(len(SUBTYPE_NAMES)))
    ax_hm.set_xticklabels(SUBTYPE_NAMES, rotation=32, ha="right")
    ax_hm.set_yticks(np.arange(len(matrix.index)))
    short_labels = {
        "Luminal subtype signature": "Luminal sig.",
        "HER2-enriched subtype signature": "HER2-enriched sig.",
        "Basal-like subtype signature": "Basal-like sig.",
        "Normal-like signature": "Normal-like sig.",
    }
    ax_hm.set_yticklabels([short_labels.get(label, label) for label in matrix.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax_hm.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label="Mean z-score")

    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.22, top=0.86)
    stem = output_dir / "classwise_performance_marker_aware_minority"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()
    output_dir = create_run_dir(args.output_dir)
    labels = load_labels(args.data_dir)
    matrices = collect_confusion_matrices(args.result_json)
    all_metrics, summary = summarize_classwise_metrics(matrices)
    marker_context, marker_availability, her2_signature = build_marker_context(
        args.data_dir, labels, args.top_signature_genes
    )

    all_metrics.to_csv(output_dir / "classwise_metrics_all_runs.csv", index=False)
    summary.to_csv(output_dir / "classwise_metrics_summary.csv", index=False)
    marker_context.to_csv(output_dir / "minority_marker_context_by_subtype.csv", index=False)
    marker_availability.to_csv(output_dir / "marker_availability.csv", index=False)
    if not her2_signature.empty:
        her2_signature.to_csv(output_dir / "subtype_data_derived_signatures.csv", index=False)

    plot_figure(summary, marker_context, output_dir)
    print(f"Class-wise performance figure completed: {output_dir}")


if __name__ == "__main__":
    main()

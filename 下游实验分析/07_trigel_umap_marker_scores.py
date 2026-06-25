#!/usr/bin/env python
"""TRIGEL embedding projection colored by BRCA subtype marker/signature scores.

The script supports UMAP, but defaults to PCA because some local environments
can hang during umap/numba initialization. Use ``--method umap`` when UMAP runs
normally in the selected Python environment.
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
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: matplotlib.") from exc

try:
    from sklearn.decomposition import PCA
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: scikit-learn.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BRCA"
DEFAULT_EMBEDDING = DEFAULT_DATA_DIR / "BRCA875_TRIGEL_embeddings.npy"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "07_trigel_umap_marker_scores"

SUBTYPE_NAMES = ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]
SUBTYPE_COLORS = ["#B279A2", "#009E73", "#D55E00", "#4C78A8", "#72B7B2"]
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
        description="Draw UMAP of TRIGEL embeddings colored by subtype marker scores."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--embedding", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-signature-genes", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument(
        "--method",
        choices=["pca", "umap"],
        default="pca",
        help=(
            "Projection method. Defaults to pca for reliable local execution; "
            "use umap when umap-learn initializes correctly."
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


def build_scores(
    data_dir: Path, labels: np.ndarray, top_signature_genes: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix, symbols = load_omics(data_dir, 1)
    if matrix.shape[0] != len(labels):
        raise ValueError("Omics 1 and labels have different sample counts.")

    score_table: dict[str, np.ndarray] = {}
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
        score_table[signature_name] = score
        signature_frames.append(signature)

    return (
        pd.DataFrame(score_table),
        pd.DataFrame(availability_rows),
        pd.concat(signature_frames, ignore_index=True),
    )


def plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    scores: pd.DataFrame,
    output_dir: Path,
    method_label: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.0), constrained_layout=True)
    axes = axes.reshape(-1)

    ax = axes[0]
    for subtype_id, subtype in enumerate(SUBTYPE_NAMES):
        mask = labels == subtype_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=14,
            color=SUBTYPE_COLORS[subtype_id],
            label=f"{subtype} (n={int(mask.sum())})",
            alpha=0.78,
            linewidths=0,
        )
    ax.set_title(f"TRIGEL embedding by PAM50 subtype ({method_label})")
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.legend(loc="best", fontsize=6.6, markerscale=1.1)

    for ax, score_name in zip(axes[1:], scores.columns):
        values = scores[score_name].to_numpy(dtype=float)
        lim = np.nanpercentile(np.abs(values), 98)
        lim = max(lim, 0.5)
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=np.clip(values, -lim, lim),
            cmap="coolwarm",
            vmin=-lim,
            vmax=lim,
            s=13,
            alpha=0.82,
            linewidths=0,
        )
        ax.set_title(score_name)
        ax.set_xlabel(f"{method_label} 1")
        ax.set_ylabel(f"{method_label} 2")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02, label="Sample z-score")

    for ax in axes[1 + len(scores.columns) :]:
        ax.axis("off")

    for ax in axes:
        if not ax.axison:
            continue
        ax.set_xticks([])
        ax.set_yticks([])

    stem = output_dir / f"trigel_{method_label.lower()}_marker_scores"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_style()
    output_dir = create_run_dir(args.output_dir)

    labels = load_labels(args.data_dir)
    embedding = np.load(args.embedding)
    if embedding.shape[0] != len(labels):
        raise ValueError(
            f"Embedding has {embedding.shape[0]} samples, but labels have {len(labels)}."
        )
    scores, marker_availability, her2_signature = build_scores(
        args.data_dir, labels, args.top_signature_genes
    )

    if args.method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Missing dependency: umap-learn.") from exc
        reducer = umap.UMAP(
            n_neighbors=20,
            min_dist=0.18,
            metric="euclidean",
            random_state=args.random_state,
            n_epochs=200,
            low_memory=False,
        )
        coords = reducer.fit_transform(embedding)
        method_label = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=args.random_state)
        coords = reducer.fit_transform(embedding)
        method_label = "PCA"

    pd.DataFrame(
        {
            "projection_1": coords[:, 0],
            "projection_2": coords[:, 1],
            "projection_method": method_label,
            "subtype_id": labels,
            "subtype": [SUBTYPE_NAMES[i] for i in labels],
        }
    ).join(scores).to_csv(output_dir / "trigel_projection_marker_scores_source_data.csv", index=False)
    marker_availability.to_csv(output_dir / "marker_availability.csv", index=False)
    if not her2_signature.empty:
        her2_signature.to_csv(output_dir / "subtype_data_derived_signatures.csv", index=False)

    plot_umap(coords, labels, scores, output_dir, method_label)
    print(f"TRIGEL {method_label} marker-score figure completed: {output_dir}")


if __name__ == "__main__":
    main()

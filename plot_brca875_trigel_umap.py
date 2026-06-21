#!/usr/bin/env python
"""Export BRCA875 TRIGEL embeddings and draw a publication-ready UMAP figure.

The default settings reproduce the highest-ranked checkpoint saved by the
specified hyperparameter-search result. The figure compares a fixed
multi-omics early-fusion input with the learned TRIGEL embedding.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import torch
    import umap
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import davies_bouldin_score, silhouette_score
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: "
        f"{exc.name}. Install torch, torch-geometric, umap-learn, "
        "scikit-learn, pandas, matplotlib, and scipy in the project environment."
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from augmentation.augmentors import DataAugmentor  # noqa: E402
from models.gcn_models import ModelFactory  # noqa: E402
from utils.data_utils import (  # noqa: E402
    construct_adjacency,
    load_dataset,
    load_precomputed_adjacency_csv,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "BRCA"
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "BRCA_hyperparameter_search_20260118_202538"
    / "best_result"
)
DEFAULT_CHECKPOINT = (
    DEFAULT_RESULT_DIR / "models" / "rank1_acc0_9011_seed97048.pt"
)
DEFAULT_CONFIG = DEFAULT_RESULT_DIR / "best_hyperparameter_result.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR / "brca875_umap"

DEFAULT_SEED = 97048
DEFAULT_UMAP_SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real BRCA875 TRIGEL embeddings and draw UMAP."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-json", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--view",
        choices=("enhanced", "original"),
        default="enhanced",
        help="Use the training-time enhanced view or the original graph view.",
    )
    parser.add_argument(
        "--graph-source",
        choices=("knn", "precomputed"),
        default="knn",
        help="Use cosine KNN graphs or adj1.csv, adj2.csv, and adj3.csv.",
    )
    parser.add_argument("--knn-k", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--umap-seed", type=int, default=DEFAULT_UMAP_SEED)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.20)
    parser.add_argument(
        "--subset",
        choices=("all", "train", "test"),
        default="all",
        help="Patients shown and used to fit UMAP.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional class names in ascending numeric-label order.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_search_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    full_result = result.get("full_result", {})
    config = dict(full_result.get("config", {}))
    params = dict(result.get("params", full_result.get("params", {})))
    if not config or not params:
        raise ValueError(f"Could not recover model configuration from {path}")
    return config, params


def load_brca875(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int, int, int]:
    (
        x_full,
        y_full,
        total_features,
        num_omics,
        train_data,
        test_data,
        num_classes,
    ) = load_dataset(str(data_dir))

    omics_full = [
        np.vstack([train_data[i], test_data[i]]).astype(np.float32)
        for i in range(num_omics)
    ]
    num_train = int(train_data[0].shape[0])
    return (
        x_full.astype(np.float32),
        y_full.astype(np.int64),
        omics_full,
        total_features,
        num_classes,
        num_train,
    )


def build_graphs(
    omics_full: list[np.ndarray],
    data_dir: Path,
    graph_source: str,
    k: int,
) -> list[torch.Tensor]:
    if graph_source == "precomputed":
        return [
            load_precomputed_adjacency_csv(str(data_dir / f"adj{i + 1}.csv"))
            for i in range(len(omics_full))
        ]
    return [
        construct_adjacency(x, k=k, use_cache=True, method="knn_binary")
        for x in omics_full
    ]


def augmentation_parameters(
    config: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    keys = (
        "minority_drop_rate",
        "majority_drop_rate",
        "edge_add_prob",
        "max_new_edges",
        "use_edge_weights",
        "minority_edge_weight",
        "quality_weight_factor",
        "class_tier_severe_threshold",
        "class_tier_moderate_threshold",
        "class_tier_intermediate_threshold",
        "effective_num_beta",
    )
    defaults = {
        "minority_drop_rate": 0.0,
        "majority_drop_rate": 0.3,
        "edge_add_prob": 0.4,
        "max_new_edges": 100,
        "use_edge_weights": True,
        "minority_edge_weight": 2.0,
        "quality_weight_factor": 0.3,
        "class_tier_severe_threshold": 0.05,
        "class_tier_moderate_threshold": 0.15,
        "class_tier_intermediate_threshold": 0.30,
        "effective_num_beta": 0.99,
    }
    return {
        key: params.get(key, config.get(key, defaults[key]))
        for key in keys
    }


def prepare_model_inputs(
    x_full: np.ndarray,
    y_full: np.ndarray,
    edge_indices: list[torch.Tensor],
    num_train: int,
    config: dict[str, Any],
    params: dict[str, Any],
    view: str,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor] | None, dict]:
    if view == "original":
        return (
            torch.as_tensor(x_full, dtype=torch.float32),
            edge_indices,
            None,
            {"view": "original"},
        )

    train_mask = np.zeros(y_full.shape[0], dtype=bool)
    train_mask[:num_train] = True
    aug_params = augmentation_parameters(config, params)
    augmentor = DataAugmentor()
    x_aug, edges_aug, weights_aug, stats = augmentor.augment(
        x_full,
        y_full,
        edge_indices,
        train_mask,
        aug_params,
    )
    stats["view"] = "enhanced"
    stats["parameters"] = aug_params
    return x_aug.float(), edges_aug, weights_aug, stats


def load_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    params: dict[str, Any],
    input_dim: int,
    num_omics: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model = ModelFactory.create_model(
        model_type="gcn_contrastive",
        input_dim=input_dim,
        hidden_dim=int(params.get("hidden_dim", config.get("hidden_dim", 64))),
        num_omics=num_omics,
        num_classes=num_classes,
        gcn_layers=int(config.get("gcn_layers", 2)),
        embedding_dim=int(
            params.get("embedding_dim", config.get("embedding_dim", 64))
        ),
        temperature=float(
            params.get("temperature", config.get("temperature", 0.05))
        ),
        fusion_num_heads=int(
            params.get(
                "fusion_num_heads", config.get("fusion_num_heads", 2)
            )
        ),
        fusion_dropout=float(
            params.get("fusion_dropout", config.get("fusion_dropout", 0.2))
        ),
    ).to(device)

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "best_model_state"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint does not contain a model state dictionary.")

    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint and current model definition do not match.\n"
            f"Missing keys: {incompatible.missing_keys}\n"
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )
    model.eval()
    return model


def export_embeddings(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_indices: list[torch.Tensor],
    edge_weights: list[torch.Tensor] | None,
    device: torch.device,
) -> np.ndarray:
    x = x.to(device)
    edges = [edge.to(device) for edge in edge_indices]
    weights = (
        [weight.to(device) if weight is not None else None for weight in edge_weights]
        if edge_weights is not None
        else None
    )
    with torch.inference_mode():
        embeddings = model.get_classifier_embeddings(x, edges, weights)
    return embeddings.detach().cpu().numpy().astype(np.float32)


def fixed_early_fusion(omics_full: list[np.ndarray]) -> np.ndarray:
    normalized = []
    for matrix in omics_full:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized.append(matrix / np.clip(norms, 1e-12, None))
    return np.concatenate(normalized, axis=1) / np.sqrt(len(normalized))


def subset_mask(subset: str, num_samples: int, num_train: int) -> np.ndarray:
    mask = np.ones(num_samples, dtype=bool)
    if subset == "train":
        mask[num_train:] = False
    elif subset == "test":
        mask[:num_train] = False
    return mask


def run_umap(
    representation: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
    )
    return reducer.fit_transform(representation).astype(np.float32)


def representation_metrics(
    representation: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    return {
        "silhouette_cosine_source": float(
            silhouette_score(representation, labels, metric="cosine")
        ),
        "davies_bouldin_umap": float(
            davies_bouldin_score(coordinates, labels)
        ),
        "umap_trustworthiness_10": float(
            trustworthiness(
                representation,
                coordinates,
                n_neighbors=10,
                metric="cosine",
            )
        ),
    }


def class_labels(
    labels: np.ndarray, supplied_names: list[str] | None
) -> dict[int, str]:
    unique = sorted(int(value) for value in np.unique(labels))
    if supplied_names is not None:
        if len(supplied_names) != len(unique):
            raise ValueError(
                f"Expected {len(unique)} class names, got {len(supplied_names)}."
            )
        return dict(zip(unique, supplied_names))
    return {value: f"Class {value}" for value in unique}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def draw_figure(
    raw_coordinates: np.ndarray,
    model_coordinates: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    output_stem: Path,
    view: str,
) -> None:
    configure_matplotlib()

    # Colorblind-friendly palette with enough contrast for print and projection.
    palette = [
        "#3B6FB6",
        "#D8902F",
        "#3B9B7A",
        "#C95A5A",
        "#8B6BB1",
        "#4E9AA6",
        "#C6A53A",
        "#8B6A55",
    ]

    class_ids = sorted(names)
    class_counts = {
        class_id: int(np.sum(labels == class_id)) for class_id in class_ids
    }
    color_map = {
        class_id: palette[index % len(palette)]
        for index, class_id in enumerate(class_ids)
    }
    # Draw large classes first so minority-class patients remain visible.
    draw_order = sorted(class_ids, key=lambda value: class_counts[value], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), facecolor="white")
    panels = (
        (
            axes[0],
            raw_coordinates,
            "Early-fusion input",
            "a",
        ),
        (
            axes[1],
            model_coordinates,
            "TRIGEL discriminative representation",
            "b",
        ),
    )

    for axis, coordinates, title, panel_label in panels:
        axis.set_facecolor("#FCFCFC")
        for class_id in draw_order:
            class_mask = labels == class_id
            axis.scatter(
                coordinates[class_mask, 0],
                coordinates[class_mask, 1],
                s=16,
                color=color_map[class_id],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.22,
            )
        axis.set_title(title, loc="left", pad=8)
        axis.set_xlabel("UMAP 1", labelpad=3, color="#4A4A4A")
        axis.set_ylabel("UMAP 2", labelpad=3, color="#4A4A4A")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
        axis.margins(0.07)
        axis.text(
            -0.055,
            1.075,
            panel_label,
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="top",
            ha="left",
            color="#222222",
        )

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map[class_id],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=5.8,
        )
        for class_id in class_ids
    ]
    legend_labels = [
        f"{names[class_id]}  (n={class_counts[class_id]})"
        for class_id in class_ids
    ]
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=min(len(legend_labels), 5),
        columnspacing=1.35,
        handletextpad=0.45,
        borderaxespad=0,
    )
    fig.text(
        0.992,
        0.015,
        f"BRCA875 | {view} view | n={len(labels)}",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#777777",
    )
    fig.subplots_adjust(
        left=0.035,
        right=0.995,
        top=0.84,
        bottom=0.105,
        wspace=0.12,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for required_path in (args.data_dir, args.checkpoint, args.config_json):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    config, params = load_search_config(args.config_json)
    k = int(args.knn_k or config.get("cosine_k", 10))
    device = resolve_device(args.device)
    set_seed(args.model_seed)

    (
        x_full,
        y_full,
        omics_full,
        total_features,
        num_classes,
        num_train,
    ) = load_brca875(args.data_dir)
    edge_indices = build_graphs(
        omics_full,
        args.data_dir,
        args.graph_source,
        k,
    )
    x_model, model_edges, model_weights, augmentation_stats = (
        prepare_model_inputs(
            x_full,
            y_full,
            edge_indices,
            num_train,
            config,
            params,
            args.view,
        )
    )

    model = load_model(
        args.checkpoint,
        config,
        params,
        total_features,
        len(omics_full),
        num_classes,
        device,
    )
    embeddings = export_embeddings(
        model,
        x_model,
        model_edges,
        model_weights,
        device,
    )
    raw_features = fixed_early_fusion(omics_full)

    mask = subset_mask(args.subset, len(y_full), num_train)
    labels_plot = y_full[mask]
    raw_plot = raw_features[mask]
    embeddings_plot = embeddings[mask]

    raw_coordinates = run_umap(
        raw_plot,
        args.umap_seed,
        args.umap_neighbors,
        args.umap_min_dist,
    )
    model_coordinates = run_umap(
        embeddings_plot,
        args.umap_seed,
        args.umap_neighbors,
        args.umap_min_dist,
    )
    raw_metrics = representation_metrics(
        raw_plot, raw_coordinates, labels_plot
    )
    model_metrics = representation_metrics(
        embeddings_plot, model_coordinates, labels_plot
    )
    names = class_labels(labels_plot, args.class_names)

    output_stem = args.output_dir / (
        f"BRCA875_early_fusion_vs_TRIGEL_{args.view}_{args.subset}"
    )
    draw_figure(
        raw_coordinates,
        model_coordinates,
        labels_plot,
        names,
        output_stem,
        args.view,
    )

    sample_indices = np.flatnonzero(mask)
    split = np.where(sample_indices < num_train, "train", "test")
    source_data = pd.DataFrame(
        {
            "patient_index": sample_indices,
            "split": split,
            "class_id": labels_plot,
            "class_name": [names[int(value)] for value in labels_plot],
            "early_fusion_umap1": raw_coordinates[:, 0],
            "early_fusion_umap2": raw_coordinates[:, 1],
            "trigel_umap1": model_coordinates[:, 0],
            "trigel_umap2": model_coordinates[:, 1],
        }
    )
    source_data.to_csv(
        args.output_dir / "BRCA875_umap_source_data.csv", index=False
    )
    np.save(args.output_dir / "BRCA875_TRIGEL_embeddings.npy", embeddings)
    np.save(args.output_dir / "BRCA875_fixed_early_fusion.npy", raw_features)
    np.savez_compressed(
        args.output_dir / "BRCA875_umap_coordinates.npz",
        patient_index=sample_indices,
        labels=labels_plot,
        early_fusion=raw_coordinates,
        trigel=model_coordinates,
    )

    report = {
        "data_dir": str(args.data_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "config_json": str(args.config_json.resolve()),
        "device": str(device),
        "view": args.view,
        "graph_source": args.graph_source,
        "knn_k": k,
        "model_seed": args.model_seed,
        "umap": {
            "random_state": args.umap_seed,
            "n_neighbors": args.umap_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": "cosine",
            "subset": args.subset,
        },
        "samples": int(mask.sum()),
        "num_train": num_train,
        "num_test": int(len(y_full) - num_train),
        "class_counts": {
            names[int(class_id)]: int(np.sum(labels_plot == class_id))
            for class_id in sorted(np.unique(labels_plot))
        },
        "early_fusion_metrics": raw_metrics,
        "trigel_metrics": model_metrics,
        "augmentation": augmentation_stats,
    }
    with (args.output_dir / "BRCA875_umap_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_ready(report), handle, indent=2, ensure_ascii=False)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"TRIGEL embeddings: {embeddings.shape}")
    print(f"Figure and source data saved to: {args.output_dir}")
    print(
        "Source-space silhouette (cosine): "
        f"early fusion={raw_metrics['silhouette_cosine_source']:.4f}, "
        f"TRIGEL={model_metrics['silhouette_cosine_source']:.4f}"
    )


if __name__ == "__main__":
    main()

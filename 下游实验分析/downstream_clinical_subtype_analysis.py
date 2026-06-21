#!/usr/bin/env python
"""Downstream clinical analyses for cancer subtype classification results.

This script covers two downstream analyses used in subtype-classification
papers:

1. Kaplan-Meier survival analysis across predicted subtypes.
2. Clinicopathological association analysis across predicted subtypes.

It intentionally does not run model ablations. Provide one table containing
sample-level subtype assignments and one clinical table containing survival
and/or clinicopathological variables.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency: matplotlib. Install it in the project environment "
        "before drawing downstream clinical figures."
    ) from exc

try:
    from scipy.stats import chi2, chi2_contingency, kruskal
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency: scipy. Install scipy for log-rank, chi-square, "
        "and Kruskal-Wallis tests."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SUBTYPES = (
    PROJECT_ROOT
    / "results"
    / "BRCA_hyperparameter_search_20260118_202538"
    / "best_result"
    / "brca875_umap"
    / "BRCA875_umap_source_data.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "downstream_clinical_analysis"

ID_CANDIDATES = (
    "sample_id",
    "patient_id",
    "case_id",
    "barcode",
    "bcr_patient_barcode",
    "submitter_id",
    "patient",
    "patient_index",
)
TIME_CANDIDATES = (
    "OS.time",
    "OS_time",
    "os_time",
    "overall_survival_time",
    "survival_time",
    "days_to_death",
    "days_to_last_follow_up",
    "time",
)
EVENT_CANDIDATES = (
    "OS",
    "OS.event",
    "OS_event",
    "os_event",
    "vital_status",
    "event",
    "status",
    "death",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run downstream survival and clinicopathological association "
            "analysis for cancer subtype classification results."
        )
    )
    parser.add_argument(
        "--subtypes",
        type=Path,
        default=DEFAULT_SUBTYPES,
        help=(
            "CSV/TSV containing sample-level subtype assignments. The default "
            "BRCA875 UMAP source table contains true class_id; for predicted "
            "subtypes, pass your prediction file here."
        ),
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        required=True,
        help=(
            "CSV/TSV clinical table containing a sample/patient id column and "
            "survival or clinicopathological variables."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory. A timestamped subdirectory will be created "
            "under this path for each run."
        ),
    )
    parser.add_argument(
        "--subtype-col",
        default="class_id",
        help=(
            "Subtype column in --subtypes. Use your predicted-label column "
            "when available, for example pred_subtype or y_pred."
        ),
    )
    parser.add_argument("--subtype-id-col", default=None)
    parser.add_argument("--clinical-id-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--event-col", default=None)
    parser.add_argument(
        "--clinical-vars",
        nargs="*",
        default=None,
        help=(
            "Clinicopathological variables to test. If omitted, variables are "
            "inferred from the clinical table."
        ),
    )
    parser.add_argument(
        "--subtype-names",
        nargs="*",
        default=None,
        help=(
            "Optional names in numeric subtype order, e.g. Normal-like "
            "Basal-like HER2-enriched Luminal-A Luminal-B."
        ),
    )
    parser.add_argument(
        "--time-scale",
        choices=("days", "months", "years"),
        default="days",
        help="Scale used for the survival-time axis.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum non-missing values required for testing a variable.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=12,
        help="Maximum categories for categorical association plots.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def create_timestamped_output_dir(base_dir: Path) -> Path:
    """Create and return a timestamped run directory under base_dir."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in column_set:
            return column_set[candidate.lower()]
    return None


def normalize_sample_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def subtype_label(value: object, subtype_names: list[str] | None) -> str:
    if subtype_names is None:
        return f"Subtype {value}"
    try:
        idx = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    if 0 <= idx < len(subtype_names):
        return subtype_names[idx]
    return f"Subtype {value}"


def prepare_merged_data(args: argparse.Namespace) -> pd.DataFrame:
    subtypes = read_table(args.subtypes)
    clinical = read_table(args.clinical)

    subtype_id_col = args.subtype_id_col or first_existing(
        subtypes.columns, ID_CANDIDATES
    )
    clinical_id_col = args.clinical_id_col or first_existing(
        clinical.columns, ID_CANDIDATES
    )

    if subtype_id_col is None:
        raise ValueError(
            "Could not infer sample id column in --subtypes. Pass "
            "--subtype-id-col explicitly."
        )
    if clinical_id_col is None:
        raise ValueError(
            "Could not infer sample id column in --clinical. Pass "
            "--clinical-id-col explicitly."
        )
    if args.subtype_col not in subtypes.columns:
        raise ValueError(
            f"Subtype column '{args.subtype_col}' was not found in "
            f"{args.subtypes}. Available columns: {list(subtypes.columns)}"
        )

    subtypes = subtypes.copy()
    clinical = clinical.copy()
    subtypes["_merge_id"] = subtypes[subtype_id_col].map(normalize_sample_id)
    clinical["_merge_id"] = clinical[clinical_id_col].map(normalize_sample_id)
    subtypes["_subtype_raw"] = subtypes[args.subtype_col]
    subtypes["subtype"] = subtypes["_subtype_raw"].map(
        lambda x: subtype_label(x, args.subtype_names)
    )

    merged = clinical.merge(
        subtypes[["_merge_id", "_subtype_raw", "subtype"]],
        on="_merge_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            "Clinical and subtype tables did not share any sample ids. Check "
            "--subtype-id-col, --clinical-id-col, and barcode formatting."
        )
    return merged


def coerce_event(series: pd.Series) -> pd.Series:
    mapping = {
        "dead": 1,
        "deceased": 1,
        "death": 1,
        "1": 1,
        "true": 1,
        "yes": 1,
        "alive": 0,
        "living": 0,
        "censored": 0,
        "0": 0,
        "false": 0,
        "no": 0,
    }
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= series.notna().sum() * 0.8:
        return numeric
    return series.astype(str).str.strip().str.lower().map(mapping)


def scale_time(time: pd.Series, scale: str) -> pd.Series:
    if scale == "months":
        return time / 30.4375
    if scale == "years":
        return time / 365.25
    return time


def km_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(time)
    time = time[order]
    event = event[order]
    event_times = np.unique(time[event == 1])
    x = [0.0]
    y = [1.0]
    survival = 1.0
    for t in event_times:
        at_risk = np.sum(time >= t)
        observed = np.sum((time == t) & (event == 1))
        if at_risk > 0:
            survival *= 1.0 - observed / at_risk
        x.extend([float(t), float(t)])
        y.extend([y[-1], survival])
    if len(time):
        x.append(float(np.max(time)))
        y.append(y[-1])
    return np.asarray(x), np.asarray(y)


def multigroup_logrank(df: pd.DataFrame, time_col: str, event_col: str) -> dict[str, float]:
    groups = list(df["subtype"].dropna().unique())
    event_times = np.sort(df.loc[df[event_col] == 1, time_col].dropna().unique())
    k = len(groups)
    observed = np.zeros(k)
    expected = np.zeros(k)
    variance = np.zeros((k, k))

    for t in event_times:
        at_risk = np.array(
            [np.sum((df["subtype"] == g) & (df[time_col] >= t)) for g in groups],
            dtype=float,
        )
        events = np.array(
            [
                np.sum(
                    (df["subtype"] == g)
                    & (df[time_col] == t)
                    & (df[event_col] == 1)
                )
                for g in groups
            ],
            dtype=float,
        )
        total_risk = at_risk.sum()
        total_events = events.sum()
        if total_risk <= 1 or total_events == 0:
            continue
        observed += events
        expected += total_events * at_risk / total_risk
        factor = total_events * (total_risk - total_events) / (
            total_risk * total_risk * (total_risk - 1)
        )
        variance += factor * (
            np.diag(at_risk * (total_risk - at_risk))
            - np.outer(at_risk, at_risk)
            + np.diag(at_risk * at_risk)
        )

    diff = observed[:-1] - expected[:-1]
    variance_reduced = variance[:-1, :-1]
    stat = float(diff.T @ np.linalg.pinv(variance_reduced) @ diff)
    dof = max(k - 1, 1)
    p_value = float(chi2.sf(stat, dof))
    return {"logrank_chisq": stat, "logrank_df": dof, "logrank_p": p_value}


def plot_survival(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    output_dir: Path,
    time_scale: str,
) -> pd.DataFrame:
    surv = df[["_merge_id", "subtype", time_col, event_col]].copy()
    surv[time_col] = pd.to_numeric(surv[time_col], errors="coerce")
    surv[event_col] = coerce_event(surv[event_col])
    surv[time_col] = scale_time(surv[time_col], time_scale)
    surv = surv.dropna(subset=["subtype", time_col, event_col])
    surv = surv[(surv[time_col] >= 0) & surv[event_col].isin([0, 1])]

    if surv["subtype"].nunique() < 2:
        raise ValueError("Survival analysis requires at least two subtypes.")

    stats = multigroup_logrank(surv, time_col, event_col)
    summary_rows = []

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for subtype, group in surv.groupby("subtype", sort=True):
        time = group[time_col].to_numpy(dtype=float)
        event = group[event_col].to_numpy(dtype=int)
        x, y = km_curve(time, event)
        ax.step(x, y, where="post", linewidth=2.0, label=f"{subtype} (n={len(group)})")
        summary_rows.append(
            {
                "subtype": subtype,
                "n": len(group),
                "events": int(event.sum()),
                "censored": int((event == 0).sum()),
                "median_time": float(np.median(time)) if len(time) else math.nan,
            }
        )

    ax.set_xlabel(f"Time ({time_scale})")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        "Kaplan-Meier survival curves by predicted subtype\n"
        f"log-rank p = {stats['logrank_p']:.3g}"
    )
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"kaplan_meier_by_subtype.{ext}", dpi=300)
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    for key, value in stats.items():
        summary[key] = value
    summary.to_csv(output_dir / "survival_summary.csv", index=False)
    surv.to_csv(output_dir / "survival_merged_data.csv", index=False)
    return summary


def is_categorical(series: pd.Series, max_categories: int) -> bool:
    clean = series.dropna()
    if clean.empty:
        return False
    if not pd.api.types.is_numeric_dtype(clean):
        return True
    return clean.nunique() <= max_categories


def cramers_v(table: pd.DataFrame, chi2_stat: float) -> float:
    n = table.to_numpy().sum()
    if n == 0:
        return math.nan
    r, c = table.shape
    denom = n * max(min(r - 1, c - 1), 1)
    return float(math.sqrt(chi2_stat / denom))


def adjusted_residuals(table: pd.DataFrame) -> pd.DataFrame:
    values = table.to_numpy(dtype=float)
    total = values.sum()
    row_prop = values.sum(axis=1, keepdims=True) / total
    col_prop = values.sum(axis=0, keepdims=True) / total
    expected = row_prop @ col_prop * total
    denom = np.sqrt(expected * (1 - row_prop) * (1 - col_prop))
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = (values - expected) / denom
    residuals[~np.isfinite(residuals)] = 0
    return pd.DataFrame(residuals, index=table.index, columns=table.columns)


def plot_categorical_variable(
    df: pd.DataFrame, var: str, output_dir: Path
) -> dict[str, object]:
    data = df[["subtype", var]].dropna()
    table = pd.crosstab(data["subtype"], data[var])
    chi2_stat, p_value, dof, _ = chi2_contingency(table)
    residuals = adjusted_residuals(table)
    proportions = table.div(table.sum(axis=1), axis=0).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    proportions.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        width=0.82,
        colormap="tab20",
    )
    axes[0].set_xlabel("Subtype")
    axes[0].set_ylabel("Proportion")
    axes[0].set_title(f"{var} distribution")
    axes[0].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")

    im = axes[1].imshow(residuals.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-3, vmax=3)
    axes[1].set_xticks(range(residuals.shape[1]))
    axes[1].set_xticklabels(residuals.columns, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(residuals.shape[0]))
    axes[1].set_yticklabels(residuals.index, fontsize=8)
    axes[1].set_title("Adjusted residuals")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f"{var}: chi-square p = {p_value:.3g}", y=1.02)
    fig.tight_layout()
    safe_var = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in var)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"clinical_categorical_{safe_var}.{ext}", dpi=300)
    plt.close(fig)

    table.to_csv(output_dir / f"clinical_categorical_{safe_var}_counts.csv")
    residuals.to_csv(output_dir / f"clinical_categorical_{safe_var}_residuals.csv")
    return {
        "variable": var,
        "type": "categorical",
        "n": int(len(data)),
        "levels": int(table.shape[1]),
        "test": "chi-square",
        "statistic": float(chi2_stat),
        "df": int(dof),
        "p_value": float(p_value),
        "effect_size": cramers_v(table, chi2_stat),
    }


def plot_numeric_variable(
    df: pd.DataFrame, var: str, output_dir: Path
) -> dict[str, object]:
    data = df[["subtype", var]].dropna().copy()
    data[var] = pd.to_numeric(data[var], errors="coerce")
    data = data.dropna()
    groups = [g[var].to_numpy(dtype=float) for _, g in data.groupby("subtype", sort=True)]
    labels = [str(label) for label, _ in data.groupby("subtype", sort=True)]
    stat, p_value = kruskal(*groups)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True)
    jitter_rng = np.random.default_rng(2026)
    for i, values in enumerate(groups, start=1):
        x = jitter_rng.normal(i, 0.04, size=len(values))
        ax.scatter(x, values, s=18, alpha=0.55)
    ax.set_xlabel("Subtype")
    ax.set_ylabel(var)
    ax.set_title(f"{var}: Kruskal-Wallis p = {p_value:.3g}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    safe_var = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in var)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"clinical_numeric_{safe_var}.{ext}", dpi=300)
    plt.close(fig)

    data.groupby("subtype")[var].describe().to_csv(
        output_dir / f"clinical_numeric_{safe_var}_summary.csv"
    )
    return {
        "variable": var,
        "type": "numeric",
        "n": int(len(data)),
        "levels": len(groups),
        "test": "Kruskal-Wallis",
        "statistic": float(stat),
        "df": len(groups) - 1,
        "p_value": float(p_value),
        "effect_size": math.nan,
    }


def infer_clinical_vars(
    df: pd.DataFrame,
    args: argparse.Namespace,
    time_col: str | None,
    event_col: str | None,
) -> list[str]:
    excluded = {"_merge_id", "_subtype_raw", "subtype", time_col, event_col}
    candidates = [
        col for col in df.columns if col not in excluded and not col.startswith("_")
    ]
    selected = []
    for col in candidates:
        clean = df[col].dropna()
        if len(clean) < args.min_count:
            continue
        if is_categorical(clean, args.max_categories):
            if clean.nunique() >= 2:
                selected.append(col)
        else:
            numeric = pd.to_numeric(clean, errors="coerce")
            if numeric.notna().sum() >= args.min_count:
                selected.append(col)
    return selected


def run_clinical_association(
    df: pd.DataFrame,
    args: argparse.Namespace,
    time_col: str | None,
    event_col: str | None,
) -> pd.DataFrame:
    variables = args.clinical_vars or infer_clinical_vars(df, args, time_col, event_col)
    if not variables:
        raise ValueError(
            "No clinicopathological variables were selected. Pass "
            "--clinical-vars explicitly."
        )

    rows = []
    for var in variables:
        if var not in df.columns:
            print(f"[skip] {var}: not found in clinical table", file=sys.stderr)
            continue
        clean = df[["subtype", var]].dropna()
        if len(clean) < args.min_count or clean["subtype"].nunique() < 2:
            print(f"[skip] {var}: insufficient non-missing values", file=sys.stderr)
            continue
        if is_categorical(clean[var], args.max_categories):
            if clean[var].nunique() < 2:
                continue
            rows.append(plot_categorical_variable(df, var, args.output_dir))
        else:
            rows.append(plot_numeric_variable(df, var, args.output_dir))

    summary = pd.DataFrame(rows).sort_values("p_value", na_position="last")
    summary.to_csv(args.output_dir / "clinical_association_summary.csv", index=False)
    return summary


def write_readme(
    output_dir: Path,
    args: argparse.Namespace,
    merged: pd.DataFrame,
    survival_summary: pd.DataFrame | None,
    clinical_summary: pd.DataFrame | None,
) -> None:
    lines = [
        "# Downstream clinical subtype analysis",
        "",
        f"- Subtype file: `{args.subtypes}`",
        f"- Clinical file: `{args.clinical}`",
        f"- Matched samples: {len(merged)}",
        f"- Subtype column: `{args.subtype_col}`",
        "",
    ]
    if survival_summary is not None:
        p = survival_summary["logrank_p"].iloc[0]
        lines.extend(
            [
                "## Survival analysis",
                "",
                f"- Global log-rank p-value: {p:.4g}",
                "- Figure: `kaplan_meier_by_subtype.png`",
                "- Table: `survival_summary.csv`",
                "",
            ]
        )
    if clinical_summary is not None and not clinical_summary.empty:
        top = clinical_summary.iloc[0]
        lines.extend(
            [
                "## Clinicopathological association",
                "",
                f"- Tested variables: {len(clinical_summary)}",
                f"- Strongest association: `{top['variable']}` "
                f"(p = {top['p_value']:.4g})",
                "- Table: `clinical_association_summary.csv`",
                "",
            ]
        )
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_output_dir = args.output_dir
    args.output_dir = create_timestamped_output_dir(base_output_dir)
    merged = prepare_merged_data(args)
    merged.to_csv(args.output_dir / "merged_subtype_clinical_data.csv", index=False)

    time_col = args.time_col or first_existing(merged.columns, TIME_CANDIDATES)
    event_col = args.event_col or first_existing(merged.columns, EVENT_CANDIDATES)

    survival_summary = None
    if time_col and event_col:
        survival_summary = plot_survival(
            merged,
            time_col=time_col,
            event_col=event_col,
            output_dir=args.output_dir,
            time_scale=args.time_scale,
        )
    else:
        print(
            "[info] Survival analysis skipped because time/event columns were "
            "not found. Pass --time-col and --event-col if they exist.",
            file=sys.stderr,
        )

    clinical_summary = run_clinical_association(merged, args, time_col, event_col)
    write_readme(args.output_dir, args, merged, survival_summary, clinical_summary)
    print(f"Downstream clinical analysis completed: {args.output_dir}")


if __name__ == "__main__":
    main()

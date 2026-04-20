from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import rankdata

from .config import REPO_ROOT
from .significance import load_run_metrics

MODEL_COLORS: dict[str, str] = {
    "HybriTE": "#1F77B4",
    "RiboNN": "#F28E2B",
    "Saluki": "#59A14F",
    "LightGBM": "#FF7F0E",
    "No structure": "#9467BD",
    "No biochemistry": "#D62728",
    "Mouse full": "#1F77B4",
    "Mouse no biochemistry": "#D62728",
    "Human→Mouse transfer": "#8C564B",
    "Human -Bio→Mouse -Bio transfer": "#7F7F7F",
    "Threshold 0.001": "#1F77B4",
    "Threshold 0.01": "#FF7F0E",
    "Threshold 0.1": "#2CA02C",
    "4/16/8 bins": "#FF7F0E",
    "8/32/16 bins": "#1F77B4",
    "12/48/24 bins": "#2CA02C",
    "Sequence": "#4C78A8",
    "Structure": "#72B7B2",
    "Long-range": "#9D755D",
    "ENCORI": "#4C78A8",
    "eCLIP": "#F28E2B",
    "M6ACLIP": "#59A14F",
}

POSITIVE_COLOR = "#1F77B4"
NEGATIVE_COLOR = "#D62728"
NEUTRAL_COLOR = "#4D4D4D"


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "grid.color": "#DDDDDD",
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
        }
    )


def panel_label(ax: Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def ensure_columns(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    source_name: str,
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}. "
            f"Available columns: {frame.columns.tolist()}"
        )


def first_existing_path(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def require_paths(paths: Sequence[Path], *, description: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required {description} paths:\n- " + "\n- ".join(missing)
        )


def output_root() -> Path:
    path = REPO_ROOT / "artifacts" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_data_root() -> Path:
    path = output_root() / "source_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: Figure, stem: str, source_data: pd.DataFrame | None) -> None:
    figure_dir = output_root()
    figure_path_pdf = figure_dir / f"{stem}.pdf"
    figure_path_png = figure_dir / f"{stem}.png"
    fig.savefig(figure_path_pdf)
    fig.savefig(figure_path_png, dpi=400)
    if source_data is not None:
        source_data.to_csv(source_data_root() / f"{stem}.csv", index=False)


def load_metric_frame(
    run_dir: str | Path,
    *,
    split_name: str = "test",
    scope: str = "mean_te",
    metric: str = "pearson",
) -> pd.DataFrame:
    run_path = Path(run_dir).expanduser().resolve()
    try:
        frame = load_run_metrics(run_path, split_name)
    except ValueError as exc:
        raise FileNotFoundError(
            f"No fold metrics found under {run_path}. Expected fold-level "
            f"{split_name}/aggregate_metrics.csv files or cross_validation_metrics.csv."
        ) from exc
    ensure_columns(
        frame, ["fold", "scope", "metric", "value"], source_name=str(run_path)
    )
    filtered = pd.DataFrame(
        frame.loc[
            (frame["scope"] == scope) & (frame["metric"] == metric),
            ["fold", "value"],
        ]
    ).copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found for scope={scope!r}, metric={metric!r} under {run_path}"
        )
    filtered["fold"] = filtered["fold"].astype(int)
    filtered["value"] = filtered["value"].astype(float)
    return filtered.sort_values("fold", kind="stable").reset_index(drop=True)


def paired_delta_frame(
    candidate_dir: str | Path,
    reference_dir: str | Path,
    *,
    split_name: str = "test",
    scope: str = "mean_te",
    metric: str = "pearson",
) -> pd.DataFrame:
    candidate = load_metric_frame(
        candidate_dir,
        split_name=split_name,
        scope=scope,
        metric=metric,
    ).rename(columns={"value": "candidate_value"})
    reference = load_metric_frame(
        reference_dir,
        split_name=split_name,
        scope=scope,
        metric=metric,
    ).rename(columns={"value": "reference_value"})
    merged = candidate.merge(reference, on="fold", how="inner")
    if merged.empty:
        raise ValueError(
            f"No shared folds between {Path(candidate_dir)} and {Path(reference_dir)}"
        )
    merged["delta"] = merged["candidate_value"] - merged["reference_value"]
    return merged.sort_values("fold", kind="stable").reset_index(drop=True)


def load_significance_row(
    summary_paths: Sequence[Path],
    *,
    scope: str = "mean_te",
    metric: str = "pearson",
) -> pd.Series | None:
    path = first_existing_path(summary_paths)
    if path is None:
        return None
    frame = pd.read_csv(path)
    ensure_columns(frame, ["scope", "metric"], source_name=str(path))
    row = frame[(frame["scope"] == scope) & (frame["metric"] == metric)]
    if row.empty:
        return None
    return row.iloc[0]


def load_per_target_metric_frame(
    run_dir: str | Path,
    *,
    split_name: str = "test",
    metric: str = "pearson",
) -> pd.DataFrame:
    run_path = Path(run_dir).expanduser().resolve()
    rows: list[pd.DataFrame] = []
    for fold_dir in sorted(run_path.glob("fold_*")):
        metric_path = fold_dir / split_name / "per_target_metrics.csv"
        if not metric_path.exists():
            continue
        metric_frame = pd.DataFrame(pd.read_csv(metric_path))
        ensure_columns(
            metric_frame, ["target_name", metric], source_name=str(metric_path)
        )
        fold_frame = pd.DataFrame(metric_frame.loc[:, ["target_name", metric]]).copy()
        fold_frame.insert(0, "fold", int(fold_dir.name.split("_")[-1]))
        rows.append(fold_frame)
    if not rows:
        raise FileNotFoundError(
            f"No fold-level per_target_metrics.csv files found under {run_path}"
        )
    return pd.concat(rows, ignore_index=True)


def aggregate_per_target_metric(
    run_dir: str | Path,
    *,
    split_name: str = "test",
    metric: str = "pearson",
) -> pd.DataFrame:
    frame = load_per_target_metric_frame(run_dir, split_name=split_name, metric=metric)
    aggregated = pd.DataFrame(
        frame.groupby("target_name", as_index=False).agg(value=(metric, "mean"))
    )
    return aggregated.sort_values(by="target_name").reset_index(drop=True)


def load_oof_mean_te_frame(
    run_dir: str | Path,
    *,
    split_name: str = "test",
) -> pd.DataFrame:
    run_path = Path(run_dir).expanduser().resolve()
    rows: list[pd.DataFrame] = []
    for fold_dir in sorted(run_path.glob("fold_*")):
        mean_te_path = fold_dir / split_name / "mean_te.csv"
        if not mean_te_path.exists():
            continue
        frame = pd.DataFrame(pd.read_csv(mean_te_path))
        ensure_columns(
            frame,
            ["target_mean_te", "prediction_mean_te"],
            source_name=str(mean_te_path),
        )
        frame = frame.copy()
        frame.insert(0, "fold", int(fold_dir.name.split("_")[-1]))
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No mean_te.csv files found under {run_path}")
    combined = pd.concat(rows, ignore_index=True)
    id_columns = [
        column
        for column in combined.columns
        if column not in {"fold", "target_mean_te", "prediction_mean_te"}
    ]
    if id_columns:
        combined = combined.drop_duplicates(subset=id_columns, keep="first")
    return combined


def load_transfer_metric_frame(
    run_dir: str | Path,
    *,
    scope: str = "mean_te",
    metric: str = "pearson",
) -> pd.DataFrame:
    run_path = Path(run_dir).expanduser().resolve()
    rows: list[pd.DataFrame] = []
    candidate_paths: list[Path] = []
    candidate_paths.extend(sorted(run_path.glob("fold_*/aggregate_metrics.csv")))
    candidate_paths.extend(
        sorted(run_path.glob("fold_*/cross_species/*/aggregate_metrics.csv"))
    )
    candidate_paths.extend(
        sorted(run_path.glob("cross_species/fold_*/aggregate_metrics.csv"))
    )
    candidate_paths.extend(
        sorted(run_path.glob("cross_species/*/aggregate_metrics.csv"))
    )
    candidate_paths.extend(sorted(run_path.glob("*/aggregate_metrics.csv")))

    seen_paths: set[Path] = set()
    for metric_path in candidate_paths:
        if metric_path in seen_paths:
            continue
        seen_paths.add(metric_path)
        metric_frame = pd.DataFrame(pd.read_csv(metric_path))
        ensure_columns(
            metric_frame, ["scope", "metric", "value"], source_name=str(metric_path)
        )
        fold_frame = pd.DataFrame(
            metric_frame.loc[
                (metric_frame["scope"] == scope) & (metric_frame["metric"] == metric),
                ["value"],
            ]
        ).copy()
        if fold_frame.empty:
            continue
        fold_number = _extract_fold_number(metric_path)
        fold_frame.insert(0, "fold", fold_number)
        rows.append(fold_frame)
    if not rows:
        raise FileNotFoundError(
            f"No transfer aggregate_metrics.csv files found under {run_path}"
        )
    combined = pd.concat(rows, ignore_index=True)
    combined["value"] = combined["value"].astype(float)
    return combined.sort_values("fold", kind="stable").reset_index(drop=True)


def _extract_fold_number(path: Path) -> int:
    for candidate in [
        path.name,
        path.parent.name,
        path.parent.parent.name,
        path.parent.parent.parent.name,
    ]:
        match = re.search(r"fold_(\d+)", candidate)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"Could not infer fold number from transfer path: {path}")


def mean_confidence_interval(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean, mean
    sem = float(array.std(ddof=1) / np.sqrt(array.size))
    delta = 1.96 * sem
    return mean, mean - delta, mean + delta


def plot_grouped_fold_points(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
    order: Sequence[str],
    ylabel: str,
    title: str,
) -> None:
    ensure_columns(frame, [group_column, value_column], source_name="plot frame")
    for index, group in enumerate(order):
        group_values = frame.loc[frame[group_column] == group, value_column].to_numpy(
            dtype=np.float64
        )
        if group_values.size == 0:
            continue
        if group_values.size == 1:
            offsets = np.asarray([0.0])
        else:
            offsets = np.linspace(-0.12, 0.12, group_values.size)
        x_values = np.full(group_values.size, index, dtype=np.float64) + offsets
        color = MODEL_COLORS.get(group, POSITIVE_COLOR)
        ax.scatter(
            x_values,
            group_values,
            s=28,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        mean, lower, upper = mean_confidence_interval(group_values)
        ax.vlines(index, lower, upper, color="black", linewidth=1.2, zorder=4)
        ax.scatter(
            [index],
            [mean],
            s=42,
            marker="D",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
    ax.set_xticks(range(len(order)), order)
    long_labels = any(len(str(label)) > 14 for label in order)
    if long_labels:
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.6)


def plot_sorted_delta_profile(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    delta_column: str,
    label_column: str,
    title: str,
    ylabel: str,
    label_count: int = 3,
) -> pd.DataFrame:
    ensure_columns(frame, [delta_column, label_column], source_name="delta frame")
    ordered = frame.sort_values(
        delta_column, ascending=False, kind="stable"
    ).reset_index(drop=True)
    x_values = np.arange(len(ordered), dtype=np.int64)
    colors = np.where(
        ordered[delta_column].to_numpy(dtype=np.float64) >= 0.0,
        POSITIVE_COLOR,
        NEGATIVE_COLOR,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.vlines(x_values, 0.0, ordered[delta_column], color=colors, linewidth=1.0)
    ax.scatter(
        x_values,
        ordered[delta_column],
        color=colors,
        s=18,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Biological source rank")
    ax.grid(axis="y", alpha=0.6)
    ax.set_xlim(-1, len(ordered))
    ax.set_xticks([])

    if len(ordered) > 0:
        label_indices = list(range(min(label_count, len(ordered))))
        label_indices.extend(
            list(range(max(0, len(ordered) - label_count), len(ordered)))
        )
        seen: set[int] = set()
        y_span = max(
            float(ordered[delta_column].max() - ordered[delta_column].min()), 1e-3
        )
        for index in label_indices:
            if index in seen:
                continue
            seen.add(index)
            row = ordered.iloc[index]
            value = float(row[delta_column])
            offset = 0.04 * y_span if value >= 0.0 else -0.04 * y_span
            va = "bottom" if value >= 0.0 else "top"
            ax.text(
                index,
                value + offset,
                pretty_target_name(str(row[label_column])),
                fontsize=7,
                rotation=30,
                ha="center",
                va=va,
            )
    return ordered


def pretty_target_name(name: str) -> str:
    cleaned = name
    if cleaned.startswith("bio_source_"):
        cleaned = cleaned.removeprefix("bio_source_")
    cleaned = cleaned.replace("_", " ")
    return cleaned


def scatter_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    ensure_columns(
        frame,
        ["target_mean_te", "prediction_mean_te"],
        source_name="scatter frame",
    )
    target = np.asarray(frame["target_mean_te"], dtype=float)
    prediction = np.asarray(frame["prediction_mean_te"], dtype=float)
    if target.size < 2:
        return float("nan"), float("nan")
    pearson_value = float(np.corrcoef(target, prediction)[0, 1])
    spearman_value = float(np.corrcoef(rankdata(target), rankdata(prediction))[0, 1])
    return pearson_value, spearman_value


def format_p_value(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "p unavailable"
    if value < 1.0e-4:
        return "p < 1e-4"
    if value < 1.0e-3:
        return f"p = {value:.1e}"
    return f"p = {value:.3f}"


def load_optional_table(
    paths: Sequence[Path], *, description: str
) -> pd.DataFrame | None:
    path = first_existing_path(paths)
    if path is None:
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{description} exists but is empty: {path}")
    return frame

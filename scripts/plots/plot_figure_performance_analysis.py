# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import _plot_bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score

from hybrite.config import REPO_ROOT
from hybrite.results_figures import (
    aggregate_per_target_metric,
    load_metric_frame,
    load_oof_mean_te_frame,
    load_per_target_metric_frame,
    load_transfer_metric_frame,
    pretty_target_name,
    save_figure,
    scatter_metrics,
)

MM_TO_INCH = 1 / 25.4

BLUE = "#002147"
LIGHT_BLUE = "#00509E"
VERMILION = "#D55E00"
TEAL = "#009E73"
PURPLE = "#7B5AA6"
AMBER = "#E69F00"
GREY = "#666666"
MISSING_FACE = "#F5F5F5"
TRANSFER_WITHIN_COLOR = BLUE
TRANSFER_CROSS_COLOR = VERMILION


@dataclass(frozen=True)
class ModelRun:
    key: str
    label: str
    color: str
    run_dir: Path
    transfer_run_dir: Path | None = None


@dataclass(frozen=True)
class AblationRun:
    key: str
    label: str
    color: str
    run_dir: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", default="figure_performance_analysis")
    return parser.parse_args()


def apply_legacy_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def add_panel_label(ax: Axes, label: str) -> None:
    ax.text(
        -0.2,
        1.1,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def benchmark_runs(species: str) -> tuple[ModelRun, ...]:
    results_root = REPO_ROOT / "results"
    return (
        ModelRun("hybrite", "HybriTE", LIGHT_BLUE, results_root / species),
        ModelRun(
            "ribonn",
            "RiboNN",
            VERMILION,
            results_root / f"{species}_ribonn",
        ),
        ModelRun(
            "saluki",
            "Saluki",
            TEAL,
            results_root / f"{species}_saluki",
        ),
    )


def transfer_runs(species: str) -> tuple[ModelRun, ...]:
    results_root = REPO_ROOT / "results"
    if species == "human":
        return (
            ModelRun(
                "hybrite",
                "HybriTE",
                LIGHT_BLUE,
                results_root / "human",
                results_root / "human",
            ),
            ModelRun(
                "ribonn",
                "RiboNN",
                VERMILION,
                results_root / "human_ribonn",
                results_root / "human_ribonn",
            ),
            ModelRun(
                "saluki",
                "Saluki",
                TEAL,
                results_root / "human_saluki",
                results_root / "human_saluki",
            ),
            ModelRun(
                "lightgbm",
                "LightGBM",
                GREY,
                results_root / "human_lightgbm",
                results_root / "human_lightgbm_to_mouse_lightgbm_transfer",
            ),
        )
    return (
        ModelRun(
            "hybrite",
            "HybriTE",
            LIGHT_BLUE,
            results_root / "mouse",
            results_root / "mouse",
        ),
        ModelRun(
            "ribonn",
            "RiboNN",
            VERMILION,
            results_root / "mouse_ribonn",
            results_root / "mouse_ribonn",
        ),
        ModelRun(
            "saluki",
            "Saluki",
            TEAL,
            results_root / "mouse_saluki",
            results_root / "mouse_saluki",
        ),
        ModelRun(
            "lightgbm",
            "LightGBM",
            GREY,
            results_root / "mouse_lightgbm",
            results_root / "mouse_lightgbm_to_human_lightgbm_transfer",
        ),
    )


def ablation_runs(species: str) -> tuple[AblationRun, ...]:
    results_root = REPO_ROOT / "results"
    if species == "human":
        return (
            AblationRun("full", "Full", LIGHT_BLUE, results_root / "human"),
            AblationRun("no_bio", "-Bio", AMBER, results_root / "human_nobio"),
            AblationRun(
                "no_structure",
                "-Struct",
                PURPLE,
                results_root / "human_nostruct",
            ),
            AblationRun(
                "lightgbm",
                "LightGBM",
                GREY,
                results_root / "human_lightgbm",
            ),
        )
    return (
        AblationRun("full", "Full", LIGHT_BLUE, results_root / "mouse"),
        AblationRun("no_bio", "-Bio", AMBER, results_root / "mouse_nobio"),
        AblationRun(
            "no_structure",
            "-Struct",
            PURPLE,
            results_root / "mouse_nostruct",
        ),
        AblationRun(
            "lightgbm",
            "LightGBM",
            GREY,
            results_root / "mouse_lightgbm",
        ),
    )


def short_species_label(species: str) -> str:
    return species[0].upper()


def display_target_name(name: str, *, max_length: int = 22) -> str:
    return pretty_target_name(name)[:max_length]


def adjusted_p_value(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    comparisons: int,
) -> float | None:
    del comparisons
    if candidate.size == 0 or reference.size == 0 or candidate.size != reference.size:
        return None
    if np.allclose(candidate, reference):
        return None
    result = wilcoxon(
        candidate, reference, zero_method="pratt", alternative="two-sided"
    )
    return float(result.pvalue)


def significance_symbol(value: float | None) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def draw_significance_brackets(
    ax: Axes,
    brackets: Sequence[tuple[float, float, str | None]],
    *,
    top_y: float,
    step: float,
    cap_height: float,
    pad_above: float = 0.01,
) -> None:
    visible = [item for item in brackets if item[2] is not None]
    if not visible:
        return

    for level, (left, right, label) in enumerate(visible):
        y = top_y + level * step
        ax.plot(
            [left, left, right, right],
            [y, y + cap_height, y + cap_height, y],
            color="black",
            linewidth=0.8,
            clip_on=False,
        )
        ax.text(
            (left + right) / 2,
            y + cap_height + step * 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold" if label != "ns" else None,
            clip_on=False,
        )

    upper = top_y + (len(visible) - 1) * step + cap_height + pad_above
    lower, current_upper = ax.get_ylim()
    if upper > current_upper:
        ax.set_ylim(lower, upper)


def dynamic_bracket_geometry(
    values: Sequence[np.ndarray],
    *,
    visible_levels: int,
) -> tuple[float, float, float, float]:
    max_value = max(float(np.nanmax(item)) for item in values)
    min_value = min(float(np.nanmin(item)) for item in values)
    data_range = max(max_value - min_value, 0.015)
    top_y = max_value + max(data_range * 0.04, 0.003)
    step = max(data_range * 0.08, 0.004) if visible_levels > 1 else 0.0
    cap_height = max(data_range * 0.035, 0.003)
    pad_above = max(data_range * 0.03, 0.003)
    return top_y, step, cap_height, pad_above


def apply_compact_y_ticks(
    ax: Axes,
    values: Sequence[np.ndarray],
    *,
    candidate_steps: Sequence[float] = (0.05, 0.1, 0.2),
) -> None:
    lower_limit, upper_limit = ax.get_ylim()
    data_min = min(float(np.nanmin(item)) for item in values)
    data_max = max(float(np.nanmax(item)) for item in values)
    span = max(upper_limit - data_min, data_max - data_min, 1.0e-6)
    target_step = span / 4.0
    step = next(
        (candidate for candidate in candidate_steps if candidate >= target_step),
        candidate_steps[-1],
    )

    lower_tick = np.floor(data_min / step) * step
    upper_tick = np.ceil(upper_limit / step) * step
    tick_values = np.arange(lower_tick, upper_tick + step * 0.5, step)
    ax.set_yticks(tick_values)
    ax.set_ylim(lower_tick, upper_limit)


def load_summary_mean_std(
    run_dir: Path,
    *,
    scope: str = "mean_te",
    metric: str = "pearson",
) -> tuple[float, float]:
    summary = pd.read_csv(run_dir / "cross_validation_summary.csv")
    row = summary[(summary["scope"] == scope) & (summary["metric"] == metric)]
    if row.empty:
        raise ValueError(
            f"No summary row for scope={scope!r}, metric={metric!r} under {run_dir}"
        )
    return float(row.iloc[0]["mean"]), float(row.iloc[0]["std"])


def draw_scatter_panel(ax: Axes, *, species: str, run: ModelRun) -> pd.DataFrame:
    frame = load_oof_mean_te_frame(run.run_dir)
    pearson_value, spearman_value = scatter_metrics(frame)
    r_squared = float(r2_score(frame["target_mean_te"], frame["prediction_mean_te"]))

    ax.hexbin(
        frame["target_mean_te"],
        frame["prediction_mean_te"],
        gridsize=30,
        cmap="plasma",
        mincnt=1,
        alpha=0.9,
    )
    x_values = frame["target_mean_te"].to_numpy(dtype=float)
    y_values = frame["prediction_mean_te"].to_numpy(dtype=float)
    lower = float(min(x_values.min(), y_values.min()))
    upper = float(max(x_values.max(), y_values.max()))
    ax.plot([lower, upper], [lower, upper], "--", lw=1, color="red")
    ax.text(
        0.05,
        0.95,
        f"r={pearson_value:.2f}\nρ={spearman_value:.2f}\nR²={r_squared:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_xlabel("Measured TE")
    ax.set_ylabel("Predicted TE")
    ax.set_title(
        f"{run.label} Scatter ({short_species_label(species)})",
        fontweight="bold",
        fontsize=9,
    )
    return frame.assign(panel="scatter", species=species, model=run.label)


def draw_violin_panel(
    ax: Axes,
    *,
    species: str,
    runs: Sequence[ModelRun],
) -> pd.DataFrame:
    violin_values: list[np.ndarray] = []
    source_frames: list[pd.DataFrame] = []
    for run in runs:
        frame = load_per_target_metric_frame(run.run_dir, metric="pearson")
        violin_values.append(frame["pearson"].to_numpy(dtype=float))
        source_frames.append(
            frame.assign(panel="violin", species=species, model=run.label)
        )

    parts = ax.violinplot(violin_values, positions=range(len(runs)), showmeans=True)
    for body, run in zip(parts["bodies"], runs, strict=True):
        body.set_facecolor(run.color)
        body.set_alpha(0.7)
        body.set_edgecolor("black")
        body.set_linewidth(0.4)
    if "cmeans" in parts:
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(0.7)

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([run.label for run in runs], fontsize=7)
    ax.set_ylabel("Pearson")
    ax.set_title(
        f"Per-Source ({short_species_label(species)})",
        fontweight="bold",
        fontsize=9,
    )

    reference_frame = aggregate_per_target_metric(
        runs[0].run_dir, metric="pearson"
    ).rename(columns={"value": runs[0].label})
    comparison_brackets: list[tuple[float, float, str | None]] = []
    comparison_count = max(len(runs) - 1, 1)
    for run_index, run in enumerate(runs[1:], start=1):
        frame = aggregate_per_target_metric(run.run_dir, metric="pearson").rename(
            columns={"value": run.label}
        )
        merged = reference_frame.merge(frame, on="target_name", how="inner")
        p_value = adjusted_p_value(
            merged[runs[0].label].to_numpy(dtype=float),
            merged[run.label].to_numpy(dtype=float),
            comparisons=comparison_count,
        )
        comparison_brackets.append((0, run_index, significance_symbol(p_value)))

    violin_max = max(float(np.nanmax(values)) for values in violin_values)
    draw_significance_brackets(
        ax,
        comparison_brackets,
        top_y=violin_max + 0.02,
        step=0.055,
        cap_height=0.015,
    )
    return pd.concat(source_frames, ignore_index=True)


def draw_ablation_panel(
    ax: Axes,
    *,
    species: str,
    runs: Sequence[AblationRun],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    means: list[float] = []
    stds: list[float] = []
    availability: list[bool] = []

    for run in runs:
        if run.run_dir is None or not run.run_dir.exists():
            means.append(float("nan"))
            stds.append(float("nan"))
            availability.append(False)
            rows.append(
                {
                    "panel": "ablation",
                    "species": species,
                    "model": run.label,
                    "mean": np.nan,
                    "std": np.nan,
                    "available": False,
                }
            )
            continue

        mean_value, std_value = load_summary_mean_std(run.run_dir)
        means.append(mean_value)
        stds.append(std_value)
        availability.append(True)
        rows.append(
            {
                "panel": "ablation",
                "species": species,
                "model": run.label,
                "mean": mean_value,
                "std": std_value,
                "available": True,
            }
        )

    x_positions = np.arange(len(runs))
    for index, run in enumerate(runs):
        if availability[index]:
            ax.bar(
                index,
                means[index],
                yerr=stds[index],
                capsize=2,
                color=run.color,
                edgecolor="black",
                linewidth=0.5,
            )
            continue

        ax.bar(
            index,
            0.0,
            color=MISSING_FACE,
            edgecolor=GREY,
            linewidth=0.5,
            hatch="//",
        )
        ax.text(index, 0.04, "NA", ha="center", va="bottom", fontsize=7)

    full_mean = means[0]
    if np.isfinite(full_mean):
        for index in range(1, len(runs)):
            if not availability[index]:
                continue
            delta = float(means[index] - full_mean)
            delta_text = f"$\\Delta$={delta:.2f}" if delta < 0 else f"+{delta:.2f}"
            ax.text(
                index,
                float(means[index] + stds[index] + 0.02),
                delta_text,
                ha="center",
                fontsize=6.5,
                color="black",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([run.label for run in runs], rotation=30, ha="right", fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Ablation ({short_species_label(species)})",
        fontweight="bold",
        fontsize=9,
    )

    full_run = runs[0]
    comparison_brackets: list[tuple[float, float, str | None]] = []
    comparison_count = sum(
        1 for run in runs[1:] if run.run_dir is not None and run.run_dir.exists()
    )
    if full_run.run_dir is not None and comparison_count > 0:
        full_frame = load_metric_frame(full_run.run_dir, metric="pearson")
        for run_index, run in enumerate(runs[1:], start=1):
            if run.run_dir is None or not run.run_dir.exists():
                continue
            comparison_frame = load_metric_frame(run.run_dir, metric="pearson")
            merged = full_frame.merge(
                comparison_frame,
                on="fold",
                suffixes=("_full", "_cmp"),
            )
            p_value = adjusted_p_value(
                merged["value_full"].to_numpy(dtype=float),
                merged["value_cmp"].to_numpy(dtype=float),
                comparisons=comparison_count,
            )
            comparison_brackets.append((0, run_index, significance_symbol(p_value)))

    available_tops = [
        float(mean + std)
        for mean, std, is_available in zip(means, stds, availability, strict=True)
        if is_available
    ]
    top_y = max(available_tops, default=0.0) + 0.025
    draw_significance_brackets(
        ax,
        comparison_brackets,
        top_y=top_y,
        step=0.06,
        cap_height=0.018,
        pad_above=0.015,
    )
    return pd.DataFrame(rows)


def draw_best_worst_panel(ax: Axes, *, species: str, run: ModelRun) -> pd.DataFrame:
    frame = aggregate_per_target_metric(run.run_dir, metric="pearson").rename(
        columns={"value": "pearson"}
    )
    frame["name"] = frame["target_name"].map(display_target_name)

    top = frame.nlargest(5, "pearson")
    bottom = frame.nsmallest(5, "pearson")
    combined = pd.concat([top, bottom], ignore_index=True)
    colors = [TEAL] * len(top) + [VERMILION] * len(bottom)

    ax.barh(
        np.arange(len(combined)),
        combined["pearson"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(np.arange(len(combined)))
    ax.set_yticklabels(combined["name"], fontsize=4.5)
    ax.set_xlabel("Pearson Correlation", fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_title(
        f"Best/Worst Sources ({short_species_label(species)})",
        fontweight="bold",
        fontsize=8,
    )
    return combined.assign(panel="best_worst", species=species, model=run.label)


def draw_cv_stability_panel(
    ax: Axes,
    *,
    species: str,
    runs: Sequence[ModelRun],
) -> pd.DataFrame:
    boxplot_values: list[np.ndarray] = []
    source_frames: list[pd.DataFrame] = []
    for run in runs:
        frame = load_metric_frame(run.run_dir, metric="pearson")
        boxplot_values.append(frame["value"].to_numpy(dtype=float))
        source_frames.append(
            frame.assign(panel="cv_stability", species=species, model=run.label)
        )

    boxplot = ax.boxplot(
        boxplot_values,
        patch_artist=True,
        tick_labels=[run.label for run in runs],
        widths=0.5,
    )
    for patch, run in zip(boxplot["boxes"], runs, strict=True):
        patch.set_facecolor(run.color)
        patch.set_alpha(0.7)
    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(0.8)

    ax.set_ylabel("Pearson")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(
        f"CV Stability ({short_species_label(species)})",
        fontweight="bold",
        fontsize=9,
    )

    apply_compact_y_ticks(ax, boxplot_values)
    return pd.concat(source_frames, ignore_index=True)


def draw_transfer_panel(
    ax: Axes,
    *,
    train_species: str,
    transfer_species: str,
    title: str,
    runs: Sequence[ModelRun],
) -> pd.DataFrame:
    x_positions = np.arange(len(runs))
    width = 0.35
    train_species_label = train_species.capitalize()
    transfer_species_label = transfer_species.capitalize()
    within_values: list[float] = []
    transfer_values: list[float] = []
    transfer_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    comparison_brackets: list[tuple[float, float, str | None]] = []

    for run_index, run in enumerate(runs):
        within_frame = load_metric_frame(run.run_dir, metric="pearson")
        transfer_frame = load_transfer_metric_frame(
            run.transfer_run_dir or run.run_dir,
            metric="pearson",
        )
        within_mean = float(np.asarray(within_frame["value"], dtype=float).mean())
        transfer_mean = float(np.asarray(transfer_frame["value"], dtype=float).mean())
        within_values.append(within_mean)
        transfer_values.append(transfer_mean)
        transfer_frames.append(transfer_frame)

        rows.extend(
            [
                {
                    "panel": "transfer",
                    "title": title,
                    "model": run.label,
                    "regime": f"Train {train_species_label} / Test {train_species_label}",
                    "value": within_mean,
                },
                {
                    "panel": "transfer",
                    "title": title,
                    "model": run.label,
                    "regime": f"Train {train_species_label} / Test {transfer_species_label}",
                    "value": transfer_mean,
                },
            ]
        )

    ax.bar(
        x_positions - width / 2,
        within_values,
        width,
        color=TRANSFER_WITHIN_COLOR,
        alpha=0.9,
    )
    ax.bar(
        x_positions + width / 2,
        transfer_values,
        width,
        color=TRANSFER_CROSS_COLOR,
        alpha=0.9,
    )
    ax.set_xticks(x_positions)
    labels = [run.label for run in runs]
    if len(labels) > 3:
        ax.set_xticklabels(labels, fontsize=6, rotation=18, ha="right")
    else:
        ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(title, fontweight="bold", fontsize=9)

    hybrite_index = next(
        (index for index, run in enumerate(runs) if run.key == "hybrite"),
        None,
    )
    comparator_indices = [
        index
        for index, run in enumerate(runs)
        if run.key != "hybrite" and hybrite_index is not None and index != hybrite_index
    ]
    comparison_count = max(len(comparator_indices), 1)
    if hybrite_index is not None:
        for comparator_index in comparator_indices:
            comparator_run = runs[comparator_index]
            merged = transfer_frames[hybrite_index].merge(
                transfer_frames[comparator_index],
                on="fold",
                suffixes=("_hybrite", f"_{comparator_run.key}"),
            )
            p_value = adjusted_p_value(
                merged["value_hybrite"].to_numpy(dtype=float),
                merged[f"value_{comparator_run.key}"].to_numpy(dtype=float),
                comparisons=comparison_count,
            )
            comparison_brackets.append(
                (
                    float(x_positions[hybrite_index] + width / 2),
                    float(x_positions[comparator_index] + width / 2),
                    significance_symbol(p_value),
                )
            )

    visible_levels = sum(1 for _, _, label in comparison_brackets if label is not None)
    top_y, step, cap_height, pad_above = dynamic_bracket_geometry(
        [
            np.asarray(within_values, dtype=float),
            np.asarray(transfer_values, dtype=float),
        ],
        visible_levels=visible_levels,
    )
    if visible_levels > 1:
        step = max(step * 1.6, 0.018)
        cap_height = max(cap_height, 0.006)
        pad_above = max(pad_above, 0.01)
    draw_significance_brackets(
        ax,
        comparison_brackets,
        top_y=top_y,
        step=step,
        cap_height=cap_height,
        pad_above=pad_above,
    )
    return pd.DataFrame(rows)


def add_shared_transfer_legend(fig: Figure, *, axes: Sequence[Axes]) -> None:
    positions = [ax.get_position() for ax in axes]
    legend_center_x = (
        min(position.x0 for position in positions)
        + max(position.x1 for position in positions)
    ) / 2
    legend_top_y = max(min(position.y0 for position in positions) - 0.05, 0.01)
    legend_labels = ["Within-species test", "Cross-species transfer"]
    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
        )
        for color in (TRANSFER_WITHIN_COLOR, TRANSFER_CROSS_COLOR)
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(legend_center_x, legend_top_y),
        ncol=2,
        fontsize=6,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def create_figure() -> Figure:
    return plt.figure(figsize=(210 * MM_TO_INCH, 200 * MM_TO_INCH))


def main() -> None:
    args = parse_args()
    apply_legacy_style()

    human_runs = benchmark_runs("human")
    mouse_runs = benchmark_runs("mouse")
    human_transfer_runs = transfer_runs("human")
    mouse_transfer_runs = transfer_runs("mouse")
    human_ablation_runs = ablation_runs("human")
    mouse_ablation_runs = ablation_runs("mouse")

    fig = create_figure()
    grid = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        hspace=0.6,
        wspace=0.4,
        left=0.06,
        right=0.98,
        top=0.95,
        bottom=0.08,
    )
    panel_labels = list("ABCDEFGHIJKL")
    source_frames: list[pd.DataFrame] = []

    ax_a = fig.add_subplot(grid[0, 0])
    source_frames.append(draw_scatter_panel(ax_a, species="human", run=human_runs[0]))
    add_panel_label(ax_a, panel_labels[0])

    ax_b = fig.add_subplot(grid[0, 1])
    source_frames.append(draw_scatter_panel(ax_b, species="mouse", run=mouse_runs[0]))
    add_panel_label(ax_b, panel_labels[1])

    ax_c = fig.add_subplot(grid[0, 2])
    source_frames.append(draw_violin_panel(ax_c, species="human", runs=human_runs))
    add_panel_label(ax_c, panel_labels[2])

    ax_d = fig.add_subplot(grid[0, 3])
    source_frames.append(draw_violin_panel(ax_d, species="mouse", runs=mouse_runs))
    add_panel_label(ax_d, panel_labels[3])

    ax_e = fig.add_subplot(grid[1, 0])
    source_frames.append(
        draw_ablation_panel(ax_e, species="human", runs=human_ablation_runs)
    )
    add_panel_label(ax_e, panel_labels[4])

    ax_f = fig.add_subplot(grid[1, 1])
    source_frames.append(
        draw_ablation_panel(ax_f, species="mouse", runs=mouse_ablation_runs)
    )
    add_panel_label(ax_f, panel_labels[5])

    ax_g = fig.add_subplot(grid[1, 2])
    source_frames.append(
        draw_best_worst_panel(ax_g, species="human", run=human_runs[0])
    )
    add_panel_label(ax_g, panel_labels[6])

    ax_h = fig.add_subplot(grid[1, 3])
    source_frames.append(
        draw_best_worst_panel(ax_h, species="mouse", run=mouse_runs[0])
    )
    add_panel_label(ax_h, panel_labels[7])

    ax_i = fig.add_subplot(grid[2, 0])
    source_frames.append(
        draw_cv_stability_panel(ax_i, species="human", runs=human_runs)
    )
    add_panel_label(ax_i, panel_labels[8])

    ax_j = fig.add_subplot(grid[2, 1])
    source_frames.append(
        draw_cv_stability_panel(ax_j, species="mouse", runs=mouse_runs)
    )
    add_panel_label(ax_j, panel_labels[9])

    ax_k = fig.add_subplot(grid[2, 2])
    source_frames.append(
        draw_transfer_panel(
            ax_k,
            train_species="human",
            transfer_species="mouse",
            title="Train on Human",
            runs=human_transfer_runs,
        )
    )
    add_panel_label(ax_k, panel_labels[10])

    ax_l = fig.add_subplot(grid[2, 3])
    source_frames.append(
        draw_transfer_panel(
            ax_l,
            train_species="mouse",
            transfer_species="human",
            title="Train on Mouse",
            runs=mouse_transfer_runs,
        )
    )
    add_panel_label(ax_l, panel_labels[11])
    add_shared_transfer_legend(fig, axes=(ax_k, ax_l))

    source_data = pd.concat(source_frames, ignore_index=True, sort=False)
    save_figure(fig, args.output_stem, source_data)
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI surface
        raise SystemExit(
            f"Performance-analysis figure generation failed: {exc}"
        ) from exc

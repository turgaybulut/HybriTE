# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import _plot_bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from hybrite.config import REPO_ROOT
from hybrite.results_figures import (
    aggregate_per_target_metric,
    pretty_target_name,
    save_figure,
)

MM_TO_INCH = 1 / 25.4

LIGHT_BLUE = "#00509E"
VERMILION = "#D55E00"
TEAL = "#009E73"
GREY = "#666666"
ROW_SHADE = "#F5F5F5"


@dataclass(frozen=True)
class ModelRun:
    label: str
    color: str
    run_dir: Path
    marker: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-output-stem", default="figure_per_target_human")
    parser.add_argument("--mouse-output-stem", default="figure_per_target_mouse")
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


def benchmark_runs(species: str) -> tuple[ModelRun, ...]:
    results_root = REPO_ROOT / "results"
    return (
        ModelRun("HybriTE", LIGHT_BLUE, results_root / species, "o"),
        ModelRun("RiboNN", VERMILION, results_root / f"{species}_ribonn", "s"),
        ModelRun("Saluki", TEAL, results_root / f"{species}_saluki", "^"),
    )


def output_stem_for_species(args: argparse.Namespace, species: str) -> str:
    if species == "human":
        return str(args.human_output_stem)
    return str(args.mouse_output_stem)


def target_height_mm(species: str) -> float:
    return 350.0 if species == "human" else 320.0


def prepare_per_target_frame(species: str) -> pd.DataFrame:
    runs = benchmark_runs(species)
    merged: pd.DataFrame | None = None
    for run in runs:
        frame = aggregate_per_target_metric(run.run_dir, metric="pearson").rename(
            columns={"value": run.label}
        )
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on="target_name", how="inner")
        )
    if merged is None:
        raise ValueError(f"No per-target data available for {species}")

    merged["display_name"] = merged["target_name"].map(
        lambda name: pretty_target_name(str(name))[:35]
    )
    merged = merged.sort_values(
        by="HybriTE", ascending=True, kind="stable"
    ).reset_index(drop=True)
    return merged


def axis_limits(frame: pd.DataFrame, labels: list[str]) -> tuple[float, float]:
    values = np.concatenate([frame[label].to_numpy(dtype=float) for label in labels])
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    x_min = max(0.0, np.floor(data_min * 10.0) / 10.0 - 0.05)
    x_max = min(1.0, data_max + 0.05)
    return x_min, x_max


def draw_mean_value_box(ax: Axes, mean_values: dict[str, float]) -> None:
    lines = ["Mean values"] + [
        f"{label}: {value:.3f}" for label, value in mean_values.items()
    ]
    colors = [None, LIGHT_BLUE, VERMILION, TEAL]
    x_position = 0.985
    y_positions = [0.14, 0.11, 0.08, 0.05]
    ax.text(
        x_position,
        y_positions[0],
        lines[0],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        fontstyle="italic",
        color=GREY,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "none",
        },
    )
    for offset, (label, value) in enumerate(mean_values.items(), start=1):
        ax.text(
            x_position,
            y_positions[offset],
            f"{label}: {value:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=colors[offset],
        )


def plot_species(species: str, *, output_stem: str) -> None:
    runs = benchmark_runs(species)
    frame = prepare_per_target_frame(species)
    n_targets = len(frame)
    x_min, x_max = axis_limits(frame, [run.label for run in runs])

    fig, ax = plt.subplots(
        figsize=(180 * MM_TO_INCH, target_height_mm(species) * MM_TO_INCH)
    )
    y_positions = np.arange(n_targets, dtype=float)

    for index in range(0, n_targets, 2):
        ax.axhspan(index - 0.5, index + 0.5, color=ROW_SHADE, zorder=0)

    grid_start = np.ceil(x_min * 10.0) / 10.0
    grid_end = np.floor(x_max * 10.0) / 10.0
    if grid_end >= grid_start:
        for x_grid in np.arange(grid_start, grid_end + 0.001, 0.1):
            ax.axvline(x_grid, color=GREY, lw=0.3, alpha=0.2, zorder=0)

    mean_values: dict[str, float] = {}
    for run in runs:
        mean_value = float(frame[run.label].mean())
        mean_values[run.label] = mean_value
        ax.axvline(
            mean_value,
            color=run.color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=0,
        )

    for y_index, row in frame.iterrows():
        model_values = [float(row[run.label]) for run in runs]
        ax.plot(
            model_values,
            [y_index] * len(model_values),
            color=GREY,
            lw=0.3,
            alpha=0.5,
            zorder=1,
        )

    for y_index, row in frame.iterrows():
        for run in runs:
            ax.plot(
                [x_min, float(row[run.label])],
                [y_index, y_index],
                color=run.color,
                lw=0.5,
                alpha=0.3,
                zorder=1,
            )

    for run in runs:
        ax.scatter(
            frame[run.label],
            y_positions,
            c=run.color,
            marker=run.marker,
            s=25 if run.label == "HybriTE" else 20,
            label=run.label,
            zorder=3 if run.label == "HybriTE" else 2,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["display_name"], fontsize=6)
    ax.set_xlabel("Pearson Correlation", fontsize=8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, n_targets - 0.5)
    ax.set_title(
        f"Per Biological Source Performance ({species.capitalize()})",
        fontweight="bold",
        fontsize=10,
    )
    ax.legend(
        loc="lower right",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        ncol=3,
        borderaxespad=0.5,
    )
    draw_mean_value_box(ax, mean_values)

    source_data = frame.copy()
    source_data.insert(0, "species", species)
    save_figure(fig, output_stem, source_data)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_legacy_style()
    for species in ["human", "mouse"]:
        plot_species(species, output_stem=output_stem_for_species(args, species))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI surface
        raise SystemExit(
            f"Per-target performance figure generation failed: {exc}"
        ) from exc

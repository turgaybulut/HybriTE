# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from pathlib import Path

import _plot_bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from hybrite.config import REPO_ROOT
from hybrite.results_figures import ensure_columns, pretty_target_name, save_figure

MM_TO_INCH = 1 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--human-output-stem", default="figure_target_correlation_human"
    )
    parser.add_argument(
        "--mouse-output-stem", default="figure_target_correlation_mouse"
    )
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
            "savefig.bbox": "tight",
        }
    )


def output_stem_for_species(args: argparse.Namespace, species: str) -> str:
    if species == "human":
        return str(args.human_output_stem)
    return str(args.mouse_output_stem)


def run_dir_for_species(species: str) -> Path:
    return REPO_ROOT / "results" / species


def figure_size_mm(species: str) -> float:
    return 450.0 if species == "human" else 400.0


def load_oof_prediction_matrix(species: str) -> tuple[np.ndarray, list[str]]:
    run_dir = run_dir_for_species(species)
    rows: list[pd.DataFrame] = []
    target_columns: list[str] | None = None
    for fold_dir in sorted(run_dir.glob("fold_*")):
        path = fold_dir / "test" / "predictions.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        ensure_columns(frame, ["gene_id"], source_name=str(path))
        current_target_columns = [
            column for column in frame.columns if column != "gene_id"
        ]
        if target_columns is None:
            target_columns = current_target_columns
        elif current_target_columns != target_columns:
            raise ValueError(
                f"Inconsistent target columns across folds under {run_dir}"
            )
        frame = frame[["gene_id", *current_target_columns]].copy()
        frame.insert(0, "fold", int(fold_dir.name.split("_")[-1]))
        rows.append(frame)

    if not rows or target_columns is None:
        raise FileNotFoundError(
            f"No fold-level predictions.csv files found under {run_dir}"
        )

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["gene_id"], keep="first")
    matrix = combined[target_columns].to_numpy(dtype=float)
    return matrix, target_columns


def correlation_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "correlation",
        [
            "#313695",
            "#4575B4",
            "#74ADD1",
            "#ABD9E9",
            "#FDAE61",
            "#F46D43",
            "#D73027",
            "#A50026",
        ],
    )


def pretty_target_labels(target_columns: list[str]) -> list[str]:
    return [pretty_target_name(column)[:25] for column in target_columns]


def long_form_correlation_frame(
    species: str,
    target_columns: list[str],
    corr_matrix: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_index, row_name in enumerate(target_columns):
        for col_index, col_name in enumerate(target_columns):
            rows.append(
                {
                    "species": species,
                    "row_index": row_index,
                    "col_index": col_index,
                    "row_target": row_name,
                    "col_target": col_name,
                    "correlation": float(corr_matrix[row_index, col_index]),
                    "abs_correlation": float(abs(corr_matrix[row_index, col_index])),
                    "is_lower_triangle": bool(row_index > col_index),
                }
            )
    return pd.DataFrame(rows)


def plot_species(species: str, *, output_stem: str) -> None:
    prediction_matrix, target_columns = load_oof_prediction_matrix(species)
    corr_matrix = np.corrcoef(prediction_matrix.T)
    abs_corr = np.abs(corr_matrix)
    mask = np.triu(np.ones_like(abs_corr, dtype=bool), k=0)
    masked_corr = np.where(mask, np.nan, abs_corr)
    labels = pretty_target_labels(target_columns)

    fig, ax = plt.subplots(
        figsize=(
            figure_size_mm(species) * MM_TO_INCH,
            figure_size_mm(species) * MM_TO_INCH,
        )
    )
    image = ax.imshow(
        masked_corr, cmap=correlation_cmap(), vmin=0, vmax=1, aspect="equal"
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5, fontweight="medium")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5, fontweight="medium")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{species.capitalize()} Prediction Similarity Across Biological Sources",
        fontweight="bold",
        fontsize=12,
        loc="left",
    )
    colorbar = fig.colorbar(
        image, ax=ax, orientation="horizontal", shrink=0.3, aspect=40, pad=0.08
    )
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Absolute Correlation", fontsize=9, fontweight="medium")

    source_data = long_form_correlation_frame(species, target_columns, corr_matrix)
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
        raise SystemExit(f"Target correlation figure generation failed: {exc}") from exc

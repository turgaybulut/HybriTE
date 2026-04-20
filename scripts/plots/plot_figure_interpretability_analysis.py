# pyright: reportMissingImports=false

from __future__ import annotations

import argparse

import _plot_bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hybrite.config import REPO_ROOT
from hybrite.results_figures import (
    MODEL_COLORS,
    apply_publication_style,
    ensure_columns,
    mean_confidence_interval,
    panel_label,
    save_figure,
)

BIOCHEMICAL_CATEGORY_ORDER = ["ENCORI", "eCLIP", "M6ACLIP", "miRNA", "Other"]
BIOCHEMICAL_CATEGORY_LABELS = {
    "ENCORI": "ENCORI",
    "eCLIP": "eCLIP",
    "M6ACLIP": "m6A CLIP-derived",
    "miRNA": "miRNA",
    "Other": "Other",
}
BIOCHEMICAL_CATEGORY_COLORS = {
    "ENCORI": MODEL_COLORS["ENCORI"],
    "eCLIP": MODEL_COLORS["eCLIP"],
    "M6ACLIP": MODEL_COLORS["M6ACLIP"],
    "miRNA": "#B07AA1",
    "Other": "#BAB0AC",
}
REGION_COLORS = {
    "5'UTR": "#FF7F0E",
    "CDS": "#1F77B4",
    "3'UTR": "#2CA02C",
}
REGION_ORDER = ["5'UTR", "CDS", "3'UTR"]
EDGE_TYPE_ORDER = ["Sequence", "Structure", "Long-range"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", default="figure_interpretability_analysis")
    parser.add_argument("--top-biochemical", type=int, default=10)
    parser.add_argument("--top-node-features", type=int, default=10)
    return parser.parse_args()


def _load_required_csv(path: str, columns: list[str]) -> pd.DataFrame:
    resolved = REPO_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(
            f"Missing interpretability artifact: {resolved}. "
            "Generate them first with: python scripts/generate_interpretability_artifacts.py"
        )
    frame = pd.read_csv(resolved)
    ensure_columns(frame, columns, source_name=str(resolved))
    return frame


def _load_manifest() -> dict[str, object]:
    resolved = (
        REPO_ROOT / "artifacts/interpretability/human/interpretability_manifest.json"
    )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Missing interpretability manifest: {resolved}. "
            "Generate them first with: python scripts/generate_interpretability_artifacts.py"
        )
    return pd.read_json(resolved, typ="series").to_dict()


def _clean_biochemical_label(label: str) -> str:
    if "." in label:
        left, right = label.split(".", 1)
        return f"{left} [{right}]"
    return label


def _node_display_label(feature_label: str, region: str) -> str:
    return f"{feature_label} [{region}]"


def _expand_biochemical_by_fold(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
) -> pd.DataFrame:
    folds = sorted(by_fold["fold"].astype(int).unique().tolist())
    features = summary[["feature_name", "feature_label", "category"]].drop_duplicates()
    template = features.assign(_key=1).merge(
        pd.DataFrame({"fold": folds, "_key": 1}),
        on="_key",
        how="inner",
    )
    template = template.drop(columns=["_key"])
    merged = template.merge(
        by_fold[["fold", "feature_name", "delta"]],
        on=["fold", "feature_name"],
        how="left",
    )
    merged["delta"] = merged["delta"].fillna(0.0)
    return merged


def _top_biochemical_summary(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    *,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expanded = _expand_biochemical_by_fold(summary, by_fold)
    ranked = summary.sort_values(by="importance", ascending=False).head(top_n).copy()
    ranked["display_label"] = ranked["feature_label"].map(_clean_biochemical_label)
    ranked = ranked.iloc[::-1].reset_index(drop=True)

    selected = expanded.merge(
        ranked[["feature_name", "display_label", "category"]],
        on=["feature_name", "category"],
        how="inner",
    )
    stats_rows: list[dict[str, object]] = []
    for feature_name in ranked["feature_name"]:
        values = selected.loc[
            selected["feature_name"] == feature_name, "delta"
        ].to_numpy(dtype=float)
        mean_value, ci_lower, ci_upper = mean_confidence_interval(values)
        stats_rows.append(
            {
                "feature_name": feature_name,
                "mean": mean_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    stats = pd.DataFrame(stats_rows)
    ranked = ranked.merge(stats, on="feature_name", how="left")
    ranked["color"] = ranked["category"].map(BIOCHEMICAL_CATEGORY_COLORS)

    category_summary = (
        expanded.groupby(["fold", "category"], as_index=False)["delta"]
        .mean()
        .rename(columns={"delta": "mean_delta_per_feature"})
    )
    category_stats_rows: list[dict[str, object]] = []
    for category in BIOCHEMICAL_CATEGORY_ORDER:
        values = category_summary.loc[
            category_summary["category"] == category,
            "mean_delta_per_feature",
        ].to_numpy(dtype=float)
        if values.size == 0:
            continue
        mean_value, ci_lower, ci_upper = mean_confidence_interval(values)
        category_stats_rows.append(
            {
                "category": category,
                "display_category": BIOCHEMICAL_CATEGORY_LABELS[category],
                "mean": mean_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "color": BIOCHEMICAL_CATEGORY_COLORS[category],
            }
        )
    category_stats = pd.DataFrame(category_stats_rows)
    return ranked, selected, category_stats


def _top_node_feature_summary(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    *,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = summary.sort_values(by="importance", ascending=False).head(top_n).copy()
    ranked["display_label"] = [
        _node_display_label(feature_label, region)
        for feature_label, region in zip(
            ranked["feature_label"], ranked["region"], strict=False
        )
    ]
    ranked = ranked.iloc[::-1].reset_index(drop=True)
    selected = by_fold.merge(
        ranked[["feature_name", "display_label", "region"]],
        on=["feature_name", "region"],
        how="inner",
    )
    stats_rows: list[dict[str, object]] = []
    for feature_name in ranked["feature_name"]:
        values = selected.loc[
            selected["feature_name"] == feature_name, "delta"
        ].to_numpy(dtype=float)
        mean_value, ci_lower, ci_upper = mean_confidence_interval(values)
        stats_rows.append(
            {
                "feature_name": feature_name,
                "mean": mean_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    stats = pd.DataFrame(stats_rows)
    ranked = ranked.merge(stats, on="feature_name", how="left")
    ranked["color"] = ranked["region"].map(REGION_COLORS)
    return ranked, selected


def _regional_curve_summary(by_fold: pd.DataFrame) -> pd.DataFrame:
    grouped = by_fold.groupby(
        ["node_index", "normalized_position", "region"], as_index=False
    )
    summary = grouped["delta"].agg(mean="mean")
    ci_rows: list[dict[str, float]] = []
    for (node_index, normalized_position, region), frame in by_fold.groupby(
        ["node_index", "normalized_position", "region"], as_index=False
    ):
        mean_value, ci_lower, ci_upper = mean_confidence_interval(
            frame["delta"].to_numpy(dtype=float)
        )
        ci_rows.append(
            {
                "node_index": int(node_index),
                "normalized_position": float(normalized_position),
                "region": str(region),
                "mean": mean_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    return pd.DataFrame(ci_rows).sort_values(by="node_index").reset_index(drop=True)


def _regional_per_bin_summary(
    by_fold: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_fold = (
        by_fold.groupby(["fold", "region"], as_index=False)["delta"]
        .mean()
        .rename(columns={"delta": "mean_delta_per_bin"})
    )
    stats_rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        values = per_fold.loc[
            per_fold["region"] == region, "mean_delta_per_bin"
        ].to_numpy(dtype=float)
        if values.size == 0:
            continue
        mean_value, ci_lower, ci_upper = mean_confidence_interval(values)
        stats_rows.append(
            {
                "region": region,
                "mean": mean_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "color": REGION_COLORS[region],
            }
        )
    stats = pd.DataFrame(stats_rows)
    return per_fold, stats


def _edge_summary(by_fold: pd.DataFrame) -> pd.DataFrame:
    stats_rows: list[dict[str, object]] = []
    for edge_type in EDGE_TYPE_ORDER:
        values = by_fold.loc[by_fold["edge_type"] == edge_type, "delta"].to_numpy(
            dtype=float
        )
        if values.size == 0:
            continue
        mean_value, ci_lower, ci_upper = mean_confidence_interval(values)
        stats_rows.append(
            {
                "edge_type": edge_type,
                "mean": mean_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "color": MODEL_COLORS[edge_type],
            }
        )
    return pd.DataFrame(stats_rows)


def _barh_with_error_bars(
    ax: Axes,
    *,
    summary: pd.DataFrame,
    label_column: str,
    color_column: str,
    xlabel: str,
    title: str,
) -> None:
    y_positions = np.arange(len(summary), dtype=float)
    widths = summary["mean"].to_numpy(dtype=float)
    colors = summary[color_column].tolist()
    xerr = np.vstack(
        [
            widths - summary["ci_lower"].to_numpy(dtype=float),
            summary["ci_upper"].to_numpy(dtype=float) - widths,
        ]
    )
    ax.barh(
        y_positions,
        widths,
        xerr=xerr,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary[label_column])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", alpha=0.45)


def _vertical_bars_with_error_bars(
    ax: Axes,
    *,
    summary: pd.DataFrame,
    label_column: str,
    color_column: str,
    ylabel: str,
    title: str,
) -> None:
    x_positions = np.arange(len(summary), dtype=float)
    heights = summary["mean"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            heights - summary["ci_lower"].to_numpy(dtype=float),
            summary["ci_upper"].to_numpy(dtype=float) - heights,
        ]
    )
    ax.bar(
        x_positions,
        heights,
        yerr=yerr,
        color=summary[color_column].tolist(),
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary[label_column])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.45)


def _draw_biochemical_panel(
    ax: Axes,
    *,
    summary: pd.DataFrame,
) -> None:
    _barh_with_error_bars(
        ax,
        summary=summary,
        label_column="display_label",
        color_column="color",
        xlabel="Δ mean-TE Pearson after feature neutralization",
        title="Biochemical perturbation sensitivity",
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BIOCHEMICAL_CATEGORY_COLORS[category])
        for category in BIOCHEMICAL_CATEGORY_ORDER
        if category in set(summary["category"])
    ]
    legend_labels = [
        BIOCHEMICAL_CATEGORY_LABELS[category]
        for category in BIOCHEMICAL_CATEGORY_ORDER
        if category in set(summary["category"])
    ]
    if legend_handles:
        ax.legend(
            legend_handles, legend_labels, loc="lower right", fontsize=7, frameon=False
        )


def _draw_biochemical_class_panel(ax: Axes, *, category_stats: pd.DataFrame) -> None:
    _vertical_bars_with_error_bars(
        ax,
        summary=category_stats,
        label_column="display_category",
        color_column="color",
        ylabel="Mean per-feature Δ mean-TE Pearson",
        title="Biochemical assay-class summary",
    )
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right")


def _draw_node_feature_panel(
    ax: Axes,
    *,
    summary: pd.DataFrame,
) -> None:
    _barh_with_error_bars(
        ax,
        summary=summary,
        label_column="display_label",
        color_column="color",
        xlabel="Δ mean-TE Pearson after feature neutralization",
        title="Node-feature perturbation sensitivity",
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=REGION_COLORS[region])
        for region in REGION_ORDER
    ]
    ax.legend(
        legend_handles, REGION_ORDER, loc="lower right", fontsize=7, frameon=False
    )


def _draw_regional_panel(
    ax: Axes,
    *,
    curve: pd.DataFrame,
) -> None:
    ax.axvspan(0.0, 8 / 55, color=REGION_COLORS["5'UTR"], alpha=0.12)
    ax.axvspan(8 / 55, 40 / 55, color=REGION_COLORS["CDS"], alpha=0.08)
    ax.axvspan(40 / 55, 1.0, color=REGION_COLORS["3'UTR"], alpha=0.10)
    ax.fill_between(
        curve["normalized_position"],
        curve["ci_lower"],
        curve["ci_upper"],
        color="#1F77B4",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        curve["normalized_position"],
        curve["mean"],
        color="#1F77B4",
        linewidth=2.0,
    )
    ax.set_xlabel("Normalized transcript position")
    ax.set_ylabel("Δ mean-TE Pearson after node occlusion")
    ax.set_title("Regional sensitivity profile", loc="left")
    ax.grid(axis="y", alpha=0.45)
    ax.text(
        0.07, 0.98, "5'UTR", transform=ax.transAxes, ha="center", va="top", fontsize=7
    )
    ax.text(
        0.48, 0.98, "CDS", transform=ax.transAxes, ha="center", va="top", fontsize=7
    )
    ax.text(
        0.88, 0.98, "3'UTR", transform=ax.transAxes, ha="center", va="top", fontsize=7
    )


def _draw_region_summary_panel(ax: Axes, *, per_region_stats: pd.DataFrame) -> None:
    _vertical_bars_with_error_bars(
        ax,
        summary=per_region_stats,
        label_column="region",
        color_column="color",
        ylabel="Per-bin mean Δ mean-TE Pearson",
        title="Region-normalized sensitivity",
    )


def _draw_edge_panel(ax: Axes, *, summary: pd.DataFrame, by_fold: pd.DataFrame) -> None:
    _vertical_bars_with_error_bars(
        ax,
        summary=summary,
        label_column="edge_type",
        color_column="color",
        ylabel="Δ mean-TE Pearson after edge removal",
        title="Edge-type sensitivity",
    )


def create_figure() -> Figure:
    return plt.figure(figsize=(11.0, 8.2), constrained_layout=True)


def main() -> None:
    args = parse_args()
    apply_publication_style()

    manifest = _load_manifest()
    if manifest.get("method") != "deterministic_occlusion_sensitivity":
        raise ValueError(
            "This figure expects perturbation-based interpretability artifacts, "
            f"but manifest reports method={manifest.get('method')!r}."
        )

    biochemical_summary = _load_required_csv(
        "artifacts/interpretability/human/biochemical_feature_importance.csv",
        ["feature_name", "feature_label", "category", "importance"],
    )
    biochemical_by_fold = _load_required_csv(
        "artifacts/interpretability/human/biochemical_feature_importance_by_fold.csv",
        ["fold", "feature_name", "feature_label", "category", "delta"],
    )
    node_summary = _load_required_csv(
        "artifacts/interpretability/human/node_feature_importance.csv",
        ["region", "feature_name", "feature_label", "importance"],
    )
    node_by_fold = _load_required_csv(
        "artifacts/interpretability/human/node_feature_importance_by_fold.csv",
        ["fold", "region", "feature_name", "feature_label", "delta"],
    )
    regional_by_fold = _load_required_csv(
        "artifacts/interpretability/human/regional_importance_by_fold.csv",
        ["fold", "node_index", "normalized_position", "region", "delta"],
    )
    edge_by_fold = _load_required_csv(
        "artifacts/interpretability/human/edge_importance_by_fold.csv",
        ["fold", "edge_type", "delta"],
    )

    biochemical_top, biochemical_top_by_fold, category_stats = _top_biochemical_summary(
        biochemical_summary,
        biochemical_by_fold,
        top_n=args.top_biochemical,
    )
    node_top, node_top_by_fold = _top_node_feature_summary(
        node_summary,
        node_by_fold,
        top_n=args.top_node_features,
    )
    regional_curve = _regional_curve_summary(regional_by_fold)
    per_region_by_fold, per_region_stats = _regional_per_bin_summary(regional_by_fold)
    edge_summary = _edge_summary(edge_by_fold)

    fig = plt.figure(figsize=(14.0, 8.6), constrained_layout=True)
    axes = fig.subplots(2, 3)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flatten()

    _draw_biochemical_panel(
        ax_a,
        summary=biochemical_top,
    )
    panel_label(ax_a, "A")

    _draw_biochemical_class_panel(ax_b, category_stats=category_stats)
    panel_label(ax_b, "B")

    _draw_node_feature_panel(ax_c, summary=node_top)
    panel_label(ax_c, "C")

    _draw_regional_panel(
        ax_d,
        curve=regional_curve,
    )
    panel_label(ax_d, "D")

    _draw_region_summary_panel(ax_e, per_region_stats=per_region_stats)
    panel_label(ax_e, "E")

    _draw_edge_panel(ax_f, summary=edge_summary, by_fold=edge_by_fold)
    panel_label(ax_f, "F")

    source_data = pd.concat(
        [
            biochemical_top.assign(panel="A"),
            biochemical_top_by_fold.assign(panel="A_fold"),
            category_stats.assign(panel="B"),
            node_top.assign(panel="C"),
            node_top_by_fold.assign(panel="C_fold"),
            regional_curve.assign(panel="D"),
            per_region_by_fold.assign(panel="E_fold"),
            per_region_stats.assign(panel="E"),
            edge_summary.assign(panel="F"),
            edge_by_fold.assign(panel="F_fold"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_figure(fig, args.output_stem, source_data)
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI surface
        raise SystemExit(f"Interpretability figure generation failed: {exc}") from exc

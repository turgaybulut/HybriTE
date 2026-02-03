from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import r2_score

torch.manual_seed(654)
warnings.filterwarnings("ignore")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5

MM_TO_INCH: float = 1 / 25.4

FIGURE_DIR: Path = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR: Path = Path("aggregated_model_data")
ABLATION_DIR: Path = Path("aggregated_model_ablation_data")
SHAP_BASE_DIR: Path = Path("figures/shap")

C_BLUE = "#002147"
C_LIGHT_BLUE = "#00509E"
C_VERMILION = "#D55E00"
C_TEAL = "#009E73"
C_PURPLE = "#7B5AA6"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_GREY = "#666666"

MODEL_COLORS = {
    "hybrite": C_LIGHT_BLUE,
    "ribonn": C_VERMILION,
    "saluki": C_TEAL,
}

MODEL_LABELS = {
    "hybrite": "HybriTE",
    "ribonn": "RiboNN",
    "saluki": "Saluki",
}

ABLATION_COLORS = {
    "hybrite": C_LIGHT_BLUE,
    "hybrite_nobio": C_ORANGE,
    "hybrite_nornaplfold": C_PURPLE,
    "hybrite_nornaplfold_nobio": C_GREY,
}

ABLATION_LABELS = {
    "hybrite": "Full",
    "hybrite_nobio": "-Bio",
    "hybrite_nornaplfold": "-Struct",
    "hybrite_nornaplfold_nobio": "-Both",
}

NODE_FEATURE_NAMES = [
    "A%",
    "U%",
    "G%",
    "C%",
    "Ln(L)",
    "Pos",
    "Unp",
    "CpG",
    "uORF",
    "Kozak",
    "TOP",
    "G4",
    "uAUG",
    "tAI",
    "CSC",
    "GC3",
    "Rare",
    "Ramp",
    "Bas",
    "ARE",
    "Dest",
    "miRNA",
    "PolyA",
    "m6A",
    "AU",
]


def load_metadata(species: str) -> dict:
    with open(DATA_DIR / f"{species}_meta.json") as f:
        return json.load(f)


def load_cv_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "agg_cross_validation_summary.csv")


def load_ablation_summary() -> pd.DataFrame:
    return pd.read_csv(ABLATION_DIR / "agg_cross_validation_summary.csv")


def load_target_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "agg_target_metrics.csv")


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "agg_predictions.csv")


def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "agg_ground_truth.csv")


def load_cross_species_summary(direction: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"agg_cross_species_summary_{direction}.csv")


def load_shap_bio_data(species: str) -> tuple:
    path = SHAP_BASE_DIR / species
    shap_values = np.load(path / "shap_values_bio.npy")
    test_samples = np.load(path / "bio_test_samples.npy")
    with open(path / "bio_feature_names.json") as f:
        feature_names = json.load(f)
    return shap_values, test_samples, feature_names


def load_shap_graph_data(species: str) -> dict:
    path = SHAP_BASE_DIR / species
    return {
        "node_mask": np.load(path / "graph_shap_node_mask_mean.npy"),
        "node_location": np.load(path / "graph_shap_node_location_mean.npy"),
        "node_feature": np.load(path / "graph_shap_node_feature_mean.npy"),
    }


def load_target_names(species: str) -> list:
    with open(DATA_DIR / f"{species}_meta.json") as f:
        meta = json.load(f)
    return meta["target_columns"]


def compute_wilcoxon_stats() -> None:
    print("Calculating Wilcoxon statistics (per-target Pearson correlations)...")
    target_metrics = load_target_metrics()
    results = []

    results.append(
        "Wilcoxon Signed-Rank Test Results (Per-Target Pearson Correlations)\n"
    )
    results.append("=" * 70 + "\n\n")

    for species in ["human", "mouse"]:
        species_data = target_metrics[target_metrics["species"] == species]

        hybrite_pivot = species_data[species_data["model"] == "hybrite"].pivot_table(
            index="target_index", values="pearson", aggfunc="mean"
        )

        results.append(f"Species: {species.capitalize()}\n")
        results.append("-" * 50 + "\n")
        results.append(f"Number of targets (N): {len(hybrite_pivot)}\n\n")

        for model in ["ribonn", "saluki"]:
            other_pivot = species_data[species_data["model"] == model].pivot_table(
                index="target_index", values="pearson", aggfunc="mean"
            )

            common_targets = hybrite_pivot.index.intersection(other_pivot.index)
            hybrite_values = hybrite_pivot.loc[common_targets, "pearson"].values
            other_values = other_pivot.loc[common_targets, "pearson"].values

            try:
                stat, p_val = wilcoxon(
                    hybrite_values, other_values, alternative="greater"
                )
                num_comparisons = 4
                p_val = min(p_val * num_comparisons, 1.0)

                hybrite_mean = np.mean(hybrite_values)
                other_mean = np.mean(other_values)
                hybrite_wins = np.sum(hybrite_values > other_values)
                other_wins = np.sum(other_values > hybrite_values)

                results.append(
                    f"  {MODEL_LABELS['hybrite']} vs {MODEL_LABELS[model]}:\n"
                )
                results.append(f"    Aligned targets: {len(common_targets)}\n")
                results.append(
                    f"    {MODEL_LABELS['hybrite']} mean: {hybrite_mean:.4f}\n"
                )
                results.append(f"    {MODEL_LABELS[model]} mean: {other_mean:.4f}\n")
                results.append(
                    f"    {MODEL_LABELS['hybrite']} wins: "
                    f"{hybrite_wins}/{len(common_targets)}\n"
                )
                results.append(
                    f"    {MODEL_LABELS[model]} wins: "
                    f"{other_wins}/{len(common_targets)}\n"
                )
                results.append(f"    Wilcoxon statistic: {stat:.2f}\n")
                results.append(f"    p-value (Bonferroni adjusted, N=4): {p_val:.6e}\n")
                alpha = 0.05
                significance_label = "Yes" if p_val < alpha else "No"
                results.append(
                    f"    Significance (alpha=0.05): {significance_label}\n\n"
                )

            except Exception as e:
                results.append(
                    f"  Error: {MODEL_LABELS['hybrite']} "
                    f"vs {MODEL_LABELS[model]}: {e}\n\n"
                )

        results.append("\n")

    with open("stats_results.txt", "w") as f:
        f.writelines(results)
    print("Saved stats_results.txt")


def compute_ablation_stats() -> None:
    print("Calculating paired t-test statistics (ablation study)...")
    metrics_df = pd.read_csv(ABLATION_DIR / "agg_metrics.csv")
    results = []

    results.append(
        "\n\nPaired t-test Results (Ablation Study - Cross-Validation Pearson)\n"
    )
    results.append("=" * 70 + "\n\n")

    folds = [f"fold_0{i}" for i in range(10)]

    for species in ["human", "mouse"]:
        species_data = metrics_df[
            (metrics_df["species"] == species) & (metrics_df["metric"] == "pearson")
        ]

        results.append(f"Species: {species.capitalize()}\n")
        results.append("-" * 50 + "\n")

        full_data = species_data[species_data["model"] == "hybrite"]
        full_values = (
            full_data.set_index("fold").loc[folds, "value"].values.astype(float)
        )

        for ablation_model in ["hybrite_nobio", "hybrite_nornaplfold"]:
            ablation_data = species_data[species_data["model"] == ablation_model]
            ablation_values = (
                ablation_data.set_index("fold").loc[folds, "value"].values.astype(float)
            )

            full_mean = np.mean(full_values)
            ablation_mean = np.mean(ablation_values)

            t_stat, p_val = ttest_rel(full_values, ablation_values)
            num_comparisons = 4
            p_val = min(p_val * num_comparisons, 1.0)

            ablation_label = ABLATION_LABELS[ablation_model]
            results.append(f"  Full vs. {ablation_label}:\n")
            results.append(f"    Full mean: {full_mean:.4f}\n")
            results.append(f"    {ablation_label} mean: {ablation_mean:.4f}\n")
            results.append(f"    t-statistic: {t_stat:.2f}\n")
            results.append(f"    p-value (Bonferroni adjusted, N=4): {p_val:.6e}\n")
            alpha = 0.05
            significance_label = "Yes" if p_val < alpha else "No"
            results.append(f"    Significance (alpha=0.05): {significance_label}\n\n")

        results.append("\n")

    with open("stats_results.txt", "a") as f:
        f.writelines(results)
    print("Appended ablation t-test results to stats_results.txt")


def figure_performance_analysis() -> None:
    print("Generating Performance Analysis Figure...")
    cv_summary = load_cv_summary()
    ablation_summary = load_ablation_summary()
    target_metrics = load_target_metrics()
    preds_df = load_predictions()
    gt_df = load_ground_truth()

    fig = plt.figure(figsize=(210 * MM_TO_INCH, 200 * MM_TO_INCH))
    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        hspace=0.6,
        wspace=0.4,
        left=0.06,
        right=0.98,
        top=0.95,
        bottom=0.05,
    )
    panel_labels = list("ABCDEFGHIJKL")

    models = ["hybrite", "ribonn", "saluki"]

    def add_label(ax, label):
        ax.text(
            -0.2,
            1.1,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    # --- Row 1: Scatter & Per-Target ---
    # A & B: Scatter
    for i, species in enumerate(["human", "mouse"]):
        ax = fig.add_subplot(gs[0, i])
        p = preds_df[
            (preds_df["species"] == species) & (preds_df["model"] == "hybrite")
        ]
        g = gt_df[(gt_df["species"] == species) & (gt_df["model"] == "hybrite")]
        meta = load_metadata(species)
        cols = [str(k) for k in range(len(meta["target_columns"]))]

        try:
            p_mean = np.nanmean(p[cols].values, axis=1)
            g_mean = np.nanmean(g[cols].values, axis=1)
            mask = ~(np.isnan(p_mean) | np.isnan(g_mean))
            p_mean, g_mean = p_mean[mask], g_mean[mask]
            idx = np.arange(len(p_mean))

            ax.hexbin(
                g_mean[idx],
                p_mean[idx],
                gridsize=30,
                cmap="plasma",
                mincnt=1,
                alpha=0.9,
            )

            r = pearsonr(g_mean[idx], p_mean[idx])[0]
            r_squared = r2_score(g_mean[idx], p_mean[idx])
            rho = spearmanr(g_mean[idx], p_mean[idx])[0]

            lims = [min(g_mean.min(), p_mean.min()), max(g_mean.max(), p_mean.max())]
            ax.plot(lims, lims, "--", lw=1, color="red")

            ax.text(
                0.05,
                0.95,
                f"r={r:.2f}\nρ={rho:.2f}\nR²={r_squared:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )
        except Exception as e:
            ax.text(0.5, 0.5, "No Data")
            print(f"Scatter Error {species}: {e}")

        ax.set_xlabel("Measured TE")
        ax.set_ylabel("Predicted TE")
        ax.set_title(
            f"{MODEL_LABELS['hybrite']} Scatter ({species[0].upper()})",
            fontweight="bold",
            fontsize=9,
        )
        add_label(ax, panel_labels[i])

    # C & D: Per-Target
    for i, species in enumerate(["human", "mouse"]):
        ax = fig.add_subplot(gs[0, i + 2])
        vdata = []
        for m in models:
            d = target_metrics[
                (target_metrics["species"] == species) & (target_metrics["model"] == m)
            ]["pearson"].values
            vdata.append(d)

        parts = ax.violinplot(vdata, positions=range(len(models)), showmeans=True)
        for pc, m in zip(parts["bodies"], models):
            pc.set_facecolor(MODEL_COLORS[m])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=7)
        ax.set_ylabel("Pearson")
        ax.set_title(
            f"Per-Target ({species[0].upper()})", fontweight="bold", fontsize=9
        )
        add_label(ax, panel_labels[2 + i])

    # --- Row 2: Ablation & Best/Worst ---
    # E & F: Ablation
    variants = [
        "hybrite",
        "hybrite_nobio",
        "hybrite_nornaplfold",
        "hybrite_nornaplfold_nobio",
    ]
    for i, species in enumerate(["human", "mouse"]):
        ax = fig.add_subplot(gs[1, i])
        means, stds = [], []
        for v in variants:
            d = ablation_summary[
                (ablation_summary["species"] == species)
                & (ablation_summary["model"] == v)
                & (ablation_summary["metric"] == "pearson")
            ]
            if len(d) == 0:
                means.append(0)
                stds.append(0)
            else:
                means.append(d["mean"].values[0])
                stds.append(d["std"].values[0])

        ax.bar(
            np.arange(len(variants)),
            means,
            yerr=stds,
            capsize=2,
            color=[ABLATION_COLORS[v] for v in variants],
            edgecolor="black",
            lw=0.5,
        )

        base = means[0]
        if base > 0:
            for k in range(1, len(means)):
                if means[k] > 0:
                    diff = means[k] - base
                    txt = f"$\\Delta$={diff:.2f}" if diff < 0 else f"+{diff:.2f}"
                    ax.text(
                        k,
                        means[k] + stds[k] + 0.02,
                        txt,
                        ha="center",
                        fontsize=8,
                        color="black",
                    )

        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(
            [ABLATION_LABELS[v] for v in variants], rotation=30, ha="right", fontsize=6
        )
        ax.set_title(f"Ablation ({species[0].upper()})", fontweight="bold", fontsize=9)
        ax.set_ylim(0, 1.05)
        add_label(ax, panel_labels[4 + i])

    # G & H: Best/Worst
    for i, species in enumerate(["human", "mouse"]):
        ax = fig.add_subplot(gs[1, i + 2])
        meta = load_metadata(species)
        df = target_metrics[
            (target_metrics["species"] == species)
            & (target_metrics["model"] == "hybrite")
        ]
        df_g = df.groupby("target_index")["pearson"].mean().reset_index()
        df_g["name"] = df_g["target_index"].apply(
            lambda x: meta["target_columns"][x]
            .replace("bio_source_", "")
            .replace("_", " ")[:22]
        )

        top = df_g.nlargest(5, "pearson")
        bot = df_g.nsmallest(5, "pearson")
        comb = pd.concat([top, bot])

        y = np.arange(len(comb))
        cols_bar = [C_TEAL] * 5 + [C_VERMILION] * 5
        ax.barh(y, comb["pearson"], color=cols_bar, edgecolor="black", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(comb["name"], fontsize=4.5)
        ax.set_title(
            f"Best/Worst Cell Types ({species[0].upper()})",
            fontweight="bold",
            fontsize=9,
        )
        ax.set_xlabel("Pearson Correlation", fontsize=7)
        ax.set_xlim(0, 1.0)
        add_label(ax, panel_labels[6 + i])

    # --- Row 3: CV Stability & Transfer ---
    # I & J: CV Stability Box Plots
    fold_cols = [f"fold_0{k}" for k in range(10)]
    for i, species in enumerate(["human", "mouse"]):
        ax = fig.add_subplot(gs[2, i])
        data = []
        for model in models:
            d = cv_summary[
                (cv_summary["species"] == species)
                & (cv_summary["model"] == model)
                & (cv_summary["metric"] == "pearson")
            ]
            if len(d) > 0:
                data.append(d[fold_cols].values[0])
            else:
                data.append([])

        bplot = ax.boxplot(
            data,
            patch_artist=True,
            labels=[MODEL_LABELS[m] for m in models],
            widths=0.5,
        )

        for patch, model in zip(bplot["boxes"], models):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.7)

        ax.set_title(
            f"CV Stability ({species[0].upper()})", fontweight="bold", fontsize=9
        )
        ax.set_ylabel("Pearson")
        ax.tick_params(axis="x", labelsize=7)
        add_label(ax, panel_labels[8 + i])

    # K & L: Transfer (Split by Train Model Species)
    cross_h = load_cross_species_summary("cross_species_human")
    cross_m = load_cross_species_summary("cross_species_mouse")

    # K: Human Models
    ax_k = fig.add_subplot(gs[2, 2])
    x_pos = np.arange(len(models))
    width = 0.35

    hh_vals = []
    for m in models:
        d = cv_summary[
            (cv_summary["species"] == "human")
            & (cv_summary["model"] == m)
            & (cv_summary["metric"] == "pearson")
        ]
        hh_vals.append(d["mean"].values[0] if len(d) > 0 else 0)

    hm_vals = []
    for m in models:
        d = cross_h[(cross_h["model"] == m) & (cross_h["metric"] == "pearson")]
        hm_vals.append(d["mean"].values[0] if len(d) > 0 else 0)

    ax_k.bar(
        x_pos - width / 2,
        hh_vals,
        width,
        label="Test Human",
        color=C_BLUE,
        alpha=0.9,
    )
    ax_k.bar(
        x_pos + width / 2,
        hm_vals,
        width,
        label="Test Mouse",
        color=C_VERMILION,
        alpha=0.9,
    )

    ax_k.set_xticks(x_pos)
    ax_k.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=7)
    ax_k.set_title("Human Models Transfer", fontweight="bold", fontsize=9)
    ax_k.legend(fontsize=5, frameon=False)
    ax_k.set_ylim(0.4, 0.9)
    add_label(ax_k, panel_labels[10])

    # L: Mouse Models
    ax_l = fig.add_subplot(gs[2, 3])

    mm_vals = []
    for m in models:
        d = cv_summary[
            (cv_summary["species"] == "mouse")
            & (cv_summary["model"] == m)
            & (cv_summary["metric"] == "pearson")
        ]
        mm_vals.append(d["mean"].values[0] if len(d) > 0 else 0)

    mh_vals = []
    for m in models:
        d = cross_m[(cross_m["model"] == m) & (cross_m["metric"] == "pearson")]
        mh_vals.append(d["mean"].values[0] if len(d) > 0 else 0)

    ax_l.bar(
        x_pos - width / 2,
        mm_vals,
        width,
        label="Test Mouse",
        color=C_BLUE,
        alpha=0.9,
    )
    ax_l.bar(
        x_pos + width / 2,
        mh_vals,
        width,
        label="Test Human",
        color=C_VERMILION,
        alpha=0.9,
    )

    ax_l.set_xticks(x_pos)
    ax_l.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=7)
    ax_l.set_title("Mouse Models Transfer", fontweight="bold", fontsize=9)
    ax_l.legend(fontsize=5, frameon=False)
    ax_l.set_ylim(0.4, 0.9)
    add_label(ax_l, panel_labels[11])

    plt.savefig(FIGURE_DIR / "performance_analysis.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURE_DIR / "performance_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def figure_shap_analysis() -> None:
    print("Generating SHAP Analysis Figure...")
    fig = plt.figure(figsize=(220 * MM_TO_INCH, 120 * MM_TO_INCH))
    gs = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        hspace=0.4,
        wspace=0.35,
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.08,
    )
    panel_labels = list("ABCDEFGH")

    def add_label(ax, label):
        ax.text(
            -0.2,
            1.1,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    BIO_CATS = {
        "encori": (C_LIGHT_BLUE, "ENCORI"),
        "clip": (C_TEAL, "eCLIP"),
        "m6a": (C_VERMILION, "m6A"),
        "mirna": (C_PURPLE, "miRNA"),
        "other": (C_GREY, "Other"),
    }

    def get_bio_cat(name):
        lower = name.lower()
        if "encori" in lower:
            return "encori"
        if "clip" in lower:
            return "clip"
        if "m6a" in lower:
            return "m6a"
        if "mirna" in lower or "targetscan" in lower:
            return "mirna"
        return "other"

    def clean_bio_name(name):
        parts = name.split("_")
        if "biochemical" in parts:
            parts.remove("biochemical")
        if "human" in parts:
            parts.remove("human")
        if "mouse" in parts:
            parts.remove("mouse")

        parts = "".join(parts).split(".")
        if "clip" in name.lower():
            if len(parts) >= 2:
                return f"{parts[-2]} {parts[-1]}"
            return parts[-1]

        if "encori" in name.lower():
            return parts[-1]

        if "m6a" in name.lower():
            return "m6A"

        return parts[-1][:15]

    species_list = ["human", "mouse"]

    # --- Row 1: Bio Features & Node Features ---
    for i, species in enumerate(species_list):
        # A & B: Bio Features
        try:
            shap_bio, _, bio_names = load_shap_bio_data(species)
            mean_shap = np.abs(shap_bio).mean(axis=0)
            top_idx = np.argsort(mean_shap)[-15:][::-1]

            ax_bio = fig.add_subplot(gs[0, i])

            y_pos = np.arange(len(top_idx))
            vals = mean_shap[top_idx]
            colors = []
            clean_labels = []

            for idx in top_idx:
                raw = bio_names[idx]
                cat = get_bio_cat(raw)
                colors.append(BIO_CATS[cat][0])
                clean_labels.append(clean_bio_name(raw))

            ax_bio.barh(y_pos, vals, color=colors, edgecolor="black", lw=0.3)
            ax_bio.set_yticks(y_pos)
            ax_bio.set_yticklabels(clean_labels, fontsize=5)
            ax_bio.invert_yaxis()
            ax_bio.set_xlabel("Mean |SHAP| Importance", fontsize=6)
            ax_bio.set_title(
                f"Biochemical Features ({species[0].upper()})",
                fontweight="bold",
                fontsize=9,
            )

            patches_ = [
                patches.Patch(color=BIO_CATS["encori"][0], label="ENCORI"),
                patches.Patch(color=BIO_CATS["clip"][0], label="eCLIP"),
            ]
            ax_bio.legend(
                handles=patches_, loc="lower right", fontsize=5, frameon=False
            )

            add_label(ax_bio, panel_labels[i])
        except Exception as e:
            print(f"Error Bio {species}: {e}")

        # C & D: Node Features
        try:
            graph_data = load_shap_graph_data(species)
            node_imp = graph_data["node_feature"]
            ax_node = fig.add_subplot(gs[0, i + 2])

            cols = []
            labels = (
                NODE_FEATURE_NAMES
                if len(node_imp) == len(NODE_FEATURE_NAMES)
                else [str(k) for k in range(len(node_imp))]
            )

            for k in range(len(node_imp)):
                if k < 7:
                    cols.append(C_GREY)
                elif k < 13:
                    cols.append(C_SKY)
                elif k < 19:
                    cols.append(C_ORANGE)
                else:
                    cols.append(C_VERMILION)

            ax_node.barh(
                range(len(node_imp)), node_imp, color=cols, edgecolor="black", lw=0.3
            )
            ax_node.set_yticks(range(len(node_imp)))
            ax_node.set_yticklabels(labels, fontsize=5)
            ax_node.invert_yaxis()
            ax_node.set_xlabel("Importance", fontsize=6, loc="left")
            ax_node.set_title(
                f"Node Features ({species[0].upper()})", fontweight="bold", fontsize=9
            )

            feat_legs = [
                patches.Patch(color=C_GREY, label="Base"),
                patches.Patch(color=C_SKY, label="5'UTR"),
                patches.Patch(color=C_ORANGE, label="CDS"),
                patches.Patch(color=C_VERMILION, label="3'UTR"),
            ]
            ax_node.legend(
                handles=feat_legs,
                loc="upper right",
                bbox_to_anchor=(1.0, -0.1),
                fontsize=5,
                frameon=False,
                ncol=2,
            )

            add_label(ax_node, panel_labels[i + 2])

        except Exception as e:
            print(f"Error Node {species}: {e}")

    # --- Row 2: Region & Edges ---
    for i, species in enumerate(species_list):
        # E & F: Region Importance
        try:
            graph_data = load_shap_graph_data(species)
            loc_imp = graph_data["node_location"]
            ax_reg = fig.add_subplot(gs[1, i])
            x = np.arange(len(loc_imp))

            ax_reg.axvspan(-0.5, 7.5, color=C_SKY, alpha=0.2, label="5'UTR")
            ax_reg.axvspan(7.5, 39.5, color=C_ORANGE, alpha=0.2, label="CDS")
            ax_reg.axvspan(39.5, 55.5, color=C_VERMILION, alpha=0.2, label="3'UTR")

            ax_reg.plot(x, loc_imp, color="black", lw=0.5, alpha=0.6)
            ax_reg.scatter(
                x,
                loc_imp,
                s=10,
                c=loc_imp,
                cmap="viridis",
                zorder=5,
                edgecolor="k",
                lw=0.1,
            )

            ax_reg.set_xlim(-0.5, 55.5)
            ax_reg.set_ylabel("Importance")
            ax_reg.set_xlabel("Node Location")
            ax_reg.set_title(
                f"Regional Importance ({species[0].upper()})",
                fontweight="bold",
                fontsize=9,
            )

            ax_reg.legend(
                loc="upper right",
                bbox_to_anchor=(1.05, -0.1),
                ncol=1,
                fontsize=5,
                frameon=False,
            )
            add_label(ax_reg, panel_labels[i + 4])

        except Exception:
            pass

        # G & H: Top 500 Edges
        try:
            with open(
                SHAP_BASE_DIR / species / "graph_shap_edge_scores_mean.json"
            ) as f:
                edge_scores = json.load(f)

            ax_edge = fig.add_subplot(gs[1, i + 2])
            pattern = re.compile(r"(\w+) \((\d+)\) -> (\w+) \((\d+)\) \((.+)\)")
            parsed_edges = []
            for item in edge_scores:
                match = pattern.match(item["label"])
                if match:
                    _, src_idx, _, tgt_idx, edge_type = match.groups()
                    typ_clean = edge_type.lower().replace(" ", "_").split("_")[0]
                    if "long" in edge_type.lower():
                        typ_clean = "long_range"
                    parsed_edges.append(
                        {
                            "source": int(src_idx),
                            "target": int(tgt_idx),
                            "importance": abs(item["score"]),
                            "edge_type": typ_clean,
                        }
                    )

            top_edges = sorted(
                parsed_edges, key=lambda x: x["importance"], reverse=True
            )[:500]

            type_map = {
                "sequential": ("o", C_BLUE, "Seq"),
                "structural": ("^", C_TEAL, "Struct"),
                "long_range": ("*", C_GREY, "Long-R"),
            }

            ax_edge.axvspan(-0.5, 7.5, color=C_SKY, alpha=0.15)
            ax_edge.axhspan(-0.5, 7.5, color=C_SKY, alpha=0.15)
            ax_edge.axvspan(7.5, 39.5, color=C_ORANGE, alpha=0.15)
            ax_edge.axhspan(7.5, 39.5, color=C_ORANGE, alpha=0.15)
            ax_edge.axvspan(39.5, 55.5, color=C_VERMILION, alpha=0.15)
            ax_edge.axhspan(39.5, 55.5, color=C_VERMILION, alpha=0.15)

            for e in top_edges:
                marker, color, _ = type_map.get(e["edge_type"], ("o", "black", "Other"))
                s = (
                    e["importance"] / max([x["importance"] for x in top_edges])
                ) * 15 + 2
                ax_edge.scatter(
                    e["source"],
                    e["target"],
                    s=s,
                    color=color,
                    marker=marker,
                    alpha=0.8 if e["edge_type"] != "long_range" else 1.0,
                    edgecolors="none",
                )

            ax_edge.plot([0, 55], [0, 55], "--", color="grey", lw=0.5, alpha=0.3)
            ax_edge.set_xlim(-1, 56)
            ax_edge.set_ylim(-1, 56)
            ax_edge.set_xlabel("Source Node Location")
            ax_edge.set_ylabel("Target Node Location")
            ax_edge.set_title(
                f"Top 500 Edges ({species[0].upper()})", fontweight="bold", fontsize=9
            )

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=v[0],
                    color="w",
                    markerfacecolor=v[1],
                    label=v[2],
                    markersize=5,
                )
                for k, v in type_map.items()
            ]
            ax_edge.legend(
                handles=handles, loc="lower right", fontsize=5, frameon=False
            )

            add_label(ax_edge, panel_labels[i + 6])

        except Exception as e:
            print(f"Error Edges {species}: {e}")

    plt.savefig(FIGURE_DIR / "shap_analysis.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURE_DIR / "shap_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def _prepare_per_target_data(species: str) -> pd.DataFrame:
    target_metrics = load_target_metrics()
    meta = load_metadata(species)

    df_species = target_metrics[target_metrics["species"] == species]
    pivot = df_species.pivot_table(
        index="target_index", columns="model", values="pearson", aggfunc="mean"
    )

    pivot["name"] = pivot.index.map(
        lambda x: meta["target_columns"][x]
        .replace("bio_source_", "")
        .replace("_", " ")[:35]
    )

    pivot["best_baseline"] = pivot[["ribonn", "saluki"]].max(axis=1)
    pivot["baseline_wins"] = pivot["best_baseline"] > pivot["hybrite"]

    return pivot.sort_values("hybrite", ascending=True).reset_index()


def _create_per_target_lollipop(
    species: str,
    height_mm: float,
    output_filename: str,
) -> None:
    pivot_sorted = _prepare_per_target_data(species)
    n_targets = len(pivot_sorted)

    fig, ax = plt.subplots(figsize=(180 * MM_TO_INCH, height_mm * MM_TO_INCH))

    y_positions = np.arange(n_targets)
    models = ["hybrite", "ribonn", "saluki"]
    markers = {"hybrite": "o", "ribonn": "s", "saluki": "^"}

    all_vals = np.concatenate(
        [
            pivot_sorted["hybrite"].values,
            pivot_sorted["ribonn"].values,
            pivot_sorted["saluki"].values,
        ]
    )
    data_min = np.nanmin(all_vals)
    data_max = np.nanmax(all_vals)
    x_min = max(0.0, np.floor(data_min * 10) / 10 - 0.05)
    x_max = min(1.0, data_max + 0.05)

    for i in range(0, n_targets, 2):
        ax.axhspan(i - 0.5, i + 0.5, color="#f5f5f5", zorder=0)

    grid_start = int(x_min * 10 + 1) / 10
    grid_end = int(x_max * 10) / 10
    grid_lines = np.arange(grid_start, grid_end + 0.05, 0.1)
    for x_grid in grid_lines:
        ax.axvline(x_grid, color=C_GREY, lw=0.3, alpha=0.2, zorder=0)

    model_means = {}
    for model in models:
        model_means[model] = pivot_sorted[model].mean()
        ax.axvline(
            model_means[model],
            color=MODEL_COLORS[model],
            ls="--",
            lw=1.2,
            alpha=0.8,
            zorder=0,
        )

    for y_idx, row in pivot_sorted.iterrows():
        model_vals = [row["hybrite"], row["ribonn"], row["saluki"]]
        ax.plot(
            model_vals,
            [y_idx, y_idx, y_idx],
            color=C_GREY,
            lw=0.3,
            alpha=0.5,
            zorder=1,
        )

    for y_idx, row in pivot_sorted.iterrows():
        for model in models:
            ax.plot(
                [x_min, row[model]],
                [y_idx, y_idx],
                color=MODEL_COLORS[model],
                lw=0.5,
                alpha=0.3,
                zorder=1,
            )

    for model in models:
        ax.scatter(
            pivot_sorted[model],
            y_positions,
            c=MODEL_COLORS[model],
            marker=markers[model],
            s=25 if model == "hybrite" else 20,
            label=MODEL_LABELS[model],
            zorder=3 if model == "hybrite" else 2,
            edgecolors="white",
            linewidths=0.3,
        )

    y_labels = [row["name"] for _, row in pivot_sorted.iterrows()]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=6, fontfamily="Arial")

    ax.set_xlabel("Pearson Correlation", fontsize=8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, n_targets - 0.5)

    ax.set_title(
        f"Per-Target Performance ({species.capitalize()})",
        fontweight="bold",
        fontsize=10,
    )

    ax.text(
        0.99,
        5.5,
        "Mean Values",
        fontsize=7,
        fontweight="bold",
        fontstyle="italic",
        va="center",
        ha="right",
        color=C_GREY,
        transform=ax.get_yaxis_transform(),
        zorder=10,
    )

    mean_y_positions = [4.5, 3.5, 2.5]
    for idx, model in enumerate(models):
        ax.text(
            0.99,
            mean_y_positions[idx],
            f"{MODEL_LABELS[model]}: {model_means[model]:.3f}",
            fontsize=7,
            fontweight="bold",
            va="center",
            ha="right",
            color=MODEL_COLORS[model],
            transform=ax.get_yaxis_transform(),
            zorder=10,
        )

    ax.legend(
        loc="lower right",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        ncol=3,
        borderaxespad=0.5,
    )

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_filename, dpi=300, bbox_inches="tight")
    png_filename = output_filename.replace(".pdf", ".png")
    plt.savefig(FIGURE_DIR / png_filename, dpi=300, bbox_inches="tight")
    plt.close()


def _compute_target_correlation_matrix(species: str) -> tuple[np.ndarray, list[str]]:
    preds_df = load_predictions()
    preds_hybrite = preds_df[
        (preds_df["species"] == species) & (preds_df["model"] == "hybrite")
    ]

    meta = load_metadata(species)
    n_targets = len(meta["target_columns"])
    target_cols = [str(i) for i in range(n_targets)]

    pred_matrix = preds_hybrite[target_cols].values
    corr_matrix = np.corrcoef(pred_matrix.T)

    target_names = [
        meta["target_columns"][i].replace("bio_source_", "").replace("_", " ")[:25]
        for i in range(n_targets)
    ]

    return corr_matrix, target_names


def figure_supp_per_target_human() -> None:
    print("Generating Supplementary Figure: Per-Target Performance (Human)...")
    _create_per_target_lollipop(
        species="human",
        height_mm=350,
        output_filename="supp_per_target_human.pdf",
    )


def figure_supp_per_target_mouse() -> None:
    print("Generating Supplementary Figure: Per-Target Performance (Mouse)...")
    _create_per_target_lollipop(
        species="mouse",
        height_mm=320,
        output_filename="supp_per_target_mouse.pdf",
    )


def figure_supp_correlation_human() -> None:
    print("Generating Supplementary Figure: Target Correlation (Human)...")

    corr_human, names_human = _compute_target_correlation_matrix("human")

    correlation_cmap = LinearSegmentedColormap.from_list(
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

    fig, ax_human = plt.subplots(figsize=(450 * MM_TO_INCH, 450 * MM_TO_INCH))

    fig.patch.set_facecolor("none")
    ax_human.patch.set_facecolor("none")

    mask_human = np.triu(np.ones_like(corr_human, dtype=bool), k=0)
    corr_human_abs = np.abs(corr_human)
    corr_human_masked = np.where(mask_human, np.nan, corr_human_abs)

    im_human = ax_human.imshow(
        corr_human_masked, cmap=correlation_cmap, vmin=0, vmax=1, aspect="equal"
    )

    ax_human.set_xticks(range(len(names_human)))
    ax_human.set_xticklabels(
        names_human, rotation=45, ha="right", fontsize=5, fontweight="medium"
    )
    ax_human.set_yticks(range(len(names_human)))
    ax_human.set_yticklabels(names_human, fontsize=5, fontweight="medium")

    for spine in ax_human.spines.values():
        spine.set_visible(False)

    ax_human.set_title(
        "Human Target Correlation",
        fontweight="bold",
        fontsize=12,
        loc="left",
    )

    cbar_human = fig.colorbar(
        im_human,
        ax=ax_human,
        orientation="horizontal",
        shrink=0.3,
        aspect=40,
        pad=0.08,
    )
    cbar_human.ax.tick_params(labelsize=8)
    cbar_human.set_label("Absolute Correlation", fontsize=9, fontweight="medium")

    plt.tight_layout()
    pdf_path = FIGURE_DIR / "supp_target_correlation_human.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", transparent=True)
    png_path = FIGURE_DIR / "supp_target_correlation_human.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    svg_path = FIGURE_DIR / "supp_target_correlation_human.svg"
    plt.savefig(svg_path, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


def figure_supp_correlation_mouse() -> None:
    print("Generating Supplementary Figure: Target Correlation (Mouse)...")

    corr_mouse, names_mouse = _compute_target_correlation_matrix("mouse")

    correlation_cmap = LinearSegmentedColormap.from_list(
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

    fig, ax_mouse = plt.subplots(figsize=(400 * MM_TO_INCH, 400 * MM_TO_INCH))

    fig.patch.set_facecolor("none")
    ax_mouse.patch.set_facecolor("none")

    mask_mouse = np.triu(np.ones_like(corr_mouse, dtype=bool), k=0)
    corr_mouse_abs = np.abs(corr_mouse)
    corr_mouse_masked = np.where(mask_mouse, np.nan, corr_mouse_abs)

    im_mouse = ax_mouse.imshow(
        corr_mouse_masked, cmap=correlation_cmap, vmin=0, vmax=1, aspect="equal"
    )

    ax_mouse.set_xticks(range(len(names_mouse)))
    ax_mouse.set_xticklabels(
        names_mouse, rotation=45, ha="right", fontsize=5, fontweight="medium"
    )
    ax_mouse.set_yticks(range(len(names_mouse)))
    ax_mouse.set_yticklabels(names_mouse, fontsize=5, fontweight="medium")

    for spine in ax_mouse.spines.values():
        spine.set_visible(False)

    ax_mouse.set_title(
        "Mouse Target Correlation",
        fontweight="bold",
        fontsize=12,
        loc="left",
    )

    cbar_mouse = fig.colorbar(
        im_mouse,
        ax=ax_mouse,
        orientation="horizontal",
        shrink=0.3,
        aspect=40,
        pad=0.08,
    )
    cbar_mouse.ax.tick_params(labelsize=8)
    cbar_mouse.set_label("Absolute Correlation", fontsize=9, fontweight="medium")

    plt.tight_layout()
    pdf_path = FIGURE_DIR / "supp_target_correlation_mouse.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", transparent=True)
    png_path = FIGURE_DIR / "supp_target_correlation_mouse.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    svg_path = FIGURE_DIR / "supp_target_correlation_mouse.svg"
    plt.savefig(svg_path, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


def main():
    print("=" * 60)
    print("Generating Figures")
    print("=" * 60)

    compute_wilcoxon_stats()
    compute_ablation_stats()
    figure_performance_analysis()
    figure_shap_analysis()
    figure_supp_per_target_human()
    figure_supp_per_target_mouse()
    figure_supp_correlation_human()
    figure_supp_correlation_mouse()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

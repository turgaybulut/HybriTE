from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
import torch.nn as nn
import yaml
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from framework.utils import seed_everything
from plugins.hybrite.model import HybriTEModel

seed_everything(654)

plt.style.use("seaborn-v0_8-paper")
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

BASE_FEATS = ["A_prop", "U_prop", "G_prop", "C_prop", "log(Len)", "Pos", "Unpaired"]
UTR5_FEATS = ["CpG_den", "uORF_cnt", "Kozak_cnt", "TOP_cnt", "G4_den", "uAUG_dist"]
CDS_FEATS = ["tAI", "CSC", "GC3", "Rare_codon", "Ramp", "Basic_den"]
UTR3_FEATS = [
    "ARE_cnt",
    "RBP_destab",
    "miRNA_cnt",
    "PolyA_cnt",
    "m6A_den",
    "Tail_AU",
]

EDGE_TYPE_MAP = {0: "Sequential", 1: "Structural", 2: "Long Range"}
TOP_K_FEATURES = 20
EDGE_TOP_K = 20
TEST_SAMPLE_SAVE = "bio_test_samples.npy"
FEATURE_NAME_SAVE = "bio_feature_names.json"


@dataclass
class BioImportance:
    shap_values: np.ndarray
    test_samples: np.ndarray
    feature_names: list[str]


@dataclass
class GraphImportance:
    node_mask_mean: np.ndarray
    node_location_mean: np.ndarray
    edge_type_mean: np.ndarray
    node_feature_mean: np.ndarray
    edge_scores: list[dict[str, Any]]


def node_region_label(index: int) -> str:
    if index < 8:
        region = "5'UTR"
    elif index < 40:
        region = "CDS"
    else:
        region = "3'UTR"
    return f"{region} ({index})"


def load_config(checkpoint_dir: Path) -> dict[str, Any]:
    hparams_path = checkpoint_dir / "hparams.yaml"
    if not hparams_path.exists():
        if (checkpoint_dir.parent / "hparams.yaml").exists():
            hparams_path = checkpoint_dir.parent / "hparams.yaml"
        else:
            hparams_path = Path("models/human/hybrite/fold_00/hparams.yaml")

    with open(hparams_path) as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    node_dim: int,
    edge_dim: int,
    bio_feat_dim: int,
    num_targets: int,
) -> HybriTEModel:
    kwargs = {
        "num_targets": num_targets,
        "node_feature_dim": node_dim,
        "edge_feature_dim": edge_dim,
        "biochemical_feature_dim": bio_feat_dim,
    }
    model = HybriTEModel(config, **kwargs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


class BioWrapper(nn.Module):
    def __init__(self, model: HybriTEModel, mean_graph_embedding: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean_graph_embedding", mean_graph_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.biochemical_features_mlp is not None:
            bio_feat = self.model.biochemical_features_mlp(x)
        else:
            bio_feat = x
        g_emb = self.mean_graph_embedding.expand(x.shape[0], -1)
        g = torch.cat([g_emb, bio_feat], dim=1)
        out = self.model.head(g)
        return out.mean(dim=1, keepdim=True)


class GraphWrapper(nn.Module):
    def __init__(self, model: HybriTEModel, mean_bio_embedding: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean_bio_embedding", mean_bio_embedding)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        curr_x = self.model.node_emb(x)
        edge_index = edge_index.to(curr_x.device)
        edge_attr = edge_attr.to(curr_x.device)
        struct_mask = edge_attr[:, 1] > 0
        encoded_edges = self.model.edge_emb(edge_attr)
        encoded_edges = self.model.edge_mlp(encoded_edges)
        if struct_mask.any():
            encoded_edges[struct_mask] = (
                encoded_edges[struct_mask] * self.model.struct_message_scale
            )
        for conv, norm in zip(self.model.convs, self.model.norms):
            curr_x = conv(x=curr_x, edge_index=edge_index, edge_attr=encoded_edges)
            curr_x = norm(curr_x, batch=batch)
        g = self.model.pool(curr_x, batch)
        b_emb = self.mean_bio_embedding.expand(g.shape[0], -1)
        combined = torch.cat([g, b_emb], dim=1)
        out = self.model.head(combined)
        return out.mean(dim=1, keepdim=True)


def get_device() -> torch.device:
    return torch.device("cpu")


def load_feature_names(meta_path: Path, feature_dim: int) -> list[str]:
    with open(meta_path) as f:
        meta = json.load(f)
    feature_names = meta.get(
        "feature_columns", [f"feat_{i}" for i in range(feature_dim)]
    )
    return [name.replace("biochemical_", "") for name in feature_names]


def precompute_embeddings(
    model: HybriTEModel, graphs: list[Any], features: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    graph_loader = DataLoader(graphs, batch_size=256, shuffle=False)
    graph_embeddings = []
    bio_embeddings = []
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        for i in range(0, len(features), 256):
            batch_x = features_tensor[i : i + 256]
            if model.biochemical_features_mlp:
                bio_embeddings.append(model.biochemical_features_mlp(batch_x).cpu())
            else:
                bio_embeddings.append(batch_x.cpu())

        for i, batch in enumerate(tqdm(graph_loader, desc="Graph Embeddings")):
            batch = batch.to(device)
            x = model.node_emb(batch.x)
            edge_index, edge_attr, struct_mask = model._prepare_edges(
                batch.edge_index, batch.edge_attr, x.device, False
            )
            edge_attr = model._encode_edges(edge_attr, struct_mask, x.device)
            for conv, norm in zip(model.convs, model.norms):
                x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
                x = norm(x, batch=batch.batch)
            g = model.pool(x, batch.batch)
            graph_embeddings.append(g.cpu())

    all_bio_emb = torch.cat(bio_embeddings, dim=0)
    all_graph_emb = torch.cat(graph_embeddings, dim=0)
    return all_bio_emb.mean(dim=0, keepdim=True).to(device), all_graph_emb.mean(
        dim=0, keepdim=True
    ).to(device)


def compute_bio_importance(
    model: HybriTEModel,
    mean_graph_emb: torch.Tensor,
    features: np.ndarray,
    subset_size: int,
    device: torch.device,
    feature_names: list[str],
) -> BioImportance:
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    bio_wrapper = BioWrapper(model, mean_graph_emb)
    bio_wrapper.eval()

    bg_indices = np.random.choice(len(features), subset_size, replace=False)
    background = features_tensor[bg_indices]
    explainer_bio = shap.GradientExplainer(bio_wrapper, background)

    test_indices = np.random.choice(
        len(features), min(subset_size, len(features)), replace=False
    )
    test_samples = features_tensor[test_indices]
    shap_values_bio = explainer_bio.shap_values(test_samples)
    shap_values_array = (
        shap_values_bio[0] if isinstance(shap_values_bio, list) else shap_values_bio
    )
    shap_values_array = np.asarray(shap_values_array)
    if shap_values_array.shape != test_samples.shape:
        shap_values_array = shap_values_array.squeeze()
    test_samples_np = test_samples.cpu().numpy()
    assert shap_values_array.shape[0] == test_samples_np.shape[0]

    names = (
        feature_names
        if len(feature_names) == shap_values_array.shape[1]
        else [f"Feat {i}" for i in range(shap_values_array.shape[1])]
    )
    return BioImportance(
        shap_values=shap_values_array,
        test_samples=test_samples_np,
        feature_names=names,
    )


def plot_bio_importance(data: BioImportance, output_dir: Path) -> None:
    mean_abs_shap = np.abs(data.shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-TOP_K_FEATURES:]

    plt.figure(figsize=(10, 12))
    plt.barh(
        [data.feature_names[i] for i in top_indices],
        mean_abs_shap[top_indices],
        color="#00509E",
    )
    plt.xlabel("Mean |SHAP Value| (Mean TE Prediction)")
    plt.title("Top 20 Important Biochemical Features")
    plt.tight_layout()
    plt.savefig(output_dir / "bio_feature_importance.pdf")
    plt.close()

    categories = {
        "ENCORI": r"ENCORI\.",
        "eCLIP": r"eCLIP\.",
        "M6ACLIP": r"M6ACLIP\.",
    }
    cat_importance = {}
    for name, pattern in categories.items():
        indices = [i for i, f in enumerate(data.feature_names) if re.search(pattern, f)]
        if indices:
            cat_importance[name] = float(np.mean(mean_abs_shap[indices]))

    if cat_importance:
        plt.figure(figsize=(8, 6))
        plt.bar(
            list(cat_importance.keys()),
            list(cat_importance.values()),
            color=["#009E73", "#D55E00", "#CC79A7"],
        )
        plt.ylabel("Average Feature Importance")
        plt.title("Importance by Biochemical Category")
        plt.tight_layout()
        plt.savefig(output_dir / "bio_category_importance.pdf")
        plt.close()

    plt.figure(figsize=(10, 10))
    shap.summary_plot(
        data.shap_values,
        data.test_samples,
        feature_names=data.feature_names,
        show=False,
    )
    plt.savefig(output_dir / "bio_shap_summary.pdf", bbox_inches="tight")
    plt.close()


def save_bio_importance(data: BioImportance, output_dir: Path) -> None:
    np.save(output_dir / "shap_values_bio.npy", data.shap_values)
    np.save(output_dir / TEST_SAMPLE_SAVE, data.test_samples)
    with open(output_dir / FEATURE_NAME_SAVE, "w") as f:
        json.dump(data.feature_names, f)


def load_bio_importance(source_dir: Path) -> BioImportance:
    shap_values = np.load(source_dir / "shap_values_bio.npy")
    test_samples = np.load(source_dir / TEST_SAMPLE_SAVE)
    with open(source_dir / FEATURE_NAME_SAVE) as f:
        feature_names = json.load(f)
    return BioImportance(
        shap_values=shap_values, test_samples=test_samples, feature_names=feature_names
    )


def compute_graph_importance(
    model: HybriTEModel,
    mean_bio_emb: torch.Tensor,
    graphs: list[Any],
    subset_size: int,
    device: torch.device,
    node_dim: int,
) -> GraphImportance:
    graph_wrapper = GraphWrapper(model, mean_bio_emb)
    graph_wrapper.eval()

    explainer_gnn = Explainer(
        model=graph_wrapper,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )

    n_utr5, n_cds, n_utr3 = 8, 32, 16
    total_nodes = n_utr5 + n_cds + n_utr3
    test_indices = np.random.choice(
        len(graphs), min(subset_size, len(graphs)), replace=False
    )

    agg_node_mask = torch.zeros(total_nodes, node_dim)
    agg_loc_importance = torch.zeros(total_nodes)
    edge_type_importance = {0: [], 1: [], 2: []}
    edge_aggregator: dict[tuple[int, int], dict[str, Any]] = {}

    for idx in tqdm(test_indices):
        data = graphs[idx].to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        explanation = explainer_gnn(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
        )
        node_mask = explanation.node_mask.cpu()
        edge_mask = explanation.edge_mask.cpu()
        if node_mask.shape[0] == total_nodes:
            agg_node_mask += node_mask
            agg_loc_importance += node_mask.mean(dim=1)
        edge_attrs = data.edge_attr.cpu()
        edge_indices = data.edge_index.cpu()
        if edge_attrs.shape[1] >= 3:
            types = torch.argmax(edge_attrs[:, :3], dim=1)
            for etype in [0, 1, 2]:
                mask_vals = edge_mask[types == etype]
                if mask_vals.numel() > 0:
                    edge_type_importance[etype].append(mask_vals.mean().item())
            for e_i in range(edge_indices.shape[1]):
                u = edge_indices[0, e_i].item()
                v = edge_indices[1, e_i].item()
                score = edge_mask[e_i].item()
                e_type_idx = types[e_i].item()
                key = (u, v)
                if key not in edge_aggregator:
                    edge_aggregator[key] = {"scores": [], "type": e_type_idx}
                edge_aggregator[key]["scores"].append(score)

    agg_node_mask /= len(test_indices)
    agg_loc_importance /= len(test_indices)

    final_edge_scores = []
    for (u, v), e_data in edge_aggregator.items():
        mean_score = float(np.mean(e_data["scores"]))
        type_name = EDGE_TYPE_MAP.get(e_data["type"], "Unknown")
        label = f"{node_region_label(u)} -> {node_region_label(v)} ({type_name})"
        final_edge_scores.append(
            {
                "label": label,
                "score": mean_score,
                "type_idx": e_data["type"],
            }
        )
    final_edge_scores.sort(key=lambda x: x["score"], reverse=True)

    base_imp = agg_node_mask[:, :7].mean(dim=0)
    utr5_imp = agg_node_mask[0:8, 7:].mean(dim=0)
    cds_imp = agg_node_mask[8:40, 7:].mean(dim=0)
    utr3_imp = agg_node_mask[40:56, 7:].mean(dim=0)
    all_imp = torch.cat([base_imp, utr5_imp, cds_imp, utr3_imp]).numpy()

    edge_type_mean = np.array(
        [
            np.mean(edge_type_importance[k]) if edge_type_importance[k] else 0
            for k in [0, 1, 2]
        ]
    )

    return GraphImportance(
        node_mask_mean=agg_node_mask.numpy(),
        node_location_mean=agg_loc_importance.numpy(),
        edge_type_mean=edge_type_mean,
        node_feature_mean=all_imp,
        edge_scores=final_edge_scores,
    )


def plot_graph_importance(data: GraphImportance, output_dir: Path) -> None:
    top_edges = data.edge_scores[:EDGE_TOP_K]
    if top_edges:
        plt.figure(figsize=(10, 8))
        plot_edges = top_edges[::-1]
        labels = [x["label"] for x in plot_edges]
        scores = [x["score"] for x in plot_edges]
        colors = [
            {
                "Sequential": "#7F7F7F",
                "Structural": "#D55E00",
                "Long Range": "#333333",
            }.get(EDGE_TYPE_MAP.get(x["type_idx"]), "#00509E")
            for x in plot_edges
        ]
        plt.barh(labels, scores, color=colors)
        plt.xlabel("Mean Importance Score")
        plt.title("Top 20 Important Graph Edges")
        plt.tight_layout()
        plt.savefig(output_dir / "graph_top_edges_importance.pdf")
        plt.close()

    all_feats = BASE_FEATS + UTR5_FEATS + CDS_FEATS + UTR3_FEATS
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(all_feats))
    colors = ["#00509E"] * 7 + ["#E69F00"] * 6 + ["#009E73"] * 6 + ["#CC79A7"] * 6
    plt.bar(x_pos, data.node_feature_mean, color=colors)
    plt.xticks(x_pos, all_feats, rotation=45, ha="right")
    plt.title("Global Graph Node Feature Importance (Averaged)")
    plt.ylabel("Importance Score")
    plt.legend(
        [
            plt.Rectangle((0, 0), 1, 1, color=c)
            for c in ["#00509E", "#E69F00", "#009E73", "#CC79A7"]
        ],
        ["Base", "UTR5 Specific", "CDS Specific", "UTR3 Specific"],
    )
    plt.tight_layout()
    plt.savefig(output_dir / "graph_node_feature_importance.pdf")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.bar(
        ["Sequential", "Structural", "Long Range"],
        data.edge_type_mean,
        color=["#7F7F7F", "#D55E00", "#333333"],
    )
    plt.ylabel("Mean Edge Importance")
    plt.title("Edge Type Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "graph_edge_type_importance.pdf")
    plt.close()

    plt.figure(figsize=(15, 6))
    plt.axvspan(-0.5, 7.5, facecolor="lightblue", alpha=0.3, label="UTR5")
    plt.axvspan(7.5, 39.5, facecolor="lightgreen", alpha=0.3, label="CDS")
    plt.axvspan(39.5, 55.5, facecolor="moccasin", alpha=0.3, label="UTR3")
    plt.plot(
        range(len(data.node_location_mean)),
        data.node_location_mean,
        marker="o",
        linestyle="-",
        linewidth=2,
        color="#333333",
    )
    plt.xlabel("Node Index (Sequence Bin)")
    plt.ylabel("Avg Importance Score")
    plt.title("Node Location Importance")
    plt.legend()
    plt.xlim(-0.5, len(data.node_location_mean) - 0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "graph_node_location_importance.pdf")
    plt.close()


def save_graph_importance(data: GraphImportance, output_dir: Path) -> None:
    np.save(output_dir / "graph_shap_node_mask_mean.npy", data.node_mask_mean)
    np.save(output_dir / "graph_shap_node_location_mean.npy", data.node_location_mean)
    np.save(output_dir / "graph_shap_edge_type_mean.npy", data.edge_type_mean)
    np.save(output_dir / "graph_shap_node_feature_mean.npy", data.node_feature_mean)
    with open(output_dir / "graph_shap_edge_scores_mean.json", "w") as f:
        json.dump(data.edge_scores, f)


def load_graph_importance(source_dir: Path) -> GraphImportance:
    node_mask_mean = np.load(source_dir / "graph_shap_node_mask_mean.npy")
    node_location_mean = np.load(source_dir / "graph_shap_node_location_mean.npy")
    edge_type_mean = np.load(source_dir / "graph_shap_edge_type_mean.npy")
    node_feature_mean = np.load(source_dir / "graph_shap_node_feature_mean.npy")
    with open(source_dir / "graph_shap_edge_scores_mean.json") as f:
        edge_scores = json.load(f)
    return GraphImportance(
        node_mask_mean=node_mask_mean,
        node_location_mean=node_location_mean,
        edge_type_mean=edge_type_mean,
        node_feature_mean=node_feature_mean,
        edge_scores=edge_scores,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--species", type=str, choices=["human", "mouse"], default="human"
    )
    parser.add_argument("--output_dir", type=str, default="figures/shap")
    parser.add_argument("--precomputed_dir", type=str, default=None)
    parser.add_argument("--use_precomputed", action="store_true")
    parser.add_argument("--subset_size", type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    data_dir = Path(args.data_dir) if args.data_dir else Path(f"data/te/{args.species}")
    out_dir = Path(args.output_dir) / args.species
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_precomputed:
        pre_dir = Path(args.precomputed_dir) if args.precomputed_dir else out_dir
        bio_data = load_bio_importance(pre_dir)
        graph_data = load_graph_importance(pre_dir)
        plot_bio_importance(bio_data, out_dir)
        plot_graph_importance(graph_data, out_dir)
        return

    assert args.checkpoint is not None
    checkpoint_dir = Path(args.checkpoint).parent
    config = load_config(checkpoint_dir)

    feature_path = data_dir / "feature.npy"
    target_path = data_dir / "target.npy"
    graph_path = data_dir / f"{data_dir.stem}_graph.pt"
    meta_path = data_dir / "meta.json"

    features = np.load(feature_path)
    targets = np.load(target_path)
    graphs = torch.load(graph_path, weights_only=False)
    feature_names = load_feature_names(meta_path, features.shape[1])

    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    bio_feat_dim = features.shape[1]
    num_targets = targets.shape[1]

    model = load_model_from_checkpoint(
        args.checkpoint, config, node_dim, edge_dim, bio_feat_dim, num_targets
    )
    model.to(device)

    mean_bio_emb, mean_graph_emb = precompute_embeddings(
        model, graphs, features, device
    )

    bio_data = compute_bio_importance(
        model,
        mean_graph_emb,
        features,
        args.subset_size,
        device,
        feature_names,
    )
    save_bio_importance(bio_data, out_dir)
    plot_bio_importance(bio_data, out_dir)

    graph_data = compute_graph_importance(
        model, mean_bio_emb, graphs, args.subset_size, device, node_dim
    )
    save_graph_importance(graph_data, out_dir)
    plot_graph_importance(graph_data, out_dir)


if __name__ == "__main__":
    main()

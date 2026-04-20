from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from .config import REPO_ROOT, load_config, require_mapping, resolve_repo_path
from .data import (
    PreparedDatasetBundle,
    create_dataloader,
    create_dataset,
    load_feature_manifest,
    load_prepared_bundle,
    load_split_manifest,
    move_batch_to_device,
)
from .inference import load_model_from_checkpoint, resolve_device
from .io import save_json

DEFAULT_HUMAN_CONFIG = REPO_ROOT / "configs/main/human.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/interpretability/human"

REGION_ORDER = ["5'UTR", "CDS", "3'UTR"]
BASE_NODE_FEATURE_LABELS = ["A%", "U%", "G%", "C%", "Ln(L)", "Pos", "Unp"]
REGION_NODE_FEATURE_LABELS = {
    "5'UTR": ["CpG", "uORF", "Kozak", "TOP", "G4", "uAUG"],
    "CDS": ["tAI", "CSC", "GC3", "Rare", "Ramp", "Basic"],
    "3'UTR": ["ARE", "Dest", "miRNA", "PolyA", "m6A", "AU"],
}
EDGE_TYPE_COLUMNS = {"Sequence": 0, "Structure": 1, "Long-range": 2}


@dataclass(frozen=True)
class FoldInterpretabilityContext:
    fold: int
    config: dict[str, Any]
    bundle: PreparedDatasetBundle
    target_names: list[str]
    selected_columns: list[str]
    train_indices: np.ndarray
    test_indices: np.ndarray
    dataloader: Any
    model: torch.nn.Module
    device: torch.device
    num_nodes: int
    feature_dim: int
    region_slices: dict[str, tuple[int, int]]
    node_reference_vectors: np.ndarray
    region_feature_medians: dict[str, np.ndarray]
    biochemical_medians: np.ndarray
    baseline_score: float


def default_output_dir() -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR


def _human_config() -> dict[str, Any]:
    return load_config(DEFAULT_HUMAN_CONFIG)


def _human_prepared_dir(config: dict[str, Any]) -> Path:
    path = resolve_repo_path(config["paths"]["prepared_dir"])
    if path is None:
        raise ValueError("Human config is missing paths.prepared_dir")
    return path


def _human_results_dir(config: dict[str, Any]) -> Path:
    path = resolve_repo_path(config["paths"]["results_dir"])
    if path is None:
        raise ValueError("Human config is missing paths.results_dir")
    return path


def _region_slices(config: dict[str, Any]) -> dict[str, tuple[int, int]]:
    graph_config = require_mapping(config, "graphs")
    utr5_bins = int(graph_config["utr5_bins"])
    cds_bins = int(graph_config["cds_bins"])
    utr3_bins = int(graph_config["utr3_bins"])
    return {
        "5'UTR": (0, utr5_bins),
        "CDS": (utr5_bins, utr5_bins + cds_bins),
        "3'UTR": (utr5_bins + cds_bins, utr5_bins + cds_bins + utr3_bins),
    }


def _node_feature_labels() -> dict[str, list[str]]:
    return {
        region_name: BASE_NODE_FEATURE_LABELS + REGION_NODE_FEATURE_LABELS[region_name]
        for region_name in REGION_ORDER
    }


def _region_for_node(node_index: int, region_slices: dict[str, tuple[int, int]]) -> str:
    for region_name in REGION_ORDER:
        start, end = region_slices[region_name]
        if start <= node_index < end:
            return region_name
    raise ValueError(
        f"Node index {node_index} does not fall into any configured region"
    )


def _biochemical_metadata(feature_name: str) -> tuple[str, str]:
    stripped = feature_name.removeprefix("biochemical_")
    if "." in stripped:
        category, label = stripped.split(".", 1)
    else:
        category, label = "Biochemical", stripped
    return category, label.replace("_", " ")


def _mean_te_pearson(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    if pred_array.shape != target_array.shape:
        raise ValueError(
            "Prediction shape "
            f"{pred_array.shape} does not match target shape {target_array.shape}"
        )
    mask = np.isfinite(target_array)
    counts = mask.sum(axis=1)
    valid_rows = counts > 0
    if int(valid_rows.sum()) < 2:
        return float("nan")
    valid_counts = counts[valid_rows].astype(np.float64)
    pred_mean = (pred_array[valid_rows] * mask[valid_rows]).sum(axis=1) / valid_counts
    target_mean = np.nansum(target_array[valid_rows], axis=1) / valid_counts
    if np.allclose(pred_mean.std(), 0.0) or np.allclose(target_mean.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(target_mean, pred_mean)[0, 1])


def _predict_score(
    context: FoldInterpretabilityContext,
    batch_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> float:
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    context.model.eval()
    with torch.no_grad():
        for batch in cast(Any, context.dataloader):
            resolved_batch = cast(
                dict[str, Any], move_batch_to_device(batch, context.device)
            )
            if batch_transform is not None:
                resolved_batch = batch_transform(resolved_batch)
            prediction = cast(Any, context.model(resolved_batch)).detach().cpu().numpy()
            targets = cast(Any, resolved_batch["target"]).detach().cpu().numpy()
            prediction_batches.append(prediction.astype(np.float32, copy=False))
            target_batches.append(targets.astype(np.float32, copy=False))
    predictions = np.concatenate(prediction_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return _mean_te_pearson(predictions, targets)


def _node_positions(batch: dict[str, Any], num_nodes: int) -> torch.Tensor:
    graph_batch = cast(Any, batch["graph"])
    return torch.arange(graph_batch.x.shape[0], device=graph_batch.x.device) % num_nodes


def _load_fold_contexts() -> list[FoldInterpretabilityContext]:
    config = _human_config()
    prepared_dir = _human_prepared_dir(config)
    results_dir = _human_results_dir(config)
    bundle = load_prepared_bundle(prepared_dir)
    training_config = require_mapping(config, "training")
    region_slices = _region_slices(config)
    device = resolve_device(str(training_config.get("accelerator", "auto")))

    contexts: list[FoldInterpretabilityContext] = []
    fold_count = int(require_mapping(config, "folds")["count"])
    for fold in range(fold_count):
        split_manifest = load_split_manifest(prepared_dir, fold)
        feature_manifest = load_feature_manifest(prepared_dir, fold)
        selected_columns = list(feature_manifest["selected_columns"])
        train_indices = np.asarray(split_manifest["train_indices"], dtype=np.int64)
        test_indices = np.asarray(split_manifest["test_indices"], dtype=np.int64)
        dataset = create_dataset(bundle, test_indices, selected_columns)
        dataloader = create_dataloader(
            dataset,
            batch_size=int(training_config["batch_size"]),
            num_workers=0,
            shuffle=False,
        )
        checkpoint_path = results_dir / f"fold_{fold:02d}" / "checkpoints" / "best.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing HybriTE checkpoint: {checkpoint_path}")

        first_graph = cast(Any, bundle.graphs[0])
        edge_attr = first_graph.edge_attr
        model = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_config=require_mapping(config, "model"),
            optimization_config=require_mapping(config, "optimization"),
            node_feature_dim=int(first_graph.x.shape[-1]),
            edge_feature_dim=int(edge_attr.shape[-1]) if edge_attr is not None else 0,
            num_targets=int(bundle.targets.shape[1]),
            biochemical_feature_dim=len(selected_columns),
        ).to(device)
        model.eval()

        node_stack = np.stack(
            [
                cast(Any, bundle.graphs[int(index)])
                .x.detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
                for index in train_indices
            ],
            axis=0,
        )
        node_reference_vectors = np.median(node_stack, axis=0).astype(
            np.float32, copy=False
        )
        region_feature_medians = {
            region_name: np.median(
                node_stack[:, start:end, :].reshape(-1, node_stack.shape[-1]),
                axis=0,
            ).astype(np.float32, copy=False)
            for region_name, (start, end) in region_slices.items()
        }

        biochemical_matrix = bundle.selected_biochemistry(selected_columns)
        if biochemical_matrix is None:
            raise ValueError("Human interpretability requires biochemical features")
        biochemical_medians = np.median(
            biochemical_matrix[train_indices], axis=0
        ).astype(np.float32, copy=False)

        context = FoldInterpretabilityContext(
            fold=fold,
            config=config,
            bundle=bundle,
            target_names=list(bundle.target_names),
            selected_columns=selected_columns,
            train_indices=train_indices,
            test_indices=test_indices,
            dataloader=dataloader,
            model=model,
            device=device,
            num_nodes=int(first_graph.num_nodes),
            feature_dim=int(first_graph.x.shape[-1]),
            region_slices=region_slices,
            node_reference_vectors=node_reference_vectors,
            region_feature_medians=region_feature_medians,
            biochemical_medians=biochemical_medians,
            baseline_score=float("nan"),
        )
        baseline_score = _predict_score(context)
        contexts.append(
            FoldInterpretabilityContext(
                fold=context.fold,
                config=context.config,
                bundle=context.bundle,
                target_names=context.target_names,
                selected_columns=context.selected_columns,
                train_indices=context.train_indices,
                test_indices=context.test_indices,
                dataloader=context.dataloader,
                model=context.model,
                device=context.device,
                num_nodes=context.num_nodes,
                feature_dim=context.feature_dim,
                region_slices=context.region_slices,
                node_reference_vectors=context.node_reference_vectors,
                region_feature_medians=context.region_feature_medians,
                biochemical_medians=context.biochemical_medians,
                baseline_score=baseline_score,
            )
        )
    return contexts


def _summarize_biochemical(by_fold: pd.DataFrame, total_folds: int) -> pd.DataFrame:
    summary = pd.DataFrame(
        by_fold.groupby(
            ["feature_name", "feature_label", "category"], as_index=False
        ).agg(
            selected_folds=("fold", "nunique"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
        )
    )
    summary["selection_rate"] = summary["selected_folds"] / float(total_folds)
    summary["importance"] = summary["mean_delta"] * summary["selection_rate"]
    return (
        pd.DataFrame(summary)
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )


def _summarize_mean_importance(
    by_fold: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    summary = pd.DataFrame(
        by_fold.groupby(group_columns, as_index=False).agg(
            importance=("delta", "mean"),
            median_delta=("delta", "median"),
        )
    )
    return (
        pd.DataFrame(summary)
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )


def _write_outputs(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    summary_name: str,
    by_fold_name: str,
) -> None:
    output_dir = default_output_dir()
    summary.to_csv(output_dir / summary_name, index=False)
    by_fold.to_csv(output_dir / by_fold_name, index=False)


def run_biochemical_importance_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_biochemical_importance_analysis(_load_fold_contexts())


def _run_biochemical_importance_analysis(
    contexts: list[FoldInterpretabilityContext],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for context in contexts:
        for feature_index, feature_name in enumerate(context.selected_columns):
            category, feature_label = _biochemical_metadata(feature_name)
            reference_value = float(context.biochemical_medians[feature_index])

            def transform(batch: dict[str, Any]) -> dict[str, Any]:
                biochemical = cast(Any, batch["biochemical_features"]).clone()
                biochemical[:, feature_index] = reference_value
                batch["biochemical_features"] = biochemical
                return batch

            perturbed_score = _predict_score(context, transform)
            rows.append(
                {
                    "fold": context.fold,
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "feature_label": feature_label,
                    "category": category,
                    "baseline_score": context.baseline_score,
                    "perturbed_score": perturbed_score,
                    "delta": context.baseline_score - perturbed_score,
                }
            )
    by_fold = pd.DataFrame(rows)
    summary = _summarize_biochemical(by_fold, total_folds=len(contexts))
    _write_outputs(
        summary,
        by_fold,
        "biochemical_feature_importance.csv",
        "biochemical_feature_importance_by_fold.csv",
    )
    _write_manifest()
    return summary, by_fold


def run_node_feature_importance_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_node_feature_importance_analysis(_load_fold_contexts())


def _run_node_feature_importance_analysis(
    contexts: list[FoldInterpretabilityContext],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_map = _node_feature_labels()
    rows: list[dict[str, object]] = []
    for context in contexts:
        if context.feature_dim != len(BASE_NODE_FEATURE_LABELS) + 6:
            raise ValueError(
                f"Unexpected node feature dimension {context.feature_dim}; expected 13"
            )
        for region_name in REGION_ORDER:
            region_start, region_end = context.region_slices[region_name]
            feature_labels = label_map[region_name]
            for feature_index, feature_label in enumerate(feature_labels):
                reference_value = float(
                    context.region_feature_medians[region_name][feature_index]
                )

                def transform(batch: dict[str, Any]) -> dict[str, Any]:
                    graph_batch = cast(Any, batch["graph"])
                    graph_batch.x = graph_batch.x.clone()
                    positions = _node_positions(batch, context.num_nodes)
                    mask = (positions >= region_start) & (positions < region_end)
                    graph_batch.x[mask, feature_index] = reference_value
                    return batch

                perturbed_score = _predict_score(context, transform)
                rows.append(
                    {
                        "fold": context.fold,
                        "region": region_name,
                        "feature_index": feature_index,
                        "feature_name": f"{region_name}_{feature_index:02d}",
                        "feature_label": feature_label,
                        "baseline_score": context.baseline_score,
                        "perturbed_score": perturbed_score,
                        "delta": context.baseline_score - perturbed_score,
                    }
                )
    by_fold = pd.DataFrame(rows)
    summary = _summarize_mean_importance(
        by_fold,
        ["region", "feature_index", "feature_name", "feature_label"],
    )
    _write_outputs(
        summary,
        by_fold,
        "node_feature_importance.csv",
        "node_feature_importance_by_fold.csv",
    )
    _write_manifest()
    return summary, by_fold


def run_regional_importance_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_regional_importance_analysis(_load_fold_contexts())


def _run_regional_importance_analysis(
    contexts: list[FoldInterpretabilityContext],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for context in contexts:
        for node_index in range(context.num_nodes):
            region_name = _region_for_node(node_index, context.region_slices)
            reference_vector = context.node_reference_vectors[node_index].copy()

            def transform(batch: dict[str, Any]) -> dict[str, Any]:
                graph_batch = cast(Any, batch["graph"])
                graph_batch.x = graph_batch.x.clone()
                positions = _node_positions(batch, context.num_nodes)
                mask = positions == node_index
                replacement = torch.as_tensor(
                    reference_vector,
                    dtype=graph_batch.x.dtype,
                    device=graph_batch.x.device,
                )
                graph_batch.x[mask, :] = replacement
                return batch

            perturbed_score = _predict_score(context, transform)
            rows.append(
                {
                    "fold": context.fold,
                    "node_index": node_index,
                    "normalized_position": node_index / max(1, context.num_nodes - 1),
                    "region": region_name,
                    "baseline_score": context.baseline_score,
                    "perturbed_score": perturbed_score,
                    "delta": context.baseline_score - perturbed_score,
                }
            )
    by_fold = pd.DataFrame(rows)
    summary = (
        _summarize_mean_importance(
            by_fold,
            ["node_index", "normalized_position", "region"],
        )
        .sort_values(by="node_index")
        .reset_index(drop=True)
    )
    summary = summary.rename(columns={"importance": "importance"})
    _write_outputs(
        summary,
        by_fold,
        "regional_importance.csv",
        "regional_importance_by_fold.csv",
    )
    _write_manifest()
    return summary, by_fold


def run_edge_importance_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_edge_importance_analysis(_load_fold_contexts())


def _run_edge_importance_analysis(
    contexts: list[FoldInterpretabilityContext],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for context in contexts:
        for edge_type, column_index in EDGE_TYPE_COLUMNS.items():

            def transform(batch: dict[str, Any]) -> dict[str, Any]:
                graph_batch = cast(Any, batch["graph"])
                if graph_batch.edge_attr.numel() == 0:
                    return batch
                keep_mask = graph_batch.edge_attr[:, column_index] < 0.5
                graph_batch.edge_index = graph_batch.edge_index[:, keep_mask].clone()
                graph_batch.edge_attr = graph_batch.edge_attr[keep_mask].clone()
                return batch

            perturbed_score = _predict_score(context, transform)
            rows.append(
                {
                    "fold": context.fold,
                    "edge_type": edge_type,
                    "baseline_score": context.baseline_score,
                    "perturbed_score": perturbed_score,
                    "delta": context.baseline_score - perturbed_score,
                }
            )
    by_fold = pd.DataFrame(rows)
    summary = _summarize_mean_importance(by_fold, ["edge_type"])
    _write_outputs(
        summary,
        by_fold,
        "edge_importance_summary.csv",
        "edge_importance_by_fold.csv",
    )
    _write_manifest()
    return summary, by_fold


def run_all_interpretability_analyses() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    contexts = _load_fold_contexts()
    return {
        "biochemical": _run_biochemical_importance_analysis(contexts),
        "node_features": _run_node_feature_importance_analysis(contexts),
        "regional": _run_regional_importance_analysis(contexts),
        "edge": _run_edge_importance_analysis(contexts),
    }


def _write_manifest() -> None:
    output_dir = default_output_dir()
    save_json(
        {
            "species": "human",
            "method": "deterministic_occlusion_sensitivity",
            "score": "delta_mean_te_pearson",
            "claim_guardrail": "predictive_only_not_mechanistic",
            "config_path": str(DEFAULT_HUMAN_CONFIG),
            "output_dir": str(output_dir),
        },
        output_dir / "interpretability_manifest.json",
    )

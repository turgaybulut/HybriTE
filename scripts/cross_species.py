from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _split_indices(prepared_dir: Path, fold: int, split_name: str) -> np.ndarray:
    from hybrite.data import load_prepared_bundle, load_split_manifest

    if split_name == "all":
        bundle = load_prepared_bundle(prepared_dir)
        return np.arange(len(bundle.ids), dtype=np.int64)
    split_manifest = load_split_manifest(prepared_dir, fold)
    key = f"{split_name}_indices"
    if key not in split_manifest:
        raise ValueError(f"Unsupported split: {split_name}")
    return np.asarray(split_manifest[key], dtype=np.int64)


def main() -> None:
    from hybrite.config import load_config, require_mapping, resolve_repo_path
    from hybrite.data import (
        create_dataloader,
        create_dataset,
        load_feature_manifest,
        load_prepared_bundle,
    )
    from hybrite.evaluation import evaluate_and_save
    from hybrite.inference import (
        load_model_from_checkpoint,
        run_inference,
    )
    from hybrite.io import save_json, write_matrix_csv

    args = parse_args()
    source_config = load_config(args.source_config)
    target_config = load_config(args.target_config)
    source_prepared_dir = resolve_repo_path(source_config["paths"]["prepared_dir"])
    target_prepared_dir = resolve_repo_path(target_config["paths"]["prepared_dir"])
    if source_prepared_dir is None or target_prepared_dir is None:
        raise ValueError("Both source and target configs need paths.prepared_dir")

    source_bundle = load_prepared_bundle(source_prepared_dir)
    target_bundle = load_prepared_bundle(target_prepared_dir)

    source_edge_attr = source_bundle.graphs[0].edge_attr
    target_edge_attr = target_bundle.graphs[0].edge_attr

    if int(cast(Any, source_bundle.graphs[0].x).shape[-1]) != int(
        cast(Any, target_bundle.graphs[0].x).shape[-1]
    ):
        raise ValueError("Source and target graph node features do not match")
    if (source_edge_attr is None) != (target_edge_attr is None):
        raise ValueError("Source and target graph edge features do not match")
    if (
        source_edge_attr is not None
        and target_edge_attr is not None
        and source_edge_attr.shape[-1] != target_edge_attr.shape[-1]
    ):
        raise ValueError("Source and target graph edge features do not match")

    selected_columns = None
    biochemical_feature_dim = None
    if source_config["model"]["use_biochemistry"]:
        if not target_config["model"]["use_biochemistry"]:
            raise ValueError(
                "Cross-species prediction requires target biochemistry "
                "for a bio-enabled checkpoint"
            )
        source_feature_manifest = load_feature_manifest(source_prepared_dir, args.fold)
        target_feature_manifest = load_feature_manifest(target_prepared_dir, args.fold)
        source_columns = list(source_feature_manifest["selected_columns"])
        target_columns = list(target_feature_manifest["selected_columns"])
        if source_columns != target_columns:
            raise ValueError(
                "Source and target feature manifests differ for the requested fold"
            )
        selected_columns = target_columns
        biochemical_feature_dim = len(selected_columns)

    training_config = require_mapping(source_config, "training")
    target_indices = _split_indices(target_prepared_dir, args.fold, args.split)
    target_dataset = create_dataset(
        target_bundle,
        target_indices,
        selected_columns,
    )
    target_dataloader = create_dataloader(
        target_dataset,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
    )

    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_config=source_config["model"],
        optimization_config=source_config["optimization"],
        node_feature_dim=int(cast(Any, source_bundle.graphs[0].x).shape[-1]),
        edge_feature_dim=(
            int(cast(Any, source_bundle.graphs[0].edge_attr).shape[-1])
            if source_bundle.graphs[0].edge_attr is not None
            else 0
        ),
        num_targets=int(source_bundle.targets.shape[1]),
        biochemical_feature_dim=biochemical_feature_dim,
    )
    predictions, targets, ids = run_inference(
        model,
        target_dataloader,
        training_config,
    )

    evaluation_target_names = target_bundle.target_names
    evaluation_predictions = predictions
    evaluation_targets = targets
    target_space_mismatch = predictions.shape[1] != targets.shape[1]
    if target_space_mismatch:
        evaluation_predictions = predictions.mean(axis=1, keepdims=True)
        evaluation_targets = np.nanmean(targets, axis=1, keepdims=True)
        evaluation_target_names = ["mean_te"]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(
            Path(args.checkpoint).resolve().parent.parent
            / "cross_species"
            / (
                f"{source_config['config_name']}_"
                f"to_{target_config['config_name']}_"
                f"fold_{args.fold:02d}_{args.split}"
            )
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if target_space_mismatch:
        write_matrix_csv(
            output_path / "source_space_predictions.csv",
            predictions,
            source_bundle.target_names,
            ids=ids,
            id_column=target_bundle.id_column,
        )
        write_matrix_csv(
            output_path / "target_space_targets.csv",
            targets,
            target_bundle.target_names,
            ids=ids,
            id_column=target_bundle.id_column,
        )

    save_json(
        {
            "source_config": source_config["config_name"],
            "target_config": target_config["config_name"],
            "fold": args.fold,
            "target_split": args.split,
            "n_target_samples": int(len(target_indices)),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "evaluation_mode": (
                "mean_te_only_due_to_target_space_mismatch"
                if target_space_mismatch
                else "full_multi_target"
            ),
        },
        output_path / "cross_species_manifest.json",
    )

    evaluate_and_save(
        output_dir=output_path,
        predictions=evaluation_predictions,
        targets=evaluation_targets,
        target_names=evaluation_target_names,
        ids=ids,
        id_column=target_bundle.id_column,
        bootstrap_samples=int(target_config["evaluation"]["bootstrap_samples"]),
        seed=int(target_config["seed"]),
    )


if __name__ == "__main__":
    main()

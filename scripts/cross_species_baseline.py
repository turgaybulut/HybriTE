from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-species evaluation for baseline tabular models."
    )
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _results_root(model_path: str | Path) -> Path | None:
    resolved = Path(model_path).expanduser().resolve()
    for parent in [resolved.parent, *resolved.parents]:
        if (parent / "cross_validation_summary.csv").exists():
            return parent
    return None


def _split_indices(fold_data: Any, split_name: str) -> np.ndarray:
    if split_name == "train":
        return np.asarray(fold_data.train_indices, dtype=np.int64)
    if split_name == "val":
        return np.asarray(fold_data.val_indices, dtype=np.int64)
    if split_name == "test":
        return np.asarray(fold_data.test_indices, dtype=np.int64)
    if split_name == "all":
        return np.arange(len(fold_data.bundle.ids), dtype=np.int64)
    raise ValueError(f"Unsupported split: {split_name}")


def _default_output_dir(
    *,
    model_path: str | Path,
    source_config_name: str,
    target_config_name: str,
    fold: int,
    split: str,
) -> Path:
    results_root = _results_root(model_path)
    parent = (
        results_root.parent
        if results_root is not None
        else Path(model_path).resolve().parent
    )
    fold_name = f"fold_{fold:02d}" if split == "test" else f"fold_{fold:02d}_{split}"
    return parent / f"{source_config_name}_to_{target_config_name}_transfer" / fold_name


def main() -> None:
    from hybrite.baselines import (
        _combined_features,
        _load_tabular_fold_data,
        _predict_lightgbm,
    )
    from hybrite.config import load_config, require_mapping
    from hybrite.evaluation import evaluate_and_save
    from hybrite.io import save_json, write_matrix_csv

    args = parse_args()
    source_config = load_config(args.source_config)
    target_config = load_config(args.target_config)
    source_baseline = require_mapping(source_config, "baseline")
    target_baseline = require_mapping(target_config, "baseline")

    if str(source_baseline.get("type")) != "lightgbm":
        raise ValueError("Source baseline config must use baseline.type=lightgbm")
    if str(target_baseline.get("type")) != "lightgbm":
        raise ValueError("Target baseline config must use baseline.type=lightgbm")

    source_fold_data = _load_tabular_fold_data(source_config, int(args.fold))
    target_fold_data = _load_tabular_fold_data(target_config, int(args.fold))

    source_columns = list(source_fold_data.biochemical_feature_names)
    target_columns = list(target_fold_data.biochemical_feature_names)
    if source_columns != target_columns:
        raise ValueError(
            "Source and target LightGBM feature columns differ for the requested fold"
        )

    model_path = Path(args.model_path).expanduser().resolve()
    with open(model_path, "rb") as handle:
        models = pickle.load(handle)

    target_indices = _split_indices(target_fold_data, str(args.split))
    target_features = _combined_features(target_fold_data, target_indices)
    predictions = _predict_lightgbm(cast(Any, models), target_features)
    targets = np.asarray(
        target_fold_data.bundle.targets[target_indices], dtype=np.float32
    )
    ids = target_fold_data.bundle.ids[target_indices]

    source_target_names = list(source_fold_data.bundle.target_names)
    evaluation_target_names = list(target_fold_data.bundle.target_names)
    evaluation_predictions = predictions
    evaluation_targets = targets
    target_space_mismatch = predictions.shape[1] != targets.shape[1]
    if target_space_mismatch:
        evaluation_predictions = predictions.mean(axis=1, keepdims=True)
        evaluation_targets = np.nanmean(targets, axis=1, keepdims=True)
        evaluation_target_names = ["mean_te"]

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(
            model_path=model_path,
            source_config_name=str(source_config["config_name"]),
            target_config_name=str(target_config["config_name"]),
            fold=int(args.fold),
            split=str(args.split),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_space_mismatch:
        write_matrix_csv(
            output_dir / "source_space_predictions.csv",
            predictions,
            source_target_names,
            ids=ids,
            id_column=target_fold_data.bundle.id_column,
        )
        write_matrix_csv(
            output_dir / "target_space_targets.csv",
            targets,
            list(target_fold_data.bundle.target_names),
            ids=ids,
            id_column=target_fold_data.bundle.id_column,
        )

    save_json(
        {
            "source_config": str(source_config["config_name"]),
            "target_config": str(target_config["config_name"]),
            "baseline_type": "lightgbm",
            "fold": int(args.fold),
            "target_split": str(args.split),
            "n_target_samples": int(len(target_indices)),
            "model_path": str(model_path),
            "biochemical_feature_count": len(source_columns),
            "evaluation_mode": (
                "mean_te_only_due_to_target_space_mismatch"
                if target_space_mismatch
                else "full_multi_target"
            ),
        },
        output_dir / "cross_species_manifest.json",
    )

    evaluate_and_save(
        output_dir=output_dir,
        predictions=evaluation_predictions,
        targets=evaluation_targets,
        target_names=evaluation_target_names,
        ids=ids,
        id_column=target_fold_data.bundle.id_column,
        bootstrap_samples=int(target_config["evaluation"]["bootstrap_samples"]),
        seed=int(target_config["seed"]),
    )


if __name__ == "__main__":
    main()

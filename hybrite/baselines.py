from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor  # pyright: ignore[reportMissingImports]

from .config import require_mapping, resolve_repo_path
from .data import (
    PreparedDatasetBundle,
    load_feature_manifest,
    load_prepared_bundle,
    load_split_manifest,
)
from .evaluation import evaluate_and_save
from .io import save_json
from .train import summarize_results


@dataclass(frozen=True)
class ConstantPredictor:
    value: float


@dataclass(frozen=True)
class TabularFoldData:
    bundle: PreparedDatasetBundle
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    biochemical_features: np.ndarray
    biochemical_feature_names: list[str]


def _selected_columns(config: dict[str, Any], fold: int) -> list[str]:
    baseline_config = require_mapping(config, "baseline")
    if str(baseline_config.get("type")) != "lightgbm":
        raise ValueError("Paper release supports only baseline.type=lightgbm")
    if not bool(baseline_config.get("use_biochemistry", True)):
        raise ValueError("LightGBM baseline requires biochemical features")
    prepared_dir = resolve_repo_path(config["paths"]["prepared_dir"])
    if prepared_dir is None:
        raise ValueError("paths.prepared_dir is required")
    feature_manifest = load_feature_manifest(prepared_dir, fold)
    return list(feature_manifest["selected_columns"])


def _load_tabular_fold_data(config: dict[str, Any], fold: int) -> TabularFoldData:
    baseline_config = require_mapping(config, "baseline")
    if str(baseline_config.get("type")) != "lightgbm":
        raise ValueError("Paper release supports only baseline.type=lightgbm")

    prepared_dir = resolve_repo_path(config["paths"]["prepared_dir"])
    if prepared_dir is None:
        raise ValueError("paths.prepared_dir is required")

    bundle = load_prepared_bundle(prepared_dir)
    split_manifest = load_split_manifest(prepared_dir, fold)
    biochemical_feature_names = _selected_columns(config, fold)
    biochemical_features = bundle.selected_biochemistry(biochemical_feature_names)
    if biochemical_features is None:
        raise ValueError("LightGBM baseline requires biochemical features")

    return TabularFoldData(
        bundle=bundle,
        train_indices=np.asarray(split_manifest["train_indices"], dtype=np.int64),
        val_indices=np.asarray(split_manifest["val_indices"], dtype=np.int64),
        test_indices=np.asarray(split_manifest["test_indices"], dtype=np.int64),
        biochemical_features=biochemical_features,
        biochemical_feature_names=biochemical_feature_names,
    )


def _combined_features(
    fold_data: TabularFoldData,
    indices: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        fold_data.biochemical_features[indices],
        columns=pd.Index(fold_data.biochemical_feature_names),
    )


def _write_feature_manifest(fold_dir: Path, fold_data: TabularFoldData) -> None:
    save_json(
        {
            "baseline_type": "lightgbm",
            "biochemical_feature_count": len(fold_data.biochemical_feature_names),
            "biochemical_feature_names": fold_data.biochemical_feature_names,
            "target_count": len(fold_data.bundle.target_names),
        },
        fold_dir / "feature_manifest.json",
    )


def _fit_lightgbm(
    config: dict[str, Any],
    fold_data: TabularFoldData,
) -> tuple[list[LGBMRegressor | ConstantPredictor], pd.DataFrame]:
    train_features = _combined_features(fold_data, fold_data.train_indices)
    train_targets = fold_data.bundle.targets[fold_data.train_indices]
    models: list[LGBMRegressor | ConstantPredictor] = []
    rows: list[dict[str, Any]] = []

    for target_index, target_name in enumerate(fold_data.bundle.target_names):
        valid_rows = np.isfinite(train_targets[:, target_index])
        target_values = train_targets[valid_rows, target_index]
        if len(target_values) == 0:
            predictor: LGBMRegressor | ConstantPredictor = ConstantPredictor(0.0)
            predictor_type = "constant"
        elif len(target_values) == 1:
            predictor = ConstantPredictor(float(target_values[0]))
            predictor_type = "constant"
        else:
            predictor = LGBMRegressor(
                importance_type="gain",
                random_state=int(config["seed"]),
                force_row_wise=True,
                verbosity=-1,
            )
            cast(Any, predictor).fit(train_features[valid_rows], target_values)
            predictor_type = "lightgbm"
        models.append(predictor)
        rows.append(
            {
                "target_name": target_name,
                "n_train": int(valid_rows.sum()),
                "predictor_type": predictor_type,
            }
        )
    return models, pd.DataFrame(rows)


def _predict_lightgbm(
    models: list[LGBMRegressor | ConstantPredictor],
    features: pd.DataFrame,
) -> np.ndarray:
    predictions = np.zeros((len(features), len(models)), dtype=np.float32)
    for target_index, predictor in enumerate(models):
        if isinstance(predictor, ConstantPredictor):
            predictions[:, target_index] = predictor.value
        else:
            predictions[:, target_index] = (
                cast(Any, predictor).predict(features).astype(np.float32, copy=False)
            )
    return predictions


def _evaluate_predictions(
    output_dir: Path,
    predictions: np.ndarray,
    targets: np.ndarray,
    ids: np.ndarray,
    fold_data: TabularFoldData,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    return evaluate_and_save(
        output_dir=output_dir,
        predictions=predictions,
        targets=targets,
        target_names=fold_data.bundle.target_names,
        ids=ids,
        id_column=fold_data.bundle.id_column,
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["seed"]),
    )


def _train_lightgbm_fold(config: dict[str, Any], fold: int) -> Path:
    results_dir = resolve_repo_path(config["paths"]["results_dir"])
    if results_dir is None:
        raise ValueError("paths.results_dir is required")

    fold_dir = results_dir / f"fold_{fold:02d}"
    model_dir = fold_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_data = _load_tabular_fold_data(config, fold)
    _write_feature_manifest(fold_dir, fold_data)

    models, training_table = _fit_lightgbm(config, fold_data)
    training_table.to_csv(fold_dir / "target_training_summary.csv", index=False)
    with open(model_dir / "model.pkl", "wb") as handle:
        pickle.dump(models, handle)

    val_features = _combined_features(fold_data, fold_data.val_indices)
    val_predictions = _predict_lightgbm(models, val_features)
    val_targets = fold_data.bundle.targets[fold_data.val_indices]
    val_summary = _evaluate_predictions(
        fold_dir / "val",
        val_predictions,
        val_targets,
        fold_data.bundle.ids[fold_data.val_indices],
        fold_data,
        config,
    )

    test_features = _combined_features(fold_data, fold_data.test_indices)
    test_predictions = _predict_lightgbm(models, test_features)
    test_targets = fold_data.bundle.targets[fold_data.test_indices]
    test_summary = _evaluate_predictions(
        fold_dir / "test",
        test_predictions,
        test_targets,
        fold_data.bundle.ids[fold_data.test_indices],
        fold_data,
        config,
    )

    save_json(
        {
            "fold": fold,
            "baseline_type": "lightgbm",
            "model_path": str(model_dir / "model.pkl"),
            "val_summary": val_summary,
            "test_summary": test_summary,
        },
        fold_dir / "fold_summary.json",
    )
    return fold_dir


def train_baseline_fold(config: dict[str, Any], fold: int) -> Path:
    baseline_config = require_mapping(config, "baseline")
    if str(baseline_config.get("type")) != "lightgbm":
        raise ValueError("Paper release supports only baseline.type=lightgbm")
    return _train_lightgbm_fold(config, fold)


def summarize_baseline_results(config: dict[str, Any]) -> None:
    summarize_results(config)

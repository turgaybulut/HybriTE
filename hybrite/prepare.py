from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import f_regression
from sklearn.model_selection import KFold, train_test_split

from .config import require_mapping, resolve_repo_path
from .io import load_json, save_json


def _select_columns(columns: list[str], patterns: list[str]) -> list[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    return [
        column
        for column in columns
        if any(pattern.search(column) for pattern in compiled)
    ]


def _numeric_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    fill_value: float | None = None,
) -> np.ndarray:
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if fill_value is not None:
        numeric = numeric.fillna(fill_value)
    return numeric.to_numpy(dtype=np.float32)


def _validation_size(train_pool_size: int, validation_fraction: float) -> int:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("folds.validation_fraction must be between 0 and 1")
    validation_size = max(1, int(round(train_pool_size * validation_fraction)))
    if validation_size >= train_pool_size:
        validation_size = train_pool_size - 1
    if validation_size <= 0:
        raise ValueError(
            "Training fold is too small for the requested validation split"
        )
    return validation_size


def _feature_score_table(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    per_target_scores: list[np.ndarray] = []
    targets_used = 0

    for column_index in range(targets.shape[1]):
        valid_rows = np.isfinite(targets[:, column_index])
        if int(valid_rows.sum()) < 3:
            continue
        scores, _ = f_regression(
            features[valid_rows], targets[valid_rows, column_index]
        )
        per_target_scores.append(np.asarray(scores, dtype=np.float64))
        targets_used += 1

    if not per_target_scores:
        raise ValueError(
            "No target columns had enough observed values for feature selection"
        )

    stacked_scores = np.vstack(per_target_scores)
    aggregated_scores = np.nanmean(stacked_scores, axis=0)
    return pd.DataFrame(
        {
            "feature_name": feature_names,
            "aggregated_score": aggregated_scores,
            "targets_used": targets_used,
        }
    ).sort_values("aggregated_score", ascending=False, kind="stable")


def _prepare_feature_manifest(
    prepared_dir: Path,
    fold: int,
    config: dict[str, Any],
    train_indices: np.ndarray,
    target_array: np.ndarray,
    biochemical_array: np.ndarray,
    biochemical_columns: list[str],
) -> None:
    feature_config = require_mapping(config, "features")
    fold_dir = prepared_dir / "folds" / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    selection_mode = str(feature_config["selection_mode"])
    feature_scores_path = fold_dir / "feature_scores.csv"
    source_manifest_path: Path | None = None

    if selection_mode == "train_fold":
        score_table = _feature_score_table(
            biochemical_array[train_indices],
            target_array[train_indices],
            biochemical_columns,
        )
        score_table.to_csv(feature_scores_path, index=False)
        select_k = int(feature_config["select_k"])
        selected_columns = score_table["feature_name"].head(select_k).tolist()
        orthology_manifest = None
    elif selection_mode == "transfer":
        source_prepared_dir = resolve_repo_path(feature_config["source_prepared_dir"])
        if source_prepared_dir is None:
            raise ValueError(
                "features.source_prepared_dir is required for transfer mode"
            )
        source_fold_dir = source_prepared_dir / "folds" / f"fold_{fold:02d}"
        source_manifest_path = source_fold_dir / "feature_manifest.json"
        source_scores_path = source_fold_dir / "feature_scores.csv"
        selected_columns = list(load_json(source_manifest_path)["selected_columns"])
        if source_scores_path.exists():
            shutil.copy2(source_scores_path, feature_scores_path)
        else:
            pd.DataFrame({"feature_name": selected_columns}).to_csv(
                feature_scores_path,
                index=False,
            )
        orthology_manifest_path = resolve_repo_path(
            feature_config.get("orthology_manifest")
        )
        orthology_manifest = (
            str(orthology_manifest_path)
            if orthology_manifest_path is not None
            else None
        )
    else:
        raise ValueError(f"Unsupported features.selection_mode: {selection_mode}")

    missing_columns = [
        name for name in selected_columns if name not in biochemical_columns
    ]
    if missing_columns:
        raise ValueError(f"Selected biochemical columns are missing: {missing_columns}")

    selected_indices = [biochemical_columns.index(name) for name in selected_columns]
    save_json(
        {
            "fold": fold,
            "selection_mode": selection_mode,
            "scoring": "mean_f_regression_over_observed_targets",
            "selected_columns": selected_columns,
            "selected_indices": selected_indices,
            "feature_scores_path": str(feature_scores_path),
            "source_feature_manifest": (
                str(source_manifest_path) if source_manifest_path is not None else None
            ),
            "orthology_manifest": orthology_manifest,
        },
        fold_dir / "feature_manifest.json",
    )


def _prepare_split_manifest(
    prepared_dir: Path,
    fold: int,
    seed: int,
    ids: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    fold_dir = prepared_dir / "folds" / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "fold": fold,
            "seed": seed,
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "train_ids": ids[train_indices].astype(str).tolist(),
            "val_ids": ids[val_indices].astype(str).tolist(),
            "test_ids": ids[test_indices].astype(str).tolist(),
            "group_aware": False,
        },
        fold_dir / "split_manifest.json",
    )


def prepare_dataset(config: dict[str, Any]) -> Path:
    dataset_config = require_mapping(config, "dataset")
    folds_config = require_mapping(config, "folds")
    feature_config = require_mapping(config, "features")
    paths_config = require_mapping(config, "paths")

    table_path = resolve_repo_path(dataset_config["table_path"])
    graph_path = resolve_repo_path(dataset_config["graph_path"])
    prepared_dir = resolve_repo_path(paths_config["prepared_dir"])

    if table_path is None or graph_path is None or prepared_dir is None:
        raise ValueError("Config is missing required dataset or path settings")

    dataframe = pd.read_csv(table_path)
    graphs = torch.load(graph_path, map_location="cpu", weights_only=False)
    if len(dataframe) != len(graphs):
        raise ValueError("Graph count does not match the input table length")

    id_column = str(dataset_config["id_column"])
    ids = np.asarray(dataframe[id_column].astype(str).tolist(), dtype=np.str_)
    target_columns = _select_columns(
        dataframe.columns.tolist(),
        list(dataset_config["target_patterns"]),
    )
    biochemical_columns = _select_columns(
        dataframe.columns.tolist(),
        list(dataset_config["biochemical_patterns"]),
    )

    if not target_columns:
        raise ValueError("No target columns matched dataset.target_patterns")
    if not biochemical_columns:
        raise ValueError("No biochemical columns matched dataset.biochemical_patterns")

    target_array = _numeric_matrix(dataframe, target_columns)
    biochemical_array = _numeric_matrix(
        dataframe,
        biochemical_columns,
        fill_value=0.0,
    )

    prepared_dir.mkdir(parents=True, exist_ok=True)
    np.save(prepared_dir / "ids.npy", ids)
    np.save(prepared_dir / "targets.npy", target_array)
    np.save(prepared_dir / "biochemistry.npy", biochemical_array)

    save_json(
        {
            "config_name": config["config_name"],
            "source_table_path": str(table_path),
            "graph_path": str(graph_path),
            "id_column": id_column,
            "ids_npy": "ids.npy",
            "targets_npy": "targets.npy",
            "biochemistry_npy": "biochemistry.npy",
            "target_columns": target_columns,
            "biochemical_columns": biochemical_columns,
            "sample_count": int(len(ids)),
            "fold_count": int(folds_config["count"]),
            "validation_fraction": float(folds_config["validation_fraction"]),
            "splitter": "kfold",
            "feature_selection_mode": str(feature_config["selection_mode"]),
            "biochemistry_missing_value_fill": 0.0,
        },
        prepared_dir / "dataset_manifest.json",
    )

    fold_count = int(folds_config["count"])
    if fold_count < 2:
        raise ValueError("folds.count must be at least 2")
    if fold_count > len(ids):
        raise ValueError("folds.count cannot exceed the prepared dataset size")

    splitter = KFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=int(config["seed"]),
    )

    for fold, (train_pool_indices, test_indices) in enumerate(splitter.split(ids)):
        validation_size = _validation_size(
            len(train_pool_indices),
            float(folds_config["validation_fraction"]),
        )
        train_indices, val_indices = train_test_split(
            train_pool_indices,
            test_size=validation_size,
            random_state=int(config["seed"]) + fold,
            shuffle=True,
        )
        train_indices = np.sort(np.asarray(train_indices, dtype=np.int64))
        val_indices = np.sort(np.asarray(val_indices, dtype=np.int64))
        test_indices = np.sort(test_indices.astype(np.int64))

        _prepare_split_manifest(
            prepared_dir,
            fold,
            int(config["seed"]) + fold,
            ids,
            train_indices,
            val_indices,
            test_indices,
        )
        _prepare_feature_manifest(
            prepared_dir,
            fold,
            config,
            train_indices,
            target_array,
            biochemical_array,
            biochemical_columns,
        )

    return prepared_dir

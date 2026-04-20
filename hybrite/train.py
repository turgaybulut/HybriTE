from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .config import require_mapping, resolve_repo_path
from .data import (
    HybriTEDataModule,
    load_feature_manifest,
    load_prepared_bundle,
    load_split_manifest,
)
from .evaluation import evaluate_and_save
from .inference import load_model_from_checkpoint, run_inference
from .io import save_json
from .lightning_utils import create_training_trainer, is_global_zero_process
from .model import HybriTELightningModule


def _selected_columns(config: dict[str, Any], fold: int) -> list[str] | None:
    if not config["model"]["use_biochemistry"]:
        return None
    prepared_dir = resolve_repo_path(config["paths"]["prepared_dir"])
    if prepared_dir is None:
        raise ValueError("paths.prepared_dir is required")
    feature_manifest = load_feature_manifest(prepared_dir, fold)
    return list(feature_manifest["selected_columns"])


def _build_datamodule(config: dict[str, Any], fold: int) -> HybriTEDataModule:
    prepared_dir = resolve_repo_path(config["paths"]["prepared_dir"])
    if prepared_dir is None:
        raise ValueError("paths.prepared_dir is required")
    bundle = load_prepared_bundle(prepared_dir)
    split_manifest = load_split_manifest(prepared_dir, fold)
    selected_columns = _selected_columns(config, fold)
    training_config = require_mapping(config, "training")
    datamodule = HybriTEDataModule(
        bundle=bundle,
        train_indices=split_manifest["train_indices"],
        val_indices=split_manifest["val_indices"],
        test_indices=split_manifest["test_indices"],
        selected_columns=selected_columns,
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config["num_workers"]),
    )
    return datamodule


def _evaluate_split(
    config: dict[str, Any],
    datamodule: HybriTEDataModule,
    checkpoint_path: Path,
    split_name: str,
    output_dir: Path,
) -> dict[str, Any] | None:
    training_config = require_mapping(config, "training")
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_config=config["model"],
        optimization_config=config["optimization"],
        node_feature_dim=datamodule.node_feature_dim,
        edge_feature_dim=datamodule.edge_feature_dim,
        num_targets=datamodule.num_targets,
        biochemical_feature_dim=datamodule.biochemical_feature_dim,
    )

    if split_name == "val":
        dataloader = datamodule.val_dataloader()
    elif split_name == "test":
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    predictions, targets, ids = run_inference(model, dataloader, training_config)
    return evaluate_and_save(
        output_dir=output_dir,
        predictions=predictions,
        targets=targets,
        target_names=datamodule.bundle.target_names,
        ids=ids,
        id_column=datamodule.bundle.id_column,
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["seed"]),
    )


def train_fold(config: dict[str, Any], fold: int) -> Path:
    training_config = require_mapping(config, "training")
    results_dir = resolve_repo_path(config["paths"]["results_dir"])
    if results_dir is None:
        raise ValueError("paths.results_dir is required")
    fold_dir = results_dir / f"fold_{fold:02d}"
    checkpoint_dir = fold_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    datamodule = _build_datamodule(config, fold)
    model = HybriTELightningModule(
        model_config=config["model"],
        optimization_config=config["optimization"],
        node_feature_dim=datamodule.node_feature_dim,
        edge_feature_dim=datamodule.edge_feature_dim,
        num_targets=datamodule.num_targets,
        biochemical_feature_dim=datamodule.biochemical_feature_dim,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(training_config["patience"]),
    )
    logger = CSVLogger(save_dir=str(fold_dir / "logs"), name="")

    trainer = create_training_trainer(
        training_config,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        default_root_dir=str(fold_dir),
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.strategy.barrier("fit_complete")

    if trainer.is_global_zero:
        best_checkpoint_path = Path(checkpoint_callback.best_model_path)
        if not best_checkpoint_path.exists():
            raise RuntimeError(f"Best checkpoint was not created for fold {fold}")

        val_summary = _evaluate_split(
            config,
            datamodule,
            best_checkpoint_path,
            "val",
            fold_dir / "val",
        )
        test_summary = _evaluate_split(
            config,
            datamodule,
            best_checkpoint_path,
            "test",
            fold_dir / "test",
        )
        save_json(
            {
                "fold": fold,
                "best_checkpoint": str(best_checkpoint_path),
                "val_summary": val_summary,
                "test_summary": test_summary,
            },
            fold_dir / "fold_summary.json",
        )
    trainer.strategy.barrier("post_fit_evaluation")
    return fold_dir


def summarize_results(config: dict[str, Any]) -> None:
    if not is_global_zero_process():
        return
    results_dir = resolve_repo_path(config["paths"]["results_dir"])
    fold_count = int(config["folds"]["count"])
    if results_dir is None:
        raise ValueError("paths.results_dir is required")

    rows: list[dict[str, Any]] = []
    for fold in range(fold_count):
        aggregate_path = (
            results_dir / f"fold_{fold:02d}" / "test" / "aggregate_metrics.csv"
        )
        if not aggregate_path.exists():
            continue
        aggregate_frame = pd.read_csv(aggregate_path)
        aggregate_frame.insert(0, "fold", fold)
        for record in aggregate_frame.to_dict(orient="records"):
            rows.append({str(key): value for key, value in record.items()})

    if not rows:
        return

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(results_dir / "cross_validation_metrics.csv", index=False)

    summary_frame = (
        metrics_frame.groupby(["scope", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_frame.to_csv(results_dir / "cross_validation_summary.csv", index=False)
    save_json(
        {
            "metrics": summary_frame.to_dict(orient="records"),
        },
        results_dir / "cross_validation_summary.json",
    )

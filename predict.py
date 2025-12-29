from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as tmf
from pytorch_lightning import Trainer
from sklearn.model_selection import KFold

from framework.config import load_config
from framework.registry import DATAMODULES, MODELS
from framework.utils import seed_everything


def _import_plugin(plugin_name: str) -> None:
    importlib.import_module(f"plugins.{plugin_name}.datamodule")
    importlib.import_module(f"plugins.{plugin_name}.model")


def _extract_fold_from_checkpoint(checkpoint_path: Path) -> int | None:
    match = re.search(r"fold_(\d+)", checkpoint_path.parent.parent.name)
    if match:
        return int(match.group(1))
    return None


def _load_model_from_checkpoint(
    checkpoint_path: str,
    model_class: type,
    config: dict[str, Any],
    model_kwargs: dict[str, Any],
) -> Any:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_class(config, **model_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _compute_aggregated_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    y_pred_valid = torch.nanmean(predictions, dim=1)
    y_true_valid = torch.nanmean(targets, dim=1)
    return {
        "mse": tmf.mean_squared_error(y_pred_valid, y_true_valid).item(),
        "r2": tmf.r2_score(y_pred_valid, y_true_valid).item(),
        "pearson": tmf.pearson_corrcoef(y_pred_valid, y_true_valid).item(),
        "spearman": tmf.spearman_corrcoef(y_pred_valid, y_true_valid).item(),
    }


def _compute_target_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    num_targets = predictions.shape[1]
    metrics = {
        "mse": torch.full((num_targets,), float("nan")),
        "r2": torch.full((num_targets,), float("nan")),
        "pearson": torch.full((num_targets,), float("nan")),
        "spearman": torch.full((num_targets,), float("nan")),
    }
    for i in range(num_targets):
        y_pred = predictions[:, i]
        y_true = targets[:, i]
        mask = ~torch.isnan(y_true)
        if not mask.any():
            continue
        y_pred_valid = y_pred[mask].unsqueeze(-1)
        y_true_valid = y_true[mask].unsqueeze(-1)
        metrics["mse"][i] = tmf.mean_squared_error(y_pred_valid, y_true_valid)
        metrics["r2"][i] = tmf.r2_score(y_pred_valid, y_true_valid)
        metrics["pearson"][i] = tmf.pearson_corrcoef(y_pred_valid, y_true_valid)
        metrics["spearman"][i] = tmf.spearman_corrcoef(y_pred_valid, y_true_valid)
    return metrics


def predict(
    checkpoint_path: str,
    config_path: str,
    output_dir: str | None = None,
    fold: int | None = None,
) -> None:
    checkpoint_path_obj = Path(checkpoint_path)
    assert checkpoint_path_obj.exists(), f"Checkpoint not found: {checkpoint_path}"

    config = load_config(config_path)
    seed_everything(config["seed"], workers=True)

    plugin_name = config["plugin"]
    _import_plugin(plugin_name)

    datamodule_name = f"{plugin_name}_datamodule"

    if fold is None:
        fold = _extract_fold_from_checkpoint(checkpoint_path_obj)

    if fold is not None:
        datamodule_full = DATAMODULES.get(datamodule_name)(config)
        datamodule_full.setup(stage="fit")
        dataset_size = (
            len(datamodule_full.train_dataset)
            + len(datamodule_full.val_dataset)
            + len(datamodule_full.test_dataset)
        )
        indices = np.arange(dataset_size)

        n_folds = config.get("n_folds", 10)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config["seed"])

        fold_splits = list(kf.split(indices))
        assert fold < len(fold_splits), (
            f"Fold {fold + 1} out of range (n_folds={n_folds})"
        )

        train_idx, val_idx = fold_splits[fold]
        datamodule = DATAMODULES.get(datamodule_name)(
            config, train_indices=train_idx, val_indices=val_idx
        )
        datamodule.setup(stage="fit")

        print(f"Using K-fold cross-validation: fold {fold + 1}/{n_folds}")
    else:
        datamodule = DATAMODULES.get(datamodule_name)(config)
        datamodule.setup(stage="fit")
        print("Using standard train/val/test split")

    model_kwargs = datamodule.model_kwargs
    num_targets = model_kwargs["num_targets"]

    model_name = f"{plugin_name}_model"
    model_class = MODELS.get(model_name)
    model = _load_model_from_checkpoint(
        checkpoint_path, model_class, config, model_kwargs
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    test_dataloader = datamodule.test_dataloader()

    predictions_list = trainer.predict(model, test_dataloader)
    predictions = torch.cat(predictions_list, dim=0).cpu()

    targets_list = []
    for batch in test_dataloader:
        if isinstance(batch, dict):
            batch_targets = batch["target"]
        else:
            batch_targets = batch[1]
        targets_list.append(batch_targets)

    test_targets = torch.cat(targets_list, dim=0).cpu()

    if test_targets.ndim == 1:
        test_targets = test_targets.unsqueeze(-1)

    if output_dir is None:
        output_dir = checkpoint_path_obj.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(predictions.numpy())
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Predictions saved to: {output_dir / 'predictions.csv'}")

    ground_truth_df = pd.DataFrame(test_targets.numpy())
    ground_truth_df.to_csv(output_dir / "ground_truth.csv", index=False)
    print(f"Ground truth saved to: {output_dir / 'ground_truth.csv'}")

    computed_metrics_aggregated = _compute_aggregated_metrics(predictions, test_targets)
    metrics_dict = {"metric": [], "value": []}
    for metric_name, metric_value in computed_metrics_aggregated.items():
        metrics_dict["metric"].append(metric_name)
        value = (
            metric_value.item()
            if isinstance(metric_value, torch.Tensor)
            else metric_value
        )
        metrics_dict["value"].append(value)
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\nMetrics saved to: {output_dir / 'metrics.csv'}")

    print("\nAggregated metrics (averaged predictions/targets):")
    for metric_name, metric_value in computed_metrics_aggregated.items():
        value = (
            metric_value.item()
            if isinstance(metric_value, torch.Tensor)
            else metric_value
        )
        print(f"  {metric_name}: {value:.6f}")

    if num_targets > 1:
        detailed_metrics = _compute_target_metrics(predictions, test_targets)
        metrics_rows = []
        for target_idx in range(num_targets):
            row = {"target_index": target_idx}
            for metric_name, metric_values in detailed_metrics.items():
                row[metric_name] = metric_values[target_idx].item()
            metrics_rows.append(row)
            metrics_df_per_target = pd.DataFrame(metrics_rows)
        detailed_path = output_dir / "target_metrics.csv"
        metrics_df_per_target.to_csv(detailed_path, index=False)
        print(f"\nPer-target metrics saved to: {detailed_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Fold number for K-fold CV"
            " (auto-detected from checkpoint filename if not provided)."
        ),
    )
    args = parser.parse_args()

    predict(args.checkpoint, args.config, args.output_dir, args.fold)

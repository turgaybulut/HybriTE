from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as tmf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from framework.config import load_config
from framework.registry import DATAMODULES, MODELS
from framework.utils import seed_everything


class TensorDataset(Dataset[Any]):
    def __init__(
        self, inputs: torch.Tensor, targets: torch.Tensor | None = None
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        if self.targets is not None:
            assert self.inputs.shape[0] == self.targets.shape[0]

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"tokens": self.inputs[index]}
        if self.targets is not None:
            item["target"] = self.targets[index]
        return item


class GraphDataset(Dataset[Any]):
    def __init__(
        self,
        graphs: list[Any],
        targets: torch.Tensor | None = None,
        biochemical_features: torch.Tensor | None = None,
    ) -> None:
        self.graphs = graphs
        self.targets = targets
        self.biochemical_features = biochemical_features
        if self.targets is not None:
            assert len(self.graphs) == self.targets.shape[0]
        if self.biochemical_features is not None:
            assert len(self.graphs) == self.biochemical_features.shape[0]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {"graph": self.graphs[index]}
        if self.targets is not None:
            item["target"] = self.targets[index]
        if self.biochemical_features is not None:
            item["biochemical_features"] = self.biochemical_features[index]
        return item


def _import_plugin(plugin_name: str) -> None:
    importlib.import_module(f"plugins.{plugin_name}.datamodule")
    importlib.import_module(f"plugins.{plugin_name}.model")


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


def _to_ncl(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[1] > tensor.shape[2]:
        tensor = tensor.permute(0, 2, 1)
    return tensor


def _graph_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    from torch_geometric.data import Batch

    result = {"graph": Batch.from_data_list([it["graph"] for it in items])}
    if "target" in items[0]:
        result["target"] = torch.stack([it["target"] for it in items])
    if "biochemical_features" in items[0]:
        result["biochemical_features"] = torch.stack(
            [it["biochemical_features"] for it in items]
        )
    return result


def _compute_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, per_target: bool
) -> dict[str, torch.Tensor | float]:
    if not per_target:
        y_pred_valid = torch.nanmean(predictions, axis=1)
        y_true_valid = torch.nanmean(targets, axis=1)

        return {
            "mse": tmf.mean_squared_error(y_pred_valid, y_true_valid).item(),
            "r2": tmf.r2_score(y_pred_valid, y_true_valid).item(),
            "pearson": tmf.pearson_corrcoef(y_pred_valid, y_true_valid).item(),
            "spearman": tmf.spearman_corrcoef(y_pred_valid, y_true_valid).item(),
        }

    num_targets = predictions.shape[1]

    metrics = {
        "mse": torch.zeros(num_targets),
        "r2": torch.zeros(num_targets),
        "pearson": torch.zeros(num_targets),
        "spearman": torch.zeros(num_targets),
    }

    for i in range(num_targets):
        y_pred = predictions[:, i]
        y_true = targets[:, i]
        mask = ~torch.isnan(y_true)
        if not mask.any():
            metrics["mse"][i] = float("nan")
            metrics["r2"][i] = float("nan")
            metrics["pearson"][i] = float("nan")
            metrics["spearman"][i] = float("nan")
            continue
        y_pred_valid = y_pred[mask].unsqueeze(-1)
        y_true_valid = y_true[mask].unsqueeze(-1)

        metrics["mse"][i] = tmf.mean_squared_error(y_pred_valid, y_true_valid)
        metrics["r2"][i] = tmf.r2_score(y_pred_valid, y_true_valid)
        metrics["pearson"][i] = tmf.pearson_corrcoef(y_pred_valid, y_true_valid)
        metrics["spearman"][i] = tmf.spearman_corrcoef(y_pred_valid, y_true_valid)

    return metrics


def predict_cross_species(
    checkpoint_path: str,
    train_config_path: str,
    test_inputs_path: str,
    test_targets_path: str | None = None,
    test_biochemical_features_path: str | None = None,
    output_dir: str | None = None,
    suffix: str = "cross_species",
) -> None:
    checkpoint_path_obj = Path(checkpoint_path)
    assert checkpoint_path_obj.exists(), f"Checkpoint not found: {checkpoint_path}"

    test_inputs_path_obj = Path(test_inputs_path)
    assert test_inputs_path_obj.exists(), f"Test inputs not found: {test_inputs_path}"

    train_config = load_config(train_config_path)
    seed_everything(train_config["seed"], workers=True)

    train_plugin_name = train_config["plugin"]
    _import_plugin(train_plugin_name)

    train_datamodule_name = f"{train_plugin_name}_datamodule"
    train_datamodule_temp = DATAMODULES.get(train_datamodule_name)(train_config)
    train_datamodule_temp.setup(stage="fit")
    model_kwargs = train_datamodule_temp.model_kwargs

    model_name = f"{train_plugin_name}_model"
    model_class = MODELS.get(model_name)
    model = _load_model_from_checkpoint(
        checkpoint_path, model_class, train_config, model_kwargs
    )

    is_graph_based = test_inputs_path_obj.suffix == ".pt"

    test_targets = None
    if test_targets_path is not None:
        test_targets_path_obj = Path(test_targets_path)
        assert test_targets_path_obj.exists(), (
            f"Test targets not found: {test_targets_path}"
        )
        test_targets_array = np.load(test_targets_path, allow_pickle=False)
        if test_targets_array.ndim == 1:
            test_targets_array = np.expand_dims(test_targets_array, axis=-1)
        test_targets = torch.from_numpy(test_targets_array).float()

    if is_graph_based:
        graphs = torch.load(test_inputs_path, map_location="cpu", weights_only=False)
        biochemical_features = None
        if (
            test_biochemical_features_path is not None
            and train_config["data"]["biochemical_features_npy"] is not None
        ):
            biochemical_features_path_obj = Path(test_biochemical_features_path)
            assert biochemical_features_path_obj.exists(), (
                f"Biochemical features not found: {test_biochemical_features_path}"
            )
            biochemical_features = torch.from_numpy(
                np.load(biochemical_features_path_obj, allow_pickle=False)
            ).float()

        test_dataset = GraphDataset(graphs, test_targets, biochemical_features)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config["data"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_graph_collate,
        )
    else:
        test_inputs_array = np.load(test_inputs_path, allow_pickle=False)
        test_inputs = torch.from_numpy(test_inputs_array).float()
        test_inputs = _to_ncl(test_inputs)

        test_dataset = TensorDataset(test_inputs, test_targets)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config["data"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    predictions_list = trainer.predict(model, test_dataloader)
    predictions = torch.cat(predictions_list, dim=0).cpu()

    if output_dir is None:
        output_dir = checkpoint_path_obj.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(predictions.numpy())
    predictions_df.to_csv(output_dir / f"predictions_{suffix}.csv", index=False)
    print(f"\nPredictions saved to: {output_dir / f'predictions_{suffix}.csv'}")

    if test_targets is not None:
        ground_truth_df = pd.DataFrame(test_targets.numpy())
        ground_truth_df.to_csv(output_dir / f"ground_truth_{suffix}.csv", index=False)
        print(f"Ground truth saved to: {output_dir / f'ground_truth_{suffix}.csv'}")

        computed_metrics_aggregated = _compute_metrics(predictions, test_targets, False)
        metrics_dict = {"metric": [], "value": []}
        for metric_name, metric_value in computed_metrics_aggregated.items():
            metrics_dict["metric"].append(metric_name)
            metrics_dict["value"].append(metric_value)
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.to_csv(output_dir / f"metrics_{suffix}.csv", index=False)
        print(
            f"\nCross-species metrics saved to: {output_dir / f'metrics_{suffix}.csv'}"
        )
        print("\nCross-species mean TE metrics:")
        for metric_name, metric_value in computed_metrics_aggregated.items():
            print(f"  {metric_name}: {metric_value:.6f}")
    else:
        print("\nNo targets provided - inference only mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (trained model)",
    )
    parser.add_argument(
        "--train_config", type=str, required=True, help="Path to training config file"
    )
    parser.add_argument(
        "--test_inputs",
        type=str,
        required=True,
        help="Path to test inputs (.npy for tensor data, .pt for graph data)",
    )
    parser.add_argument(
        "--test_targets",
        type=str,
        default=None,
        help="Path to test targets .npy file (optional)",
    )
    parser.add_argument(
        "--test_biochemical_features",
        type=str,
        default=None,
        help="Path to biochemical features .npy file "
        "(optional, for graph-based models)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for cross-species results",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="cross_species",
        help="Suffix for output filenames",
    )
    args = parser.parse_args()

    predict_cross_species(
        args.checkpoint,
        args.train_config,
        args.test_inputs,
        args.test_targets,
        args.test_biochemical_features,
        args.output_dir,
        args.suffix,
    )

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import lightning
import numpy as np
import torch
from torch.utils.data import DataLoader

from .lightning_utils import create_inference_trainer
from .model import HybriTELightningModule


def resolve_device(accelerator: str) -> torch.device:
    if accelerator in {"auto", "gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        accelerator in {"auto", "mps"}
        and mps_backend is not None
        and mps_backend.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_config: dict[str, Any],
    optimization_config: dict[str, Any],
    node_feature_dim: int,
    edge_feature_dim: int,
    num_targets: int,
    biochemical_feature_dim: int | None,
) -> HybriTELightningModule:
    del model_config, optimization_config, node_feature_dim
    del edge_feature_dim, num_targets, biochemical_feature_dim
    model = HybriTELightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )
    model.eval()
    return model


def _flatten_prediction_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for output in outputs:
        if isinstance(output, list):
            flattened.extend(cast(list[dict[str, Any]], output))
            continue
        flattened.append(cast(dict[str, Any], output))
    return flattened


def run_inference(
    model: lightning.LightningModule,
    dataloader: DataLoader[Any],
    training_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    trainer = create_inference_trainer(training_config)
    outputs = trainer.predict(
        model,
        dataloaders=dataloader,
        return_predictions=True,
    )
    if outputs is None:
        raise RuntimeError("Lightning predict did not return any outputs")

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ids: list[str] = []

    for batch_output in _flatten_prediction_outputs(outputs):
        ids.extend(cast(list[str], batch_output["sample_id"]))
        predictions.append(cast(torch.Tensor, batch_output["predictions"]).numpy())
        targets.append(cast(torch.Tensor, batch_output["targets"]).numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0), ids

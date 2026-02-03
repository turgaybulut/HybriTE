from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import MetricCollection

from .registry import LOSSES, METRICS, OPTIMIZERS, SCHEDULERS


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        config: dict[str, Any],
        train_indices: np.ndarray | None = None,
        val_indices: np.ndarray | None = None,
        test_indices: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.data_config = config["data"]
        root = Path(self.data_config.get("root", "."))
        self.data_root = root.expanduser().resolve()
        self.train_indices = self._to_numpy(train_indices)
        self.val_indices = self._to_numpy(val_indices)
        self.test_indices = self._to_numpy(test_indices)
        self.batch_size = int(self.data_config["batch_size"])
        self.num_workers = int(self.data_config.get("num_workers", 0))
        self.pin_memory = bool(self.data_config.get("pin_memory", True))
        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None

    def _to_numpy(
        self, indices: np.ndarray | list[int] | torch.Tensor | None
    ) -> np.ndarray | None:
        if indices is None:
            return None
        return np.asarray(indices, dtype=np.int64)

    def _resolve_indices(
        self, dataset_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.train_indices is not None and self.val_indices is not None:
            train_idx = self.train_indices
            val_idx = self.val_indices
            test_idx = self.test_indices if self.test_indices is not None else val_idx
            self.test_indices = test_idx
            return train_idx, val_idx, test_idx
        splits = self.data_config.get("splits", {})
        train_frac = float(splits.get("train", 0.8))
        val_frac = float(splits.get("val", 0.1))
        n_train = max(1, min(dataset_size, int(dataset_size * train_frac)))
        remaining = max(0, dataset_size - n_train)
        n_val = int(dataset_size * val_frac)
        n_val = max(1, min(remaining, n_val)) if remaining > 0 else 0
        indices = torch.randperm(dataset_size).numpy()
        train_idx = indices[:n_train]
        val_end = n_train + n_val
        val_idx = indices[n_train:val_end]
        test_idx = indices[val_end:] if val_end < dataset_size else val_idx
        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx
        return train_idx, val_idx, test_idx

    def _assign_datasets(
        self,
        dataset: Dataset[Any],
        indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        train_idx, val_idx, test_idx = indices or self._resolve_indices(len(dataset))
        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)
        self.test_dataset = Subset(dataset, test_idx)

    def _create_dataloader(
        self,
        dataset: Dataset[Any],
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Any | None = None,
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

    def _resolve_path(self, pathlike: str | Path) -> Path:
        path_obj = Path(pathlike)
        if path_obj.is_absolute():
            return path_obj
        return (self.data_root / path_obj).expanduser().resolve()


class BaseModel(LightningModule):
    def __init__(self, config: dict[str, Any], *, num_outputs: int) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.num_outputs = int(num_outputs)
        self.batch_size = int(self.config["data"]["batch_size"])
        self.loss_fn = LOSSES.get(self.config["loss"])()
        metric_names = self.config.get("metrics", [])
        self.metric_groups = nn.ModuleDict()
        if metric_names:
            metrics = [
                METRICS.get(name)(num_outputs=self.num_outputs) for name in metric_names
            ]
            base_collection = MetricCollection(metrics)
            for stage in ("train", "val", "test"):
                self.metric_groups[f"metrics_{stage}"] = base_collection.clone(
                    prefix=f"{stage}_"
                )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds, targets = self._shared_step(batch, "train")
        loss = self._compute_masked_loss(preds, targets)
        self._update_metrics("train", preds, targets)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        preds, targets = self._shared_step(batch, "val")
        loss = self._compute_masked_loss(preds, targets)
        self._update_metrics("val", preds, targets)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        preds, targets = self._shared_step(batch, "test")
        loss = self._compute_masked_loss(preds, targets)
        self._update_metrics("test", preds, targets)
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._predict(batch, "predict")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer_cfg = copy.deepcopy(self.hparams.optimizer)
        optimizer_name = optimizer_cfg.pop("name")
        optimizer = OPTIMIZERS.get(optimizer_name)(self.parameters(), **optimizer_cfg)
        if "scheduler" not in self.hparams:
            return {"optimizer": optimizer}
        scheduler_cfg = copy.deepcopy(self.hparams.scheduler)
        scheduler_name = scheduler_cfg.pop("name")
        scheduler = SCHEDULERS.get(scheduler_name)(optimizer, **scheduler_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def _predict(self, batch: Any, stage: str) -> torch.Tensor:
        if isinstance(batch, dict):
            return self(batch)
        if isinstance(batch, (list, tuple)):
            return self(batch[0])
        return self(batch)

    def _extract_target(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            return batch["target"]
        if isinstance(batch, (list, tuple)):
            return batch[1]
        raise RuntimeError("Unsupported batch format")

    def _shared_step(self, batch: Any, stage: str) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self._predict(batch, stage)
        targets = self._extract_target(batch)
        return preds, targets

    def _update_metrics(
        self, stage: str, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        metrics_key = f"metrics_{stage}"
        if metrics_key not in self.metric_groups:
            return
        preds_valid, targets_valid = self._prepare_metrics_input(preds, targets)
        self.metric_groups[metrics_key].update(preds_valid, targets_valid)

    def _log_metrics(self, stage: str) -> None:
        metrics_key = f"metrics_{stage}"
        if metrics_key not in self.metric_groups:
            return
        metrics = self.metric_groups[metrics_key]
        computed = metrics.compute()
        formatted = self._format_metrics(computed)
        self.log_dict(
            formatted,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        metrics.reset()

    def _compute_masked_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        if not mask.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        preds_valid = preds[mask]
        targets_valid = targets[mask]
        return self.loss_fn(preds_valid, targets_valid)

    def _prepare_metrics_input(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds_view = preds if preds.ndim > 1 else preds.unsqueeze(-1)
        targets_view = targets if targets.ndim > 1 else targets.unsqueeze(-1)
        mask = ~torch.isnan(targets_view)
        row_mask = mask.all(dim=1)
        if not row_mask.any():
            dummy = torch.zeros(1, preds_view.shape[1], device=preds.device)
            return dummy, dummy
        return preds_view[row_mask], targets_view[row_mask]

    def _format_metrics(
        self, metrics_dict: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        formatted: dict[str, float] = {}
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    formatted[key] = value.item()
                else:
                    formatted[key] = value.mean().item()
            else:
                formatted[key] = float(value)
        return formatted

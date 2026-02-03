from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data as GeoData

from framework.base import BaseDataModule
from framework.registry import DATAMODULES


class HybriTEDataset(Dataset[Any]):
    def __init__(
        self,
        graphs: list[GeoData],
        targets: torch.Tensor,
        biochemical_features: torch.Tensor | None = None,
    ) -> None:
        self.graphs = graphs
        self.targets = targets
        self.biochemical_features = biochemical_features
        assert len(self.graphs) == self.targets.shape[0]
        if self.biochemical_features is not None:
            assert len(self.graphs) == self.biochemical_features.shape[0]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {"graph": self.graphs[i], "target": self.targets[i]}
        if self.biochemical_features is not None:
            item["biochemical_features"] = self.biochemical_features[i]
        return item


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    targets = torch.stack([it["target"] for it in items])
    batch = Batch.from_data_list([it["graph"] for it in items])
    result = {"graph": batch, "target": targets}
    if "biochemical_features" in items[0]:
        result["biochemical_features"] = torch.stack(
            [it["biochemical_features"] for it in items]
        )
    return result


@DATAMODULES.register("hybrite_datamodule")
class HybriTEDataModule(BaseDataModule):
    def __init__(
        self,
        config: dict[str, Any],
        train_indices: np.ndarray | None = None,
        val_indices: np.ndarray | None = None,
        test_indices: np.ndarray | None = None,
    ) -> None:
        super().__init__(config, train_indices, val_indices, test_indices)

    def setup(self, stage: str | None = None) -> None:
        graphs = self._load_graphs(self.data_root / self.config["data"]["graphs_pt"])
        targets = self._select_target_columns(self._load_targets(graphs))
        biochemical_features = self._load_biochemical_features()

        node_dim, edge_dim = self._infer_dims(graphs)
        self.node_feature_dim = node_dim
        self.edge_feature_dim = edge_dim
        self.biochemical_feature_dim = (
            biochemical_features.shape[1] if biochemical_features is not None else None
        )
        self.num_targets = targets.shape[1]

        dataset = HybriTEDataset(graphs, targets, biochemical_features)

        indices = None
        if self.train_indices is not None and self.val_indices is not None:
            test_idx = (
                self.test_indices if self.test_indices is not None else self.val_indices
            )
            indices = (self.train_indices, self.val_indices, test_idx)
        self._assign_datasets(dataset, indices)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._create_dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self._create_dataloader(self.val_dataset, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self._create_dataloader(self.test_dataset, drop_last=False)

    @property
    def model_kwargs(self) -> dict[str, Any]:
        return {
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
            "biochemical_feature_dim": self.biochemical_feature_dim,
            "num_targets": self.num_targets,
        }

    def _create_dataloader(
        self, dataset: Dataset, shuffle: bool = False, drop_last: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=True,
            persistent_workers=self.config["data"]["num_workers"] > 0,
            collate_fn=_collate,
            drop_last=drop_last,
        )

    def _load_graphs(self, path: Path) -> list[GeoData]:
        graphs = torch.load(str(path), map_location="cpu", weights_only=False)
        return graphs

    def _load_targets(self, graphs: list[GeoData]) -> torch.Tensor:
        if "target_npy" in self.config["data"]:
            targets = torch.from_numpy(
                np.load(self.data_root / self.config["data"]["target_npy"])
            ).float()
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)
            return targets

        targets = torch.stack([g.y for g in graphs]).float()
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        return targets

    def _select_target_columns(self, targets: torch.Tensor) -> torch.Tensor:
        indices = self.config["data"].get("target_indices")
        if indices is None:
            return targets
        if isinstance(indices, int):
            index_list = [indices]
        else:
            index_list = list(indices)
        selection = torch.as_tensor(index_list, dtype=torch.long)
        assert torch.all(selection >= 0)
        return targets[:, selection]

    def _load_biochemical_features(self) -> torch.Tensor | None:
        if (
            "biochemical_features_npy" not in self.config["data"]
            or self.config["data"]["biochemical_features_npy"] is None
        ):
            return None
        features = torch.from_numpy(
            np.load(self.data_root / self.config["data"]["biochemical_features_npy"])
        ).float()
        return features

    def _infer_dims(self, graphs: list[GeoData]) -> tuple[int, int]:
        g0 = graphs[0]
        node_dim = g0.x.shape[-1]
        edge_dim = (
            g0.edge_attr.shape[-1]
            if hasattr(g0, "edge_attr") and g0.edge_attr is not None
            else 0
        )
        return node_dim, edge_dim

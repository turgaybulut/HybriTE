from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import lightning
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data as GeoData

from .io import load_json


@dataclass(frozen=True)
class PreparedDatasetBundle:
    prepared_dir: Path
    id_column: str
    ids: np.ndarray
    graphs: list[GeoData]
    targets: np.ndarray
    target_names: list[str]
    biochemical: np.ndarray | None
    biochemical_names: list[str]

    def selected_biochemistry(
        self,
        selected_columns: list[str] | None,
    ) -> np.ndarray | None:
        if not selected_columns:
            return None
        if self.biochemical is None:
            raise ValueError("Prepared dataset does not include biochemical features")
        name_to_index = {name: idx for idx, name in enumerate(self.biochemical_names)}
        missing = [name for name in selected_columns if name not in name_to_index]
        if missing:
            raise ValueError(f"Missing biochemical columns: {missing}")
        indices = [name_to_index[name] for name in selected_columns]
        return self.biochemical[:, indices]


@lru_cache(maxsize=8)
def load_prepared_bundle(prepared_dir: str | Path) -> PreparedDatasetBundle:
    prepared_path = Path(prepared_dir).expanduser().resolve()
    manifest = load_json(prepared_path / "dataset_manifest.json")
    ids = np.load(prepared_path / manifest["ids_npy"], allow_pickle=False)
    targets = np.load(
        prepared_path / manifest["targets_npy"], allow_pickle=False
    ).astype(
        np.float32,
        copy=False,
    )
    graphs = torch.load(manifest["graph_path"], map_location="cpu", weights_only=False)

    biochemical = None
    biochemical_names: list[str] = manifest.get("biochemical_columns", [])
    biochemical_npy = manifest.get("biochemistry_npy")
    if biochemical_npy is not None:
        biochemical = np.load(
            prepared_path / biochemical_npy, allow_pickle=False
        ).astype(
            np.float32,
            copy=False,
        )

    if len(ids) != len(graphs) or len(ids) != len(targets):
        raise ValueError("Prepared ids, graphs, and targets are misaligned")
    if biochemical is not None and len(ids) != len(biochemical):
        raise ValueError("Prepared ids and biochemical features are misaligned")

    return PreparedDatasetBundle(
        prepared_dir=prepared_path,
        id_column=manifest["id_column"],
        ids=ids,
        graphs=graphs,
        targets=targets,
        target_names=manifest["target_columns"],
        biochemical=biochemical,
        biochemical_names=biochemical_names,
    )


def load_split_manifest(prepared_dir: str | Path, fold: int) -> dict[str, Any]:
    split_path = (
        Path(prepared_dir).expanduser().resolve()
        / "folds"
        / f"fold_{fold:02d}"
        / "split_manifest.json"
    )
    return load_json(split_path)


def load_feature_manifest(prepared_dir: str | Path, fold: int) -> dict[str, Any]:
    feature_path = (
        Path(prepared_dir).expanduser().resolve()
        / "folds"
        / f"fold_{fold:02d}"
        / "feature_manifest.json"
    )
    return load_json(feature_path)


class HybriTESplitDataset(Dataset[Any]):
    def __init__(
        self,
        bundle: PreparedDatasetBundle,
        indices: np.ndarray,
        biochemical_features: np.ndarray | None,
    ) -> None:
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=np.int64)
        self.biochemical_features = biochemical_features

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = int(self.indices[index])
        item: dict[str, Any] = {
            "sample_id": str(self.bundle.ids[source_index]),
            "graph": self.bundle.graphs[source_index],
            "target": torch.from_numpy(self.bundle.targets[source_index]).float(),
        }
        if self.biochemical_features is not None:
            item["biochemical_features"] = torch.from_numpy(
                self.biochemical_features[source_index]
            ).float()
        return item


def collate_graph_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "sample_id": [item["sample_id"] for item in items],
        "graph": Batch.from_data_list([item["graph"] for item in items]),
        "target": torch.stack([item["target"] for item in items]),
    }
    if "biochemical_features" in items[0]:
        batch["biochemical_features"] = torch.stack(
            [item["biochemical_features"] for item in items]
        )
    return batch


def create_dataset(
    bundle: PreparedDatasetBundle,
    indices: np.ndarray,
    selected_columns: list[str] | None,
) -> HybriTESplitDataset:
    selected_biochemistry = bundle.selected_biochemistry(selected_columns)
    return HybriTESplitDataset(bundle, indices, selected_biochemistry)


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_graph_batch,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "sample_id":
            moved[key] = value
            continue
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class HybriTEDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        bundle: PreparedDatasetBundle,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        selected_columns: list[str] | None,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.val_indices = np.asarray(val_indices, dtype=np.int64)
        self.test_indices = np.asarray(test_indices, dtype=np.int64)
        self.selected_columns = selected_columns if selected_columns else None
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None

        first_graph = self.bundle.graphs[0]
        self.node_feature_dim = int(cast(Any, first_graph.x).shape[-1])
        edge_attr = first_graph.edge_attr
        self.edge_feature_dim = int(edge_attr.shape[-1]) if edge_attr is not None else 0
        self.num_targets = int(self.bundle.targets.shape[1])
        self.biochemical_feature_dim = (
            len(self.selected_columns) if self.selected_columns is not None else None
        )

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = create_dataset(
            self.bundle,
            self.train_indices,
            self.selected_columns,
        )
        self.val_dataset = create_dataset(
            self.bundle,
            self.val_indices,
            self.selected_columns,
        )
        self.test_dataset = create_dataset(
            self.bundle,
            self.test_indices,
            self.selected_columns,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError(
                "DataModule.setup() must be called before train_dataloader()"
            )
        return create_dataloader(
            self.train_dataset,
            self.batch_size,
            self.num_workers,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            raise RuntimeError(
                "DataModule.setup() must be called before val_dataloader()"
            )
        return create_dataloader(self.val_dataset, self.batch_size, self.num_workers)

    def test_dataloader(self) -> DataLoader[Any]:
        if self.test_dataset is None:
            raise RuntimeError(
                "DataModule.setup() must be called before test_dataloader()"
            )
        return create_dataloader(self.test_dataset, self.batch_size, self.num_workers)

from __future__ import annotations

from typing import Any, cast

import lightning
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GraphNorm
from torch_geometric.nn.aggr import AttentionalAggregation

from .lightning_utils import should_sync_dist


def _activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


def _pooling(name: str, node_dim: int) -> Any:
    if name != "attention":
        raise ValueError("Paper release supports only attention pooling")
    gate = nn.Sequential(
        nn.Linear(node_dim, node_dim // 2),
        nn.ReLU(),
        nn.Linear(node_dim // 2, 1),
    )
    return AttentionalAggregation(gate_nn=gate)


class HybriTE(nn.Module):
    def __init__(
        self,
        model_config: dict[str, Any],
        node_feature_dim: int,
        edge_feature_dim: int,
        num_targets: int,
        biochemical_feature_dim: int | None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        hidden_dim = int(model_config["hidden_dim"])
        self.dropedge_p = float(model_config["dropedge_p"])
        self.dropedge_struct_p = float(model_config["dropedge_struct_p"])
        self.struct_weight_scale = float(model_config["struct_weight_scale"])
        self.struct_message_scale = float(model_config["struct_message_scale"])

        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(model_config["num_layers"])):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=int(model_config["num_heads"]),
                    edge_dim=hidden_dim,
                    dropout=float(model_config["dropout"]),
                    concat=False,
                )
            )
            self.norms.append(GraphNorm(hidden_dim))

        self.pool = _pooling(str(model_config["pooling"]), hidden_dim)
        head_input_dim = hidden_dim

        self.biochemical_mlp = None
        if biochemical_feature_dim is not None:
            biochemical_input_dim = biochemical_feature_dim
            biochemical_hidden_dims = list(
                model_config.get("biochemical_hidden_dims", [])
            )
            if biochemical_hidden_dims:
                biochemical_layers: list[nn.Module] = []
                for hidden_units in biochemical_hidden_dims:
                    biochemical_layers.extend(
                        [
                            nn.Linear(biochemical_input_dim, int(hidden_units)),
                            _activation(str(model_config["biochemical_activation"])),
                            nn.Dropout(float(model_config["biochemical_dropout"])),
                        ]
                    )
                    biochemical_input_dim = int(hidden_units)
                self.biochemical_mlp = nn.Sequential(*biochemical_layers)
            head_input_dim += biochemical_input_dim

        head_layers: list[nn.Module] = []
        current_dim = head_input_dim
        for hidden_units in model_config["head_hidden_dims"]:
            head_layers.extend(
                [
                    nn.Linear(current_dim, int(hidden_units)),
                    _activation(str(model_config["head_activation"])),
                    nn.LayerNorm(int(hidden_units)),
                    nn.Dropout(float(model_config["head_dropout"])),
                ]
            )
            current_dim = int(hidden_units)
        head_layers.append(nn.Linear(current_dim, num_targets))
        self.head = nn.Sequential(*head_layers)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        graph_batch = cast(Any, batch["graph"])
        node_states = self.node_embedding(graph_batch.x)
        edge_index, edge_attr, struct_mask = self._prepare_edges(
            graph_batch.edge_index,
            graph_batch.edge_attr,
            node_states.device,
            self.training,
        )
        encoded_edges = self._encode_edges(edge_attr, struct_mask, node_states.device)

        for conv, norm in zip(self.convs, self.norms):
            node_states = conv(
                x=node_states,
                edge_index=edge_index,
                edge_attr=encoded_edges,
            )
            node_states = norm(node_states, batch=graph_batch.batch)

        pooled_graph = self.pool(node_states, graph_batch.batch)
        if "biochemical_features" in batch:
            biochemical_features = batch["biochemical_features"]
            if self.biochemical_mlp is not None:
                biochemical_features = self.biochemical_mlp(biochemical_features)
            pooled_graph = torch.cat([pooled_graph, biochemical_features], dim=1)
        return self.head(pooled_graph)

    def _prepare_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        device: torch.device,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        if edge_attr.numel() == 0:
            return (
                edge_index,
                edge_attr,
                torch.zeros(0, dtype=torch.bool, device=device),
            )

        struct_mask = (
            edge_attr[:, 1] > 0
            if edge_attr.shape[1] > 1
            else torch.zeros(edge_attr.shape[0], dtype=torch.bool, device=device)
        )

        if edge_attr.shape[1] >= 4:
            edge_attr = edge_attr.clone()
            edge_attr[struct_mask, 3] = torch.log1p(
                edge_attr[struct_mask, 3] * self.struct_weight_scale
            )

        if training and (self.dropedge_p > 0.0 or self.dropedge_struct_p > 0.0):
            drop_prob = torch.full(
                (edge_attr.shape[0],), self.dropedge_p, device=device
            )
            drop_prob[struct_mask] = self.dropedge_struct_p
            keep_mask = torch.rand_like(drop_prob) > drop_prob
            if not bool(keep_mask.any()):
                keep_mask[torch.argmax(drop_prob)] = True
            edge_index = edge_index[:, keep_mask]
            edge_attr = edge_attr[keep_mask]
            struct_mask = struct_mask[keep_mask]

        return edge_index, edge_attr, struct_mask

    def _encode_edges(
        self,
        edge_attr: torch.Tensor,
        struct_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if edge_attr.numel() == 0:
            return torch.empty((0, self.edge_embedding.out_features), device=device)

        encoded_edges = self.edge_embedding(edge_attr)
        encoded_edges = self.edge_mlp(encoded_edges)
        if struct_mask.any():
            encoded_edges[struct_mask] = (
                encoded_edges[struct_mask] * self.struct_message_scale
            )
        return encoded_edges


class HybriTELightningModule(lightning.LightningModule):
    def __init__(
        self,
        model_config: dict[str, Any],
        optimization_config: dict[str, Any],
        node_feature_dim: int,
        edge_feature_dim: int,
        num_targets: int,
        biochemical_feature_dim: int | None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "model_config": model_config,
                "optimization_config": optimization_config,
                "node_feature_dim": node_feature_dim,
                "edge_feature_dim": edge_feature_dim,
                "num_targets": num_targets,
                "biochemical_feature_dim": biochemical_feature_dim,
            }
        )
        self.model = HybriTE(
            model_config=model_config,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            num_targets=num_targets,
            biochemical_feature_dim=biochemical_feature_dim,
        )
        self.optimization_config = optimization_config

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss = self._masked_mse(predictions, batch["target"])
        batch_size = self._graph_batch_size(batch)
        self.log(
            "train_loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=should_sync_dist(self),
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss = self._masked_mse(predictions, batch["target"])
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self._graph_batch_size(batch),
            sync_dist=should_sync_dist(self),
        )
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss = self._masked_mse(predictions, batch["target"])
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self._graph_batch_size(batch),
            sync_dist=should_sync_dist(self),
        )
        return loss

    def predict_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        del batch_idx, dataloader_idx
        predictions = self(batch)
        return {
            "predictions": predictions.detach().cpu(),
            "targets": batch["target"].detach().cpu(),
            "sample_id": list(cast(list[str], batch["sample_id"])),
        }

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.optimization_config["lr"]),
            weight_decay=float(self.optimization_config["weight_decay"]),
            betas=tuple(self.optimization_config["betas"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.optimization_config["scheduler_factor"]),
            patience=int(self.optimization_config["scheduler_patience"]),
            min_lr=float(self.optimization_config["scheduler_min_lr"]),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    @staticmethod
    def _graph_batch_size(batch: dict[str, Any]) -> int:
        return int(cast(Any, batch["graph"]).num_graphs)

    @staticmethod
    def _masked_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = torch.isfinite(targets)
        if not bool(valid_mask.any()):
            return predictions.sum() * 0.0
        return torch.mean((predictions[valid_mask] - targets[valid_mask]) ** 2)

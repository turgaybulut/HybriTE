from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import (
    GATv2Conv,
    GraphNorm,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation

from framework.base import BaseModel
from framework.registry import MODELS


def _activation(name: str) -> nn.Module:
    mapping = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}
    return mapping[name.lower()]()


def _pooling(name: str, node_dim: int | None) -> Any:
    key = name.lower()
    if key == "attention":
        gate_nn = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2), nn.ReLU(), nn.Linear(node_dim // 2, 1)
        )
        return AttentionalAggregation(gate_nn=gate_nn)
    if key == "set2set":
        return Set2Set(in_channels=node_dim, processing_steps=4)
    mapping = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}
    return mapping[key]


@MODELS.register("hybrite_model")
class HybriTEModel(BaseModel):
    def __init__(self, config: dict[str, Any], **kwargs) -> None:
        super().__init__(config, num_outputs=kwargs["num_targets"])

        hybrite_cfg = self.config["model"]["hybrite"]
        head_cfg = self.config["model"]["head"]
        bio_feat_cfg = self.config["model"]["biochemical_features_mlp"]

        hidden = hybrite_cfg["hidden_dim"]
        assert kwargs["node_feature_dim"] > 0
        assert kwargs["edge_feature_dim"] > 0
        self.dropedge_p = hybrite_cfg["dropedge_p"]
        self.dropedge_struct_p = hybrite_cfg.get(
            "dropedge_struct_p", self.dropedge_p * 0.1
        )
        self.struct_weight_scale = hybrite_cfg.get("struct_weight_scale", 1.0)
        self.struct_message_scale = hybrite_cfg.get("struct_message_scale", 1.0)

        self.node_emb = nn.Linear(kwargs["node_feature_dim"], hidden)
        self.edge_emb = nn.Linear(kwargs["edge_feature_dim"], hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(hybrite_cfg["num_layers"]):
            conv = GATv2Conv(
                in_channels=hidden,
                out_channels=hidden,
                heads=hybrite_cfg["num_heads"],
                edge_dim=hidden,
                dropout=hybrite_cfg["dropout"],
                concat=False,
            )
            self.convs.append(conv)
            self.norms.append(GraphNorm(hidden))

        self.pool = _pooling(hybrite_cfg["pool"], hidden)

        mult = 2 if hybrite_cfg["pool"] == "set2set" else 1
        head_in = hidden * mult

        self.biochemical_features_mlp = None
        if kwargs.get("biochemical_feature_dim") is not None:
            bio_feat_in = kwargs["biochemical_feature_dim"]
            if (
                bio_feat_cfg["hidden_dims"] is not None
                and len(bio_feat_cfg["hidden_dims"]) > 0
            ):
                layers = []
                for u in bio_feat_cfg["hidden_dims"]:
                    layers.extend(
                        [
                            nn.Linear(bio_feat_in, u),
                            _activation(bio_feat_cfg["activation"]),
                            nn.Dropout(bio_feat_cfg["dropout"]),
                        ]
                    )
                    bio_feat_in = u
                self.biochemical_features_mlp = nn.Sequential(*layers)
            head_in += bio_feat_in

        head = []
        for u in head_cfg["hidden_dims"]:
            head.extend(
                [
                    nn.Linear(head_in, u),
                    _activation(head_cfg["activation"]),
                    nn.LayerNorm(u),
                    nn.Dropout(head_cfg["dropout"]),
                ]
            )
            head_in = u
        head.append(nn.Linear(head_in, kwargs["num_targets"]))
        self.head = nn.Sequential(*head)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        b: Batch = batch["graph"]
        x = self.node_emb(b.x)
        edge_index, edge_attr, struct_mask = self._prepare_edges(
            b.edge_index, b.edge_attr, x.device, self.training
        )
        edge_attr = self._encode_edges(edge_attr, struct_mask, x.device)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = norm(x, batch=b.batch)

        g = self.pool(x, b.batch)

        if "biochemical_features" in batch:
            bio_feat = batch["biochemical_features"]
            if self.biochemical_features_mlp is not None:
                bio_feat = self.biochemical_features_mlp(bio_feat)
            g = torch.cat([g, bio_feat], dim=1)

        return self.head(g)

    def _prepare_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        device: torch.device,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = edge_index.to(device)
        if edge_attr is None or edge_attr.numel() == 0:
            empty_attr = torch.zeros(
                (edge_index.size(1), self.edge_emb.in_features), device=device
            )
            empty_mask = torch.zeros(
                edge_index.size(1), dtype=torch.bool, device=device
            )
            return edge_index, empty_attr, empty_mask

        edge_attr = edge_attr.to(device)
        struct_mask = (
            edge_attr[:, 1] > 0
            if edge_attr.shape[1] > 1
            else torch.zeros(edge_attr.size(0), dtype=torch.bool, device=device)
        )

        if edge_attr.shape[1] >= 4:
            edge_attr = edge_attr.clone()
            edge_attr[struct_mask, 3] = torch.log1p(
                edge_attr[struct_mask, 3] * self.struct_weight_scale
            )

        if training and (self.dropedge_p > 0.0 or self.dropedge_struct_p > 0.0):
            drop_prob = torch.full((edge_attr.size(0),), self.dropedge_p, device=device)
            drop_prob[struct_mask] = self.dropedge_struct_p
            keep = torch.rand_like(drop_prob) > drop_prob
            if not keep.any():
                keep[torch.argmax(drop_prob)] = True
            edge_index = edge_index[:, keep]
            edge_attr = edge_attr[keep]
            struct_mask = struct_mask[keep]

        return edge_index, edge_attr, struct_mask

    def _encode_edges(
        self,
        edge_attr: torch.Tensor,
        struct_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if edge_attr.numel() == 0:
            return torch.empty((0, self.edge_emb.out_features), device=device)

        encoded = self.edge_emb(edge_attr)
        encoded = self.edge_mlp(encoded)

        if struct_mask.any():
            encoded[struct_mask] = encoded[struct_mask] * self.struct_message_scale

        return encoded

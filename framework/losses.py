from __future__ import annotations

import torch.nn as nn

from .registry import LOSSES

LOSSES.register("mse")(nn.MSELoss)
LOSSES.register("l1")(nn.L1Loss)
LOSSES.register("cross_entropy")(nn.CrossEntropyLoss)
LOSSES.register("bce")(nn.BCELoss)
LOSSES.register("bce_with_logits")(nn.BCEWithLogitsLoss)

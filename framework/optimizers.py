from __future__ import annotations

import torch.optim as optim

from .registry import OPTIMIZERS

OPTIMIZERS.register("adam")(optim.Adam)
OPTIMIZERS.register("adamw")(optim.AdamW)
OPTIMIZERS.register("sgd")(optim.SGD)
OPTIMIZERS.register("rmsprop")(optim.RMSprop)

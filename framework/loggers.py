from __future__ import annotations

from pytorch_lightning.loggers import TensorBoardLogger

from .registry import LOGGERS

LOGGERS.register("tensorboard")(TensorBoardLogger)

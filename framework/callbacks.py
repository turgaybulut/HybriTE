from __future__ import annotations

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

from .registry import CALLBACKS

CALLBACKS.register("early_stopping")(EarlyStopping)
CALLBACKS.register("model_checkpoint")(ModelCheckpoint)
CALLBACKS.register("learning_rate_monitor")(LearningRateMonitor)
CALLBACKS.register("rich_progress_bar")(RichProgressBar)
CALLBACKS.register("rich_model_summary")(RichModelSummary)

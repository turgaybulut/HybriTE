from __future__ import annotations

from typing import Any

from torchmetrics.regression import (
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
)

from .registry import METRICS


class R2ScoreWrapper(R2Score):
    def __init__(self, num_outputs: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)


METRICS.register("mse")(MeanSquaredError)
METRICS.register("r2")(R2ScoreWrapper)
METRICS.register("pearson")(PearsonCorrCoef)
METRICS.register("spearman")(SpearmanCorrCoef)

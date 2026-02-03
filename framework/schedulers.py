from __future__ import annotations

import torch.optim.lr_scheduler as sched

from .registry import SCHEDULERS

SCHEDULERS.register("reduce_lr_on_plateau")(sched.ReduceLROnPlateau)
SCHEDULERS.register("step_lr")(sched.StepLR)
SCHEDULERS.register("multi_step_lr")(sched.MultiStepLR)
SCHEDULERS.register("exponential_lr")(sched.ExponentialLR)
SCHEDULERS.register("cosine_annealing_lr")(sched.CosineAnnealingLR)
SCHEDULERS.register("cosine_annealing_warm_restarts")(sched.CosineAnnealingWarmRestarts)

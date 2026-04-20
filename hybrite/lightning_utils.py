from __future__ import annotations

import os
from typing import Any

import lightning


def is_global_zero_process() -> bool:
    rank = os.environ.get("RANK")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return True


def should_sync_dist(module: lightning.LightningModule) -> bool:
    trainer = module.trainer
    return trainer is not None and trainer.world_size > 1


def create_training_trainer(
    training_config: dict[str, Any],
    *,
    callbacks: list[Any] | None = None,
    logger: Any | None = None,
    default_root_dir: str | None = None,
) -> lightning.Trainer:
    trainer_kwargs: dict[str, Any] = {
        "accelerator": training_config["accelerator"],
        "devices": training_config["devices"],
        "precision": training_config["precision"],
        "max_epochs": int(training_config["max_epochs"]),
        "gradient_clip_val": float(training_config["gradient_clip_val"]),
        "deterministic": True,
        "use_distributed_sampler": bool(
            training_config.get("use_distributed_sampler", True)
        ),
    }
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks
    if logger is not None:
        trainer_kwargs["logger"] = logger
    if default_root_dir is not None:
        trainer_kwargs["default_root_dir"] = default_root_dir
    strategy = training_config.get("strategy")
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    return lightning.Trainer(**trainer_kwargs)


def create_inference_trainer(training_config: dict[str, Any]) -> lightning.Trainer:
    trainer_kwargs: dict[str, Any] = {
        "accelerator": training_config["accelerator"],
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
    }
    precision = training_config.get("precision")
    if precision is not None:
        trainer_kwargs["precision"] = precision
    return lightning.Trainer(**trainer_kwargs)

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
from pathlib import Path

import numpy as np
from pytorch_lightning import Trainer
from sklearn.model_selection import KFold

from framework.config import load_config
from framework.registry import CALLBACKS, DATAMODULES, LOGGERS, MODELS
from framework.utils import seed_everything


def _import_plugin(plugin_name: str) -> None:
    importlib.import_module(f"plugins.{plugin_name}.datamodule")
    importlib.import_module(f"plugins.{plugin_name}.model")


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed_everything(config["seed"], workers=True)

    plugin_name = config["plugin"]
    _import_plugin(plugin_name)

    datamodule_name = f"{plugin_name}_datamodule"
    datamodule_full = DATAMODULES.get(datamodule_name)(config)
    datamodule_full.setup()
    dataset_size = (
        len(datamodule_full.train_dataset)
        + len(datamodule_full.val_dataset)
        + len(datamodule_full.test_dataset)
    )
    indices = np.arange(dataset_size)

    del datamodule_full
    gc.collect()

    n_folds = config.get("n_folds", 10)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config["seed"])

    all_test_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"========== Fold {fold + 1}/{n_folds} ==========")

        datamodule = DATAMODULES.get(datamodule_name)(
            config, train_indices=train_idx, val_indices=val_idx
        )
        datamodule.setup(stage="fit")

        model_kwargs = datamodule.model_kwargs
        model_name = f"{plugin_name}_model"
        model = MODELS.get(model_name)(config, **model_kwargs)

        callbacks = []
        for callback_spec in config["callbacks"]:
            params = copy.deepcopy(callback_spec)
            registry_name = params.pop("name")
            if registry_name == "model_checkpoint":
                fold_dir = Path(params["dirpath"]).resolve() / f"fold_{fold:02d}"
                params["dirpath"] = str(fold_dir / "checkpoints")
            callbacks.append(CALLBACKS.get(registry_name)(**params))

        logger_config = config.get("logger")
        logger = None
        if logger_config:
            logger_params = copy.deepcopy(
                {k: v for k, v in logger_config.items() if k != "name"}
            )
            logger_params["save_dir"] = str(fold_dir)
            logger_params["name"] = ""
            logger = LOGGERS.get(logger_config["name"])(**logger_params)

        trainer = Trainer(
            callbacks=callbacks,
            logger=logger,
            **config["trainer"],
        )

        trainer.fit(model=model, datamodule=datamodule)
        test_metrics = trainer.test(ckpt_path="best", datamodule=datamodule)
        all_test_metrics.append(test_metrics[0])

    print("========== Cross-Validation Summary ==========")
    avg_metrics = {
        k: np.mean([m[k] for m in all_test_metrics]) for k in all_test_metrics[0]
    }
    std_metrics = {
        k: np.std([m[k] for m in all_test_metrics]) for k in all_test_metrics[0]
    }
    for k, v in avg_metrics.items():
        print(f"Average {k}: {v:.4f} ± {std_metrics[k]:.4f}")

    results = {
        "config_path": config_path,
        "n_folds": n_folds,
        "fold_metrics": all_test_metrics,
        "avg_metrics": {k: float(v) for k, v in avg_metrics.items()},
        "std_metrics": {k: float(v) for k, v in std_metrics.items()},
    }

    results_dir = Path(config["output_dir"]).resolve() / plugin_name
    with open(results_dir / "cross_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)

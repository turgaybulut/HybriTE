from __future__ import annotations

import argparse
import copy
import importlib
from pathlib import Path

from pytorch_lightning import Trainer

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
    datamodule = DATAMODULES.get(datamodule_name)(config)
    datamodule.setup(stage="fit")

    model_kwargs = datamodule.model_kwargs
    model_name = f"{plugin_name}_model"
    model = MODELS.get(model_name)(config, **model_kwargs)

    callbacks = []
    for callback_spec in config["callbacks"]:
        params = copy.deepcopy(callback_spec)
        registry_name = params.pop("name")
        if registry_name == "model_checkpoint":
            params["dirpath"] = str(Path(params["dirpath"]).resolve() / "checkpoints")
        callbacks.append(CALLBACKS.get(registry_name)(**params))

    logger_config = config.get("logger")
    logger = None
    if logger_config:
        logger_params = copy.deepcopy(
            {k: v for k, v in logger_config.items() if k != "name"}
        )
        logger_params["name"] = plugin_name
        logger = LOGGERS.get(logger_config["name"])(**logger_params)

    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        **config["trainer"],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)

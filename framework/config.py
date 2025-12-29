from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def load_config(path: str) -> OmegaConf:
    p = Path(path).resolve()
    config = OmegaConf.load(p)

    plugin_name = config.get("plugin")
    if not plugin_name:
        raise ValueError("Plugin not specified in config.")

    plugin_config_path = Path(f"plugins/{plugin_name}/config.yaml").resolve()
    if plugin_config_path.exists():
        plugin_config = OmegaConf.load(plugin_config_path)
        config = OmegaConf.merge(plugin_config, config)
    return config

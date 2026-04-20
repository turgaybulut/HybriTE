from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}

    parent_name = config.pop("extends", None)
    if parent_name is not None:
        parent_path = (config_path.parent / parent_name).resolve()
        parent_config = load_config(parent_path)
        config = _merge_dicts(parent_config, config)

    config["config_path"] = str(config_path)
    config["config_name"] = config.get("name", config_path.stem)
    return config


def resolve_repo_path(pathlike: str | Path | None) -> Path | None:
    if pathlike is None:
        return None
    path = Path(pathlike).expanduser()
    if path.is_absolute():
        return path.absolute()
    return (REPO_ROOT / path).absolute()


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing config section: {key}")
    return value

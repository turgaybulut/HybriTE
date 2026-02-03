from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[Any], Any]:
        def decorator(item: Any) -> Any:
            key = (name or item.__name__).lower()
            if key in self._items:
                raise ValueError(
                    f"Item {key} already registered in {self._name} registry."
                )
            self._items[key] = item
            return item

        return decorator

    def get(self, name: str) -> Any:
        key = name.lower()
        if key not in self._items:
            raise ValueError(f"Item {key} not found in {self._name} registry.")
        return self._items[key]

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Registry(name={self.name}, items={list(self._items.keys())})"


LOSSES = Registry("Losses")
METRICS = Registry("Metrics")
OPTIMIZERS = Registry("Optimizers")
SCHEDULERS = Registry("Schedulers")
CALLBACKS = Registry("Callbacks")
DATAMODULES = Registry("DataModules")
MODELS = Registry("Models")
LOGGERS = Registry("Loggers")

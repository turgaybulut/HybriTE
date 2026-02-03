import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from lightning.pytorch import seed_everything

__all__ = ["seed_everything"]

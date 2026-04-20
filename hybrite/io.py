from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(Path(path)) as handle:
        return json.load(handle)


def write_matrix_csv(
    path: str | Path,
    values: np.ndarray,
    column_names: list[str],
    ids: list[str] | np.ndarray | None = None,
    id_column: str | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(np.asarray(values), columns=column_names)
    if ids is not None:
        frame.insert(0, id_column or "sample_id", list(ids))
    frame.to_csv(output_path, index=False)


def read_matrix_csv(
    path: str | Path,
    id_column: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray, list[str], str | None]:
    frame = pd.read_csv(path)
    resolved_id_column = None

    if id_column is not None and id_column in frame.columns:
        resolved_id_column = id_column
    elif len(frame.columns) > 0 and not pd.api.types.is_numeric_dtype(
        frame.dtypes.iloc[0]
    ):
        resolved_id_column = str(frame.columns[0])

    ids = None
    if resolved_id_column is not None:
        ids = frame.pop(resolved_id_column).astype(str).to_numpy()

    values = frame.to_numpy(dtype=np.float32)
    return ids, values, frame.columns.tolist(), resolved_id_column

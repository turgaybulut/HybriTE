from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FileContext:
    species: str
    model: str
    fold: str | None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=root / "models")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "aggregated_model_data"
    )
    return parser.parse_args()


def collect_groups(models_dir: Path) -> dict[str, list[Path]]:
    assert models_dir.is_dir()
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(models_dir.rglob("*.csv")):
        groups[path.name].append(path)
    return groups


def detect_context(path: Path, models_dir: Path) -> FileContext:
    relative = path.relative_to(models_dir)
    parts = relative.parts
    assert len(parts) >= 3

    species, model = parts[0], parts[1]
    fold = next((part for part in parts[2:-1] if part.startswith("fold")), None)
    return FileContext(species, model, fold)


def prepend_context(
    df: pd.DataFrame, context: FileContext, include_fold: bool
) -> pd.DataFrame:
    frame = df.copy()
    frame.insert(0, "model", context.model)
    frame.insert(0, "species", context.species)
    if include_fold:
        assert context.fold is not None
        frame.insert(2, "fold", context.fold)
    return frame


def aggregate_file(
    name: str, paths: Iterable[Path], models_dir: Path, output_dir: Path
) -> None:
    path_list = list(paths)
    contexts = [detect_context(path, models_dir) for path in path_list]

    fold_presence = {context.fold is not None for context in contexts}
    assert len(fold_presence) == 1
    include_fold = fold_presence.pop()

    frames: list[pd.DataFrame] = []
    column_order: list[str] = []

    for path, context in zip(path_list, contexts):
        df = pd.read_csv(path)
        df.columns = [
            column.strip() if isinstance(column, str) else column
            for column in df.columns
        ]
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        for column in df.columns:
            if column not in column_order:
                column_order.append(column)
        frames.append(prepend_context(df, context, include_fold))

    combined = pd.concat(frames, ignore_index=True)
    sort_keys = ["species", "model"] + (["fold"] if include_fold else [])
    ordered_columns = sort_keys + column_order
    combined = combined[ordered_columns]
    combined = combined.sort_values(sort_keys).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / f"agg_{name}", index=False)


def main() -> None:
    args = parse_args()
    groups = collect_groups(args.models_dir.resolve())
    for name, paths in sorted(groups.items()):
        aggregate_file(
            name, paths, args.models_dir.resolve(), args.output_dir.resolve()
        )


if __name__ == "__main__":
    main()

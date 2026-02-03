from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def select_by_regex(columns: Iterable[str], patterns: Sequence[str]) -> list[str]:
    compiled = [re.compile(p) for p in patterns]
    return [c for c in list(columns) if any(p.search(c) for p in compiled)]


def build_feature_matrix(
    df: pd.DataFrame,
    feature_patterns: list[str],
    targets: np.ndarray | None = None,
    select_k: int | None = None,
    feature_subset: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    cols = feature_subset or select_by_regex(df.columns, feature_patterns)
    assert cols, "No feature columns found matching the patterns"

    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing feature columns: {missing}"

    X = df[cols].values.astype(np.float32)

    if feature_subset is None and select_k is not None and targets is not None:
        assert select_k > 0, "select_k must be positive"
        k = min(select_k, X.shape[1])
        y_flat = targets[:, 0] if targets.ndim > 1 else targets
        selector = SelectKBest(score_func=f_regression, k=k)
        X = selector.fit_transform(X, y_flat)
        support_mask: np.ndarray = np.asarray(selector.get_support(indices=False))
        cols = [c for c, m in zip(cols, support_mask.tolist(), strict=False) if m]

    return X, cols


def build_targets_matrix(
    df: pd.DataFrame, target_patterns: list[str]
) -> tuple[np.ndarray, list[str]]:
    cols = select_by_regex(df.columns, target_patterns)
    assert cols, "No target columns found matching the patterns"
    y = df[cols].values.astype(np.float32)
    return y, cols


def save_npy(x: np.ndarray | None, out: Path) -> None:
    if x is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, x)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def run(args: argparse.Namespace) -> None:
    df = read_csv(args.input_csv)

    targets, target_cols = build_targets_matrix(df, args.target_cols_regex)

    feature_subset = None
    if args.feature_meta:
        with open(Path(args.feature_meta)) as f:
            feature_subset = json.load(f).get("feature_columns") or []
        assert feature_subset, "feature_meta missing feature_columns"

    features, feature_cols = build_feature_matrix(
        df,
        args.feature_cols_regex,
        targets,
        None if feature_subset else args.select_k,
        feature_subset,
    )

    out_dir = Path(args.out_dir)
    save_npy(targets, out_dir / "target.npy")
    save_npy(features, out_dir / "feature.npy")
    save_json(
        {"target_columns": target_cols, "feature_columns": feature_cols},
        out_dir / "meta.json",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CSV data to NumPy arrays for HybriTE training"
    )
    p.add_argument("--input_csv", type=str, required=True, help="Input CSV file path")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument(
        "--feature_cols_regex",
        nargs="*",
        default=[r"^biochemical_.*$"],
        help="Regex patterns for feature columns",
    )
    p.add_argument(
        "--target_cols_regex",
        nargs="*",
        default=[r"^bio_source_.*$"],
        help="Regex patterns for target columns",
    )
    p.add_argument(
        "--select_k",
        type=int,
        default=None,
        help="Select top k features using f-regression",
    )
    p.add_argument(
        "--feature_meta",
        type=str,
        default=None,
        help="Path to meta.json with predefined feature columns",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

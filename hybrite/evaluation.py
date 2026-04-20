from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .io import save_json, write_matrix_csv

METRICS = ["pearson", "spearman", "r2", "mae", "mse"]


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, found shape {array.shape}")
    return array


def _safe_float(value: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {metric: float("nan") for metric in METRICS}

    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float("nan"),
        "pearson": float("nan"),
        "spearman": float("nan"),
    }

    if len(y_true) < 2:
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result["r2"] = float(r2_score(y_true, y_pred))
        except ValueError:
            result["r2"] = float("nan")
        try:
            result["pearson"] = float(pearsonr(y_true, y_pred).statistic)
        except ValueError:
            result["pearson"] = float("nan")
        try:
            result["spearman"] = float(spearmanr(y_true, y_pred).statistic)
        except ValueError:
            result["spearman"] = float("nan")
    return result


def _mean_te_vectors(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(targets)
    counts = mask.sum(axis=1)
    valid_rows = counts > 0
    valid_counts = counts[valid_rows].astype(np.float32)
    masked_predictions = predictions[valid_rows] * mask[valid_rows]
    prediction_mean = masked_predictions.sum(axis=1) / valid_counts
    target_mean = np.nansum(targets[valid_rows], axis=1) / valid_counts
    return prediction_mean, target_mean, valid_rows


def _bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if bootstrap_samples <= 0 or len(y_true) < 2:
        return {
            metric: {"lower": float("nan"), "upper": float("nan")} for metric in METRICS
        }

    rng = np.random.default_rng(seed)
    metric_samples: dict[str, list[float]] = {metric: [] for metric in METRICS}

    for _ in range(bootstrap_samples):
        sample_index = rng.integers(0, len(y_true), len(y_true))
        sample_metrics = _regression_metrics(y_true[sample_index], y_pred[sample_index])
        for metric, value in sample_metrics.items():
            if math.isfinite(value):
                metric_samples[metric].append(value)

    intervals: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        samples = metric_samples[metric]
        if not samples:
            intervals[metric] = {"lower": float("nan"), "upper": float("nan")}
            continue
        intervals[metric] = {
            "lower": float(np.percentile(samples, 2.5)),
            "upper": float(np.percentile(samples, 97.5)),
        }
    return intervals


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: list[str],
    ids: list[str] | np.ndarray | None = None,
    bootstrap_samples: int = 200,
    seed: int = 654,
) -> dict[str, Any]:
    pred_array = _as_2d(predictions)
    target_array = _as_2d(targets)

    if pred_array.shape != target_array.shape:
        raise ValueError(
            "Prediction shape "
            f"{pred_array.shape} does not match target shape {target_array.shape}"
        )
    if len(target_names) != pred_array.shape[1]:
        raise ValueError("Target names do not match the prediction width")

    per_target_rows: list[dict[str, Any]] = []
    for column_index, target_name in enumerate(target_names):
        valid = np.isfinite(target_array[:, column_index]) & np.isfinite(
            pred_array[:, column_index]
        )
        metrics = _regression_metrics(
            target_array[valid, column_index], pred_array[valid, column_index]
        )
        per_target_rows.append(
            {
                "target_name": target_name,
                "n": int(valid.sum()),
                **metrics,
            }
        )

    per_target_frame = pd.DataFrame(per_target_rows)

    macro_metrics = {
        metric: float(np.nanmean(per_target_frame[metric].to_numpy(dtype=np.float64)))
        for metric in METRICS
    }
    macro_target_count = int((per_target_frame["n"] > 0).sum())

    mean_te_prediction, mean_te_target, valid_rows = _mean_te_vectors(
        pred_array,
        target_array,
    )
    mean_te_metrics = _regression_metrics(mean_te_target, mean_te_prediction)
    mean_te_intervals = _bootstrap_intervals(
        mean_te_target,
        mean_te_prediction,
        bootstrap_samples,
        seed,
    )

    if ids is None:
        mean_te_ids = np.arange(len(pred_array))[valid_rows].astype(str)
        id_column = "sample_id"
    else:
        mean_te_ids = np.asarray(ids).astype(str)[valid_rows]
        id_column = "sample_id"

    mean_te_frame = pd.DataFrame(
        {
            id_column: mean_te_ids,
            "target_mean_te": mean_te_target,
            "prediction_mean_te": mean_te_prediction,
        }
    )

    aggregate_rows: list[dict[str, Any]] = []
    for metric, value in mean_te_metrics.items():
        interval = mean_te_intervals[metric]
        aggregate_rows.append(
            {
                "scope": "mean_te",
                "metric": metric,
                "value": value,
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
                "n": int(len(mean_te_target)),
            }
        )
    for metric, value in macro_metrics.items():
        aggregate_rows.append(
            {
                "scope": "macro_target",
                "metric": metric,
                "value": value,
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n": macro_target_count,
            }
        )
    aggregate_frame = pd.DataFrame(aggregate_rows)

    summary = {
        "n_samples": int(pred_array.shape[0]),
        "n_targets": int(pred_array.shape[1]),
        "aggregate": {
            "mean_te": {
                "n": int(len(mean_te_target)),
                "metrics": {
                    metric: _safe_float(value)
                    for metric, value in mean_te_metrics.items()
                },
                "confidence_intervals": {
                    metric: {
                        "lower": _safe_float(mean_te_intervals[metric]["lower"]),
                        "upper": _safe_float(mean_te_intervals[metric]["upper"]),
                    }
                    for metric in METRICS
                },
            },
            "macro_target": {
                "n": macro_target_count,
                "metrics": {
                    metric: _safe_float(value)
                    for metric, value in macro_metrics.items()
                },
            },
        },
    }

    return {
        "summary": summary,
        "aggregate_metrics": aggregate_frame,
        "per_target_metrics": per_target_frame,
        "mean_te_table": mean_te_frame,
    }


def evaluate_and_save(
    output_dir: str | Path,
    predictions: np.ndarray,
    target_names: list[str],
    ids: list[str] | np.ndarray | None = None,
    targets: np.ndarray | None = None,
    id_column: str = "gene_id",
    bootstrap_samples: int = 200,
    seed: int = 654,
) -> dict[str, Any] | None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_matrix_csv(
        output_path / "predictions.csv",
        predictions,
        target_names,
        ids=ids,
        id_column=id_column,
    )

    if targets is None:
        return None

    write_matrix_csv(
        output_path / "targets.csv",
        targets,
        target_names,
        ids=ids,
        id_column=id_column,
    )

    evaluation = evaluate_predictions(
        predictions,
        targets,
        target_names,
        ids=ids,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    mean_te_table = evaluation["mean_te_table"]
    if "sample_id" in mean_te_table.columns:
        evaluation["mean_te_table"] = mean_te_table.rename(
            columns={"sample_id": id_column}
        )

    evaluation["aggregate_metrics"].to_csv(
        output_path / "aggregate_metrics.csv",
        index=False,
    )
    evaluation["per_target_metrics"].to_csv(
        output_path / "per_target_metrics.csv",
        index=False,
    )
    evaluation["mean_te_table"].to_csv(output_path / "mean_te.csv", index=False)
    save_json(evaluation["summary"], output_path / "summary.json")
    return evaluation["summary"]

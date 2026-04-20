from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

from .io import save_json

HIGHER_IS_BETTER = {"pearson", "spearman", "r2"}
LOWER_IS_BETTER = {"mae", "mse"}


def _metric_direction(metric: str) -> int:
    if metric in HIGHER_IS_BETTER:
        return 1
    if metric in LOWER_IS_BETTER:
        return -1
    raise ValueError(f"Unsupported metric direction: {metric}")


def load_run_metrics(run_dir: str | Path, split_name: str = "test") -> pd.DataFrame:
    run_path = Path(run_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for fold_dir in sorted(run_path.glob("fold_*")):
        aggregate_path = fold_dir / split_name / "aggregate_metrics.csv"
        if not aggregate_path.exists():
            continue
        aggregate_frame = pd.read_csv(aggregate_path)
        aggregate_frame.insert(0, "fold", int(fold_dir.name.split("_")[-1]))
        rows.extend(aggregate_frame.to_dict(orient="records"))

    if not rows:
        cross_validation_path = run_path / "cross_validation_metrics.csv"
        if cross_validation_path.exists():
            return pd.read_csv(cross_validation_path)
        raise ValueError(f"No fold metrics found under {run_path}")

    return pd.DataFrame(rows)


def compare_run_directories(
    candidate_dir: str | Path,
    reference_dir: str | Path,
    split_name: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate_metrics = load_run_metrics(candidate_dir, split_name)
    reference_metrics = load_run_metrics(reference_dir, split_name)

    paired = candidate_metrics.merge(
        reference_metrics,
        on=["fold", "scope", "metric"],
        suffixes=("_candidate", "_reference"),
    )
    if paired.empty:
        raise ValueError("No matched fold metrics were found between the two runs")

    paired = paired.rename(
        columns={
            "value_candidate": "candidate_value",
            "value_reference": "reference_value",
            "n_candidate": "candidate_n",
            "n_reference": "reference_n",
        }
    )
    paired = paired[
        [
            "fold",
            "scope",
            "metric",
            "candidate_value",
            "reference_value",
            "candidate_n",
            "reference_n",
        ]
    ].copy()
    paired["delta"] = paired["candidate_value"] - paired["reference_value"]
    paired["improvement_direction"] = paired["metric"].map(_metric_direction)
    paired["improvement_delta"] = paired["delta"] * paired["improvement_direction"]

    summary_rows: list[dict[str, Any]] = []
    for (scope, metric), group in paired.groupby(["scope", "metric"], sort=False):
        improvements = group["improvement_delta"].to_numpy(dtype=np.float64)
        finite_improvements = improvements[np.isfinite(improvements)]
        wilcoxon_p = None
        ttest_p = None
        if len(finite_improvements) >= 2 and not np.allclose(finite_improvements, 0.0):
            wilcoxon_p = float(
                wilcoxon(
                    finite_improvements,
                    zero_method="pratt",
                    alternative="two-sided",
                ).pvalue
            )
            ttest_p = float(ttest_1samp(finite_improvements, popmean=0.0).pvalue)

        summary_rows.append(
            {
                "scope": scope,
                "metric": metric,
                "n_folds": int(len(group)),
                "candidate_mean": float(group["candidate_value"].mean()),
                "reference_mean": float(group["reference_value"].mean()),
                "delta_mean": float(group["delta"].mean()),
                "improvement_mean": float(group["improvement_delta"].mean()),
                "improvement_std": float(group["improvement_delta"].std(ddof=1))
                if len(group) > 1
                else float("nan"),
                "improved_folds": int((group["improvement_delta"] > 0).sum()),
                "tied_folds": int((group["improvement_delta"] == 0).sum()),
                "worse_folds": int((group["improvement_delta"] < 0).sum()),
                "wilcoxon_p": wilcoxon_p,
                "paired_t_p": ttest_p,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["scope", "metric"], kind="stable").reset_index(
        drop=True
    )
    payload = {
        "candidate_dir": str(Path(candidate_dir).expanduser().resolve()),
        "reference_dir": str(Path(reference_dir).expanduser().resolve()),
        "split_name": split_name,
        "paired_metrics": paired.to_dict(orient="records"),
        "summary": [
            {
                key: (
                    None
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
                for key, value in record.items()
            }
            for record in summary.to_dict(orient="records")
        ],
    }
    return paired, summary, payload


def save_comparison_outputs(
    output_dir: str | Path,
    paired_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    payload: dict[str, Any],
) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    paired_metrics.to_csv(output_path / "paired_fold_metrics.csv", index=False)
    summary.to_csv(output_path / "significance_summary.csv", index=False)
    save_json(payload, output_path / "significance_summary.json")

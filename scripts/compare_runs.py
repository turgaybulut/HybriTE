from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    from hybrite.significance import (
        compare_run_directories,
        save_comparison_outputs,
    )

    args = parse_args()
    paired_metrics, summary, payload = compare_run_directories(
        candidate_dir=args.candidate_dir,
        reference_dir=args.reference_dir,
        split_name=args.split,
    )

    output_dir = args.output_dir
    if output_dir is None:
        candidate_name = Path(args.candidate_dir).expanduser().resolve().name
        reference_name = Path(args.reference_dir).expanduser().resolve().name
        output_dir = str(
            Path(args.candidate_dir).expanduser().resolve()
            / "comparisons"
            / f"{candidate_name}_vs_{reference_name}"
        )
    save_comparison_outputs(output_dir, paired_metrics, summary, payload)

    display_columns = [
        "scope",
        "metric",
        "n_folds",
        "candidate_mean",
        "reference_mean",
        "improvement_mean",
        "wilcoxon_p",
        "paired_t_p",
    ]
    print(summary.loc[:, display_columns].to_string(index=False))
    print(f"\nSaved comparison outputs to {Path(output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()

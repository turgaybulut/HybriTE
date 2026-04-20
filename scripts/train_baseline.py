from __future__ import annotations

import argparse
import gc

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--all-folds", action="store_true")
    return parser.parse_args()


def main() -> None:
    import lightning

    from hybrite.baselines import summarize_baseline_results, train_baseline_fold
    from hybrite.config import load_config

    args = parse_args()
    config = load_config(args.config)
    lightning.seed_everything(int(config["seed"]), workers=True)

    if args.all_folds:
        fold_count = int(config["folds"]["count"])
        for fold in tqdm(range(fold_count), desc="Baseline folds", unit="fold"):
            train_baseline_fold(config, fold)
            gc.collect()
        summarize_baseline_results(config)
        return

    if args.fold is None:
        raise ValueError("Pass --fold or --all-folds")
    train_baseline_fold(config, args.fold)
    summarize_baseline_results(config)


if __name__ == "__main__":
    main()

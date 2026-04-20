from __future__ import annotations

import argparse
import gc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--all-folds", action="store_true")
    return parser.parse_args()


def main() -> None:
    import lightning

    from hybrite.config import load_config
    from hybrite.train import summarize_results, train_fold

    args = parse_args()
    config = load_config(args.config)
    lightning.seed_everything(int(config["seed"]), workers=True)

    if args.all_folds:
        for fold in range(int(config["folds"]["count"])):
            train_fold(config, fold)
            gc.collect()
        summarize_results(config)
        return

    if args.fold is None:
        raise ValueError("Pass --fold or --all-folds")
    train_fold(config, args.fold)
    summarize_results(config)


if __name__ == "__main__":
    main()

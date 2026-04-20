from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    from hybrite.config import load_config
    from hybrite.prepare import prepare_dataset

    args = parse_args()
    prepare_dataset(load_config(args.config))


if __name__ == "__main__":
    main()

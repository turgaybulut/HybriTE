from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    from hybrite.config import load_config, resolve_repo_path
    from hybrite.graph_builder import build_graphs

    args = parse_args()
    config = load_config(args.config)
    output_path = (
        resolve_repo_path(args.output_path) if args.output_path is not None else None
    )
    build_graphs(config, output_path=output_path, limit=args.limit)


if __name__ == "__main__":
    main()

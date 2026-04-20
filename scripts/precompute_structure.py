from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    from hybrite.config import load_config, resolve_repo_path
    from hybrite.structure_cache import precompute_structure_cache

    args = parse_args()
    config = load_config(args.config)
    output_dir = (
        resolve_repo_path(args.output_dir) if args.output_dir is not None else None
    )
    if args.limit is not None and output_dir is None:
        raise ValueError(
            "--limit requires --output-dir to avoid overwriting the main cache"
        )
    cache_dir = precompute_structure_cache(
        config, output_dir=output_dir, limit=args.limit
    )
    print(cache_dir)
    print(cache_dir / "manifest.json")


if __name__ == "__main__":
    main()

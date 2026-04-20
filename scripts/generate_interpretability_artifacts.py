from __future__ import annotations

from hybrite.interpretability import run_all_interpretability_analyses


def main() -> None:
    run_all_interpretability_analyses()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI surface
        raise SystemExit(f"Interpretability artifact generation failed: {exc}") from exc

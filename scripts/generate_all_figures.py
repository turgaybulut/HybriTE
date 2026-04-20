from __future__ import annotations

import subprocess
import sys

from hybrite.config import REPO_ROOT


def _run_script(script_name: str, extra_args: list[str] | None = None) -> None:
    script_path = REPO_ROOT / "scripts" / script_name
    command = [sys.executable, str(script_path), *(extra_args or [])]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"All-figure generation stopped at {script_name}")


def main() -> None:
    # Main Results figures.
    _run_script("plots/plot_figure_performance_analysis.py")
    _run_script("plots/plot_figure_interpretability_analysis.py")
    _run_script("plots/plot_figure_per_target_performance.py")
    _run_script("plots/plot_figure_target_correlation.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI surface
        raise SystemExit(f"All-figure generation failed: {exc}") from exc

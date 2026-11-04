"""Run the ORIEON paper experiments in their required order.

Examples:
    python run_all.py                 # complete workflow
    python run_all.py classical       # persistence, ARIMA, Ridge, RF, LightGBM
    python run_all.py deep            # MLP, LSTM, GRU and temporal selection
    python run_all.py dynamic         # dynamic graph, static graph, no sentiment
    python run_all.py figures         # regenerate figures and derived summary
    python run_all.py compare         # compare outputs with paper references
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def require_files(paths: list[Path], stage: str) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(
            f"Cannot run '{stage}'. Missing prerequisite file(s): {', '.join(missing)}\n"
            "Run the earlier stages first, or use 'python run_all.py' for the full workflow."
        )


def execute(script: str, argument: str | None = None) -> None:
    command = [sys.executable, str(ROOT / script)]
    if argument:
        command.append(argument)
    print("\n" + "=" * 78, flush=True)
    print("RUNNING:", " ".join(command), flush=True)
    print("=" * 78, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", "classical", "deep", "dynamic", "static", "nosentiment", "figures", "compare"],
        help="Workflow stage to execute (default: all).",
    )
    args = parser.parse_args()

    require_files(
        [ROOT / "data" / "gold_clean.csv", ROOT / "data" / "news_with_scores.csv"],
        args.stage,
    )

    if args.stage in {"all", "classical"}:
        execute("classical_experiments.py")

    if args.stage in {"all", "deep"}:
        require_files([ROOT / "processed_data.csv", RESULTS / "data_protocol.json"], "deep")
        execute("deep_baselines.py")

    if args.stage in {"all", "dynamic", "static", "nosentiment"}:
        require_files(
            [
                ROOT / "processed_data.csv",
                RESULTS / "data_protocol.json",
                RESULTS / "selected_temporal.json",
            ],
            args.stage,
        )
        dynamic_argument = None if args.stage in {"all", "dynamic"} else args.stage
        execute("dynamic_experiment.py", dynamic_argument)

    if args.stage in {"all", "figures"}:
        require_files([RESULTS / "metrics.csv", ROOT / "filtered_news.csv"], "figures")
        execute("generate_figures.py")

    if args.stage in {"all", "compare"}:
        require_files([RESULTS / "metrics.csv"], "compare")
        execute("compare_results.py")

    print("\nRequested workflow completed successfully.", flush=True)
    print(f"Metrics: {RESULTS / 'metrics.csv'}", flush=True)
    print(f"Figures: {ROOT / 'figures'}", flush=True)


if __name__ == "__main__":
    main()

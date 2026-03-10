from __future__ import annotations

import argparse

from mekubbal.experiment_log import list_experiment_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="List recent experiment runs from the SQLite log.")
    parser.add_argument("--db", default="logs/experiments.db", help="SQLite log path")
    parser.add_argument("--limit", type=int, default=20, help="Maximum runs to show")
    parser.add_argument(
        "--run-type",
        help=(
            "Optional run_type filter "
            "(train/evaluate/paper/retrain_window/walkforward_fold/selection)"
        ),
    )
    parser.add_argument("--symbol", help="Optional symbol filter")
    args = parser.parse_args()

    runs = list_experiment_runs(
        db_path=args.db,
        limit=args.limit,
        run_type=args.run_type,
        symbol=args.symbol,
    )
    if not runs:
        print("No runs found.")
        return

    columns = [
        "id",
        "created_at",
        "run_type",
        "symbol",
        "timesteps",
        "reward",
        "final_equity",
        "baseline_equity",
        "rows_logged",
        "model_path",
    ]
    print(" | ".join(columns))
    for run in runs:
        values = [str(run.get(column, "")) for column in columns]
        print(" | ".join(values))


if __name__ == "__main__":
    main()

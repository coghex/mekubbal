from __future__ import annotations

import argparse

from mekubbal.env import parse_position_levels
from mekubbal.retrain import run_periodic_retraining


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run weekly/monthly retraining using data available up to each cutoff date."
    )
    parser.add_argument("--data", required=True, help="Input OHLCV CSV path")
    parser.add_argument("--models-dir", required=True, help="Output directory for model snapshots")
    parser.add_argument("--report", required=True, help="CSV path for retraining metrics report")
    parser.add_argument("--cadence", choices=["weekly", "monthly"], default="weekly")
    parser.add_argument("--timesteps", type=int, default=10000, help="PPO timesteps per retrain run")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--trade-cost", type=float, default=0.001, help="Per-trade cost")
    parser.add_argument("--risk-penalty", type=float, default=0.0002, help="Quadratic position penalty")
    parser.add_argument("--switch-penalty", type=float, default=0.0001, help="Penalty when action changes")
    parser.add_argument(
        "--position-levels",
        default="-1,-0.5,0,0.5,1",
        help="Comma-separated target positions (example: -1,-0.5,0,0.5,1)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--symbol", help="Optional symbol label for experiment logging")
    parser.add_argument(
        "--log-db",
        default="logs/experiments.db",
        help="SQLite path for experiment logging",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable experiment logging")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=120,
        help="Minimum feature rows required before running a retrain window",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Optional cap on most recent retrain windows to execute",
    )
    args = parser.parse_args()

    metrics = run_periodic_retraining(
        data_path=args.data,
        models_dir=args.models_dir,
        report_path=args.report,
        cadence=args.cadence,
        total_timesteps=args.timesteps,
        train_ratio=args.train_ratio,
        trade_cost=args.trade_cost,
        risk_penalty=args.risk_penalty,
        switch_penalty=args.switch_penalty,
        position_levels=parse_position_levels(args.position_levels),
        seed=args.seed,
        min_feature_rows=args.min_rows,
        max_runs=args.max_runs,
        symbol=args.symbol,
        log_db_path=None if args.no_log else args.log_db,
    )
    print(metrics)


if __name__ == "__main__":
    main()

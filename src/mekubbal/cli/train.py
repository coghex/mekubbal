from __future__ import annotations

import argparse

from mekubbal.env import parse_position_levels
from mekubbal.experiment_log import log_experiment_run
from mekubbal.train import train_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on stock OHLCV CSV.")
    parser.add_argument("--data", required=True, help="Input OHLCV CSV path")
    parser.add_argument("--model", required=True, help="Model output path")
    parser.add_argument("--timesteps", type=int, default=30000, help="PPO training timesteps")
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
    args = parser.parse_args()

    metrics = train_from_csv(
        data_path=args.data,
        model_path=args.model,
        total_timesteps=args.timesteps,
        train_ratio=args.train_ratio,
        trade_cost=args.trade_cost,
        risk_penalty=args.risk_penalty,
        switch_penalty=args.switch_penalty,
        position_levels=parse_position_levels(args.position_levels),
        seed=args.seed,
    )
    if not args.no_log:
        model_path = args.model if str(args.model).endswith(".zip") else f"{args.model}.zip"
        run_id = log_experiment_run(
            db_path=args.log_db,
            run_type="train",
            symbol=args.symbol,
            data_path=args.data,
            model_path=model_path,
            timesteps=args.timesteps,
            train_ratio=args.train_ratio,
            trade_cost=args.trade_cost,
            metrics={
                **metrics,
                "risk_penalty_setting": args.risk_penalty,
                "switch_penalty_setting": args.switch_penalty,
                "position_levels_setting": args.position_levels,
            },
        )
        metrics = {**metrics, "experiment_run_id": run_id}
    print(metrics)


if __name__ == "__main__":
    main()

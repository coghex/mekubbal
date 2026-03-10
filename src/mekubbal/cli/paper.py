from __future__ import annotations

import argparse

from mekubbal.env import parse_position_levels
from mekubbal.experiment_log import log_experiment_run
from mekubbal.paper import run_paper_trading


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic paper-trading replay and log action/PnL rows."
    )
    parser.add_argument("--data", required=True, help="Input OHLCV CSV path")
    parser.add_argument("--model", required=True, help="Saved model path (.zip)")
    parser.add_argument("--output", required=True, help="Paper-trade output CSV path")
    parser.add_argument("--trade-cost", type=float, default=0.001, help="Per-trade cost")
    parser.add_argument("--risk-penalty", type=float, default=0.0002, help="Quadratic position penalty")
    parser.add_argument("--switch-penalty", type=float, default=0.0001, help="Penalty when action changes")
    parser.add_argument(
        "--position-levels",
        default="-1,-0.5,0,0.5,1",
        help="Comma-separated target positions (example: -1,-0.5,0,0.5,1)",
    )
    parser.add_argument("--start-date", help="Optional lower bound date YYYY-MM-DD")
    parser.add_argument("--append", action="store_true", help="Append/resume from existing log")
    parser.add_argument("--symbol", help="Optional symbol label for experiment logging")
    parser.add_argument(
        "--log-db",
        default="logs/experiments.db",
        help="SQLite path for experiment logging",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable experiment logging")
    args = parser.parse_args()

    metrics = run_paper_trading(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        trade_cost=args.trade_cost,
        risk_penalty=args.risk_penalty,
        switch_penalty=args.switch_penalty,
        position_levels=parse_position_levels(args.position_levels),
        start_date=args.start_date,
        append=args.append,
    )
    if not args.no_log:
        run_id = log_experiment_run(
            db_path=args.log_db,
            run_type="paper",
            symbol=args.symbol,
            data_path=args.data,
            model_path=args.model,
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

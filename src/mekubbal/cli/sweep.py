from __future__ import annotations

import argparse

from mekubbal.env import parse_position_levels
from mekubbal.sweep import parse_penalty_grid, run_reward_penalty_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run downside/drawdown penalty sweep using repeated ablation runs."
    )
    parser.add_argument("--data", required=True, help="Input OHLCV CSV path")
    parser.add_argument("--output-dir", required=True, help="Directory for sweep artifacts")
    parser.add_argument("--report", required=True, help="CSV path for ranked sweep outcomes")
    parser.add_argument(
        "--downside-grid",
        default="0,0.005,0.01,0.02",
        help="Comma-separated downside penalty values",
    )
    parser.add_argument(
        "--drawdown-grid",
        default="0,0.02,0.05,0.1",
        help="Comma-separated drawdown penalty values",
    )
    parser.add_argument("--train-window", type=int, default=252, help="Training rows per fold")
    parser.add_argument("--test-window", type=int, default=63, help="Test rows per fold")
    parser.add_argument(
        "--step-window",
        type=int,
        help="Rows to move forward between folds (defaults to test-window)",
    )
    parser.add_argument(
        "--expanding",
        action="store_true",
        help="Use expanding train window instead of rolling window",
    )
    parser.add_argument("--timesteps", type=int, default=10000, help="PPO timesteps per fold")
    parser.add_argument("--trade-cost", type=float, default=0.001, help="Per-trade cost")
    parser.add_argument("--risk-penalty", type=float, default=0.0002, help="Quadratic position penalty")
    parser.add_argument("--switch-penalty", type=float, default=0.0001, help="Penalty when action changes")
    parser.add_argument("--downside-window", type=int, default=20, help="Downside volatility rolling window")
    parser.add_argument(
        "--regime-tie-break-tolerance",
        type=float,
        default=0.01,
        help=(
            "Treat v2-v1 delta values within this tolerance as near-equal and "
            "rank them by turbulent-risk diagnostics first (set 0 to disable)."
        ),
    )
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

    metrics = run_reward_penalty_sweep(
        data_path=args.data,
        output_dir=args.output_dir,
        sweep_report_path=args.report,
        downside_penalties=parse_penalty_grid(args.downside_grid, label="downside_grid"),
        drawdown_penalties=parse_penalty_grid(args.drawdown_grid, label="drawdown_grid"),
        train_window=args.train_window,
        test_window=args.test_window,
        step_window=args.step_window,
        expanding=args.expanding,
        total_timesteps=args.timesteps,
        trade_cost=args.trade_cost,
        risk_penalty=args.risk_penalty,
        switch_penalty=args.switch_penalty,
        downside_window=args.downside_window,
        position_levels=parse_position_levels(args.position_levels),
        seed=args.seed,
        symbol=args.symbol,
        log_db_path=None if args.no_log else args.log_db,
        regime_tie_break_tolerance=args.regime_tie_break_tolerance,
    )
    print(metrics)


if __name__ == "__main__":
    main()

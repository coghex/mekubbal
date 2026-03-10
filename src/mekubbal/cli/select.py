from __future__ import annotations

import argparse

from mekubbal.experiment_log import log_experiment_run
from mekubbal.selection import run_model_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote active model when recent walk-forward folds beat baseline."
    )
    parser.add_argument("--report", required=True, help="Walk-forward report CSV path")
    parser.add_argument(
        "--state",
        default="models/current_model.json",
        help="JSON state file storing active model pointer",
    )
    parser.add_argument("--lookback", type=int, default=3, help="Recent folds to evaluate")
    parser.add_argument("--min-gap", type=float, default=0.0, help="Required policy-baseline equity gap")
    parser.add_argument(
        "--allow-average-rule",
        action="store_true",
        help="Use average recent gap > min-gap instead of requiring all recent folds",
    )
    parser.add_argument(
        "--min-turbulent-steps",
        type=float,
        default=0.0,
        help="Minimum turbulent step count across recent folds required for regime gate",
    )
    parser.add_argument(
        "--min-turbulent-reward-mean",
        type=float,
        help="Minimum mean diag_turbulent_reward_mean across recent folds",
    )
    parser.add_argument(
        "--min-turbulent-win-rate",
        type=float,
        help="Minimum mean diag_turbulent_win_rate across recent folds",
    )
    parser.add_argument(
        "--min-turbulent-equity-factor",
        type=float,
        help="Minimum mean diag_turbulent_equity_factor across recent folds",
    )
    parser.add_argument(
        "--max-turbulent-drawdown",
        type=float,
        help="Maximum mean diag_turbulent_max_drawdown across recent folds",
    )
    parser.add_argument("--symbol", help="Optional symbol label for experiment logging")
    parser.add_argument("--log-db", default="logs/experiments.db", help="SQLite path for experiment logging")
    parser.add_argument("--no-log", action="store_true", help="Disable experiment logging")
    args = parser.parse_args()

    summary = run_model_selection(
        report_path=args.report,
        state_path=args.state,
        lookback_folds=args.lookback,
        min_gap=args.min_gap,
        require_all_recent=not args.allow_average_rule,
        min_turbulent_step_count=args.min_turbulent_steps,
        min_turbulent_reward_mean=args.min_turbulent_reward_mean,
        min_turbulent_win_rate=args.min_turbulent_win_rate,
        min_turbulent_equity_factor=args.min_turbulent_equity_factor,
        max_turbulent_max_drawdown=args.max_turbulent_drawdown,
    )
    if not args.no_log:
        run_id = log_experiment_run(
            db_path=args.log_db,
            run_type="selection",
            symbol=args.symbol,
            data_path=args.report,
            model_path=summary["active_model_path"],
            metrics=summary,
        )
        summary = {**summary, "experiment_run_id": run_id}
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from mekubbal.profile_monitor import run_profile_monitor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate active-profile health snapshot/history and drift alerts."
    )
    parser.add_argument(
        "--profile-symbol-summary",
        required=True,
        help="Path to profile_symbol_summary.csv from profile-matrix output",
    )
    parser.add_argument(
        "--selection-state",
        required=True,
        help="Path to profile_selection_state.json",
    )
    parser.add_argument(
        "--health-snapshot",
        default="logs/profile_matrix/reports/active_profile_health.csv",
        help="Output CSV for current active-profile health snapshot",
    )
    parser.add_argument(
        "--health-history",
        default="logs/profile_matrix/reports/active_profile_health_history.csv",
        help="Output CSV (append) for active-profile health history",
    )
    parser.add_argument(
        "--drift-alerts-csv",
        default="logs/profile_matrix/reports/profile_drift_alerts.csv",
        help="Output CSV for drift/regression alerts",
    )
    parser.add_argument(
        "--drift-alerts-html",
        default="logs/profile_matrix/reports/profile_drift_alerts.html",
        help="Output HTML for drift/regression alerts",
    )
    parser.add_argument("--lookback-runs", type=int, default=3, help="Prior runs to compare against")
    parser.add_argument(
        "--max-gap-drop",
        type=float,
        default=0.03,
        help="Alert if active gap drops more than this value vs prior lookback average",
    )
    parser.add_argument(
        "--max-rank-worsening",
        type=float,
        default=0.75,
        help="Alert if active rank worsens more than this value vs prior lookback average",
    )
    parser.add_argument(
        "--min-active-minus-base-gap",
        type=float,
        default=-0.01,
        help="Alert if active profile underperforms base by more than this threshold",
    )
    args = parser.parse_args()
    summary = run_profile_monitor(
        profile_symbol_summary_path=args.profile_symbol_summary,
        selection_state_path=args.selection_state,
        health_snapshot_path=args.health_snapshot,
        health_history_path=args.health_history,
        drift_alerts_csv_path=args.drift_alerts_csv,
        drift_alerts_html_path=args.drift_alerts_html,
        lookback_runs=args.lookback_runs,
        max_gap_drop=args.max_gap_drop,
        max_rank_worsening=args.max_rank_worsening,
        min_active_minus_base_gap=args.min_active_minus_base_gap,
    )
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from mekubbal.profile_rollback import run_profile_rollback


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend or apply active-profile rollback when drift alerts persist "
            "across consecutive schedule runs."
        )
    )
    parser.add_argument(
        "--selection-state",
        required=True,
        help="Path to profile_selection_state.json",
    )
    parser.add_argument(
        "--health-history",
        required=True,
        help="Path to active_profile_health_history.csv",
    )
    parser.add_argument(
        "--rollback-state",
        default="logs/profile_matrix/reports/profile_rollback_state.json",
        help="Output JSON for rollback recommendations/actions",
    )
    parser.add_argument("--lookback-runs", type=int, default=3, help="Monitor lookback runs")
    parser.add_argument("--max-gap-drop", type=float, default=0.03, help="Monitor max gap drop")
    parser.add_argument(
        "--max-rank-worsening",
        type=float,
        default=0.75,
        help="Monitor max rank worsening",
    )
    parser.add_argument(
        "--min-active-minus-base-gap",
        type=float,
        default=-0.01,
        help="Monitor minimum active-minus-base gap threshold",
    )
    parser.add_argument(
        "--min-consecutive-alert-runs",
        type=int,
        default=2,
        help="Consecutive alert runs needed before rollback recommendation/action",
    )
    parser.add_argument(
        "--rollback-on-drift-alerts",
        action="store_true",
        default=True,
        help="Enable drift-alert based rollback trigger (default: enabled)",
    )
    parser.add_argument(
        "--no-rollback-on-drift-alerts",
        dest="rollback_on_drift_alerts",
        action="store_false",
        help="Disable drift-alert based rollback trigger",
    )
    parser.add_argument(
        "--rollback-on-ensemble-events",
        action="store_true",
        help="Enable rollback trigger based on consecutive ensemble ops events",
    )
    parser.add_argument(
        "--ensemble-alerts-history",
        help="Path to profile_ensemble_alerts_history.csv (required with --rollback-on-ensemble-events)",
    )
    parser.add_argument(
        "--min-consecutive-ensemble-event-runs",
        type=int,
        default=2,
        help="Consecutive ensemble-event runs needed before rollback recommendation/action",
    )
    parser.add_argument(
        "--rollback-profile",
        help="Profile name to roll back to (default uses selection base profile)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply rollback directly to selection state active_profiles",
    )
    parser.add_argument(
        "--run-timestamp-utc",
        help="Optional timestamp to treat as latest run for rollback decision",
    )
    args = parser.parse_args()
    summary = run_profile_rollback(
        selection_state_path=args.selection_state,
        health_history_path=args.health_history,
        rollback_state_path=args.rollback_state,
        lookback_runs=args.lookback_runs,
        max_gap_drop=args.max_gap_drop,
        max_rank_worsening=args.max_rank_worsening,
        min_active_minus_base_gap=args.min_active_minus_base_gap,
        min_consecutive_alert_runs=args.min_consecutive_alert_runs,
        rollback_on_drift_alerts=args.rollback_on_drift_alerts,
        rollback_on_ensemble_events=args.rollback_on_ensemble_events,
        ensemble_alerts_history_path=args.ensemble_alerts_history,
        min_consecutive_ensemble_event_runs=args.min_consecutive_ensemble_event_runs,
        rollback_profile=args.rollback_profile,
        apply_rollback=args.apply,
        run_timestamp_utc=args.run_timestamp_utc,
    )
    print(summary)


if __name__ == "__main__":
    main()

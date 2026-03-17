from __future__ import annotations

import argparse
import json

from mekubbal.profile_backfill import run_profile_backfill


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild profile-schedule history by replaying the schedule over historical "
            "dates using raw ticker CSV snapshots."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/profile-schedule.toml",
        help="Path to profile schedule TOML config",
    )
    parser.add_argument(
        "--start-date",
        help="Optional inclusive replay start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        help="Optional inclusive replay end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Replay every Nth eligible trading date (default: 1)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Replay only the most recent N eligible runs after filtering",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete the configured output root before replaying history",
    )
    args = parser.parse_args()
    summary = run_profile_backfill(
        args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        every=args.every,
        max_runs=args.max_runs,
        reset_output=args.reset_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

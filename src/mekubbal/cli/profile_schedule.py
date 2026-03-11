from __future__ import annotations

import argparse

from mekubbal.profile_schedule import run_profile_schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run profile matrix workflow with monitoring, optional shadow gating, and drift alerts."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/profile-schedule.toml",
        help="Path to profile schedule TOML config",
    )
    args = parser.parse_args()
    summary = run_profile_schedule(args.config)
    print(summary)


if __name__ == "__main__":
    main()

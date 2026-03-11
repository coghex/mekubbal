from __future__ import annotations

import argparse

from mekubbal.profile_runner import run_profile_runner


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple profile control configs for one symbol and "
            "produce pairwise significance comparison outputs."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/profile-runner.toml",
        help="Path to profile-runner TOML config",
    )
    args = parser.parse_args()
    summary = run_profile_runner(args.config)
    print(summary)


if __name__ == "__main__":
    main()

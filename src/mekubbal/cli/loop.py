from __future__ import annotations

import argparse

from mekubbal.initial_loop import run_initial_training_loop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run initial training loop from TOML config (optional refresh/train/paper/logging)."
    )
    parser.add_argument(
        "--config",
        default="configs/initial-loop.toml",
        help="Path to loop TOML config",
    )
    args = parser.parse_args()

    metrics = run_initial_training_loop(args.config)
    print(metrics)


if __name__ == "__main__":
    main()


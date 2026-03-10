from __future__ import annotations

import argparse

from mekubbal.control import run_research_control


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run orchestrated research control workflow "
            "(walk-forward, ablation, sweep, selection, report) from TOML config."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/research-control.toml",
        help="Path to research control TOML config",
    )
    args = parser.parse_args()
    metrics = run_research_control(args.config)
    print(metrics)


if __name__ == "__main__":
    main()

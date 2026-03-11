from __future__ import annotations

import argparse

from mekubbal.multi_symbol import parse_symbols
from mekubbal.profile_matrix import run_profile_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-symbol x multi-profile matrix by reusing profile-runner configs "
            "and build cross-symbol aggregate leaderboards."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/profile-matrix.toml",
        help="Path to profile-matrix TOML config",
    )
    parser.add_argument(
        "--symbols",
        help="Optional comma-separated symbol override (example: AAPL,MSFT,NVDA)",
    )
    args = parser.parse_args()
    symbols_override = parse_symbols(args.symbols) if args.symbols else None
    summary = run_profile_matrix(args.config, symbols_override=symbols_override)
    print(summary)


if __name__ == "__main__":
    main()

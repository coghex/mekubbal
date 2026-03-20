from __future__ import annotations

import argparse

from mekubbal.reporting import render_ticker_tabs_report


def _parse_mapping(values: list[str], *, uppercase_key: bool, field_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Each --{field_name} value must use NAME=path format.")
        ticker, path = raw.split("=", 1)
        ticker = ticker.strip()
        path = path.strip()
        if not ticker or not path:
            raise ValueError(f"Each --{field_name} value must include both name and path.")
        if uppercase_key:
            ticker = ticker.upper()
        mapping[ticker] = path
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tabbed HTML dashboard linking per-ticker reports.")
    parser.add_argument("--output", required=True, help="Output HTML path for tabs dashboard")
    parser.add_argument(
        "--tab",
        action="append",
        default=[],
        help="Ticker mapping in TICKER=report_path form (repeat for multiple tickers)",
    )
    parser.add_argument(
        "--leaderboard",
        action="append",
        default=[],
        help="Leaderboard mapping in NAME=report_path form (repeat for multiple leaderboards)",
    )
    parser.add_argument("--title", default="Mekubbal Multi-Ticker Dashboard", help="Dashboard title")
    args = parser.parse_args()
    tabs = _parse_mapping(args.tab, uppercase_key=True, field_name="tab")
    leaderboards = _parse_mapping(args.leaderboard, uppercase_key=False, field_name="leaderboard")
    if not tabs and not leaderboards:
        raise ValueError("At least one --tab or --leaderboard value is required.")

    report = render_ticker_tabs_report(
        output_path=args.output,
        ticker_reports=tabs,
        leaderboard_reports=leaderboards or None,
        title=args.title,
    )
    print({"output": str(report)})


if __name__ == "__main__":
    main()

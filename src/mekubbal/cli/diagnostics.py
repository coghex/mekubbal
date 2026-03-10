from __future__ import annotations

import argparse

import pandas as pd

from mekubbal.diagnostics import diagnostics_from_paper_log, summarize_walkforward_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize diagnostics from paper logs or walk-forward reports.")
    parser.add_argument("--input", required=True, help="CSV path (paper log or walk-forward report)")
    parser.add_argument("--mode", choices=["paper", "walkforward"], required=True)
    args = parser.parse_args()

    if args.mode == "paper":
        metrics = diagnostics_from_paper_log(pd.read_csv(args.input))
    else:
        metrics = summarize_walkforward_report(args.input)
    print(metrics)


if __name__ == "__main__":
    main()


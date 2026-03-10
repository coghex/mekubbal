from __future__ import annotations

import argparse

from mekubbal.profile_compare import compare_profile_reports, parse_profile_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired significance tests between profile walk-forward reports."
    )
    parser.add_argument(
        "--profile-report",
        action="append",
        default=[],
        help="Profile mapping in NAME=walkforward_report.csv form (repeatable)",
    )
    parser.add_argument(
        "--output-csv",
        default="logs/profile_compare/pairwise_significance.csv",
        help="Output CSV path for pairwise significance results",
    )
    parser.add_argument(
        "--output-html",
        default="logs/profile_compare/pairwise_significance.html",
        help="Output HTML path for pairwise significance results",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Bootstrap confidence level in [0.5, 1.0) (default: 0.95)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap sample count (default: 2000)",
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=20000,
        help="Permutation sample count (default: 20000, exact for <=16 folds)",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument(
        "--title",
        default="Profile Pairwise Significance",
        help="HTML report title",
    )
    args = parser.parse_args()
    reports = parse_profile_reports(args.profile_report)
    summary = compare_profile_reports(
        profile_reports=reports,
        output_csv_path=args.output_csv,
        output_html_path=args.output_html,
        confidence_level=args.confidence_level,
        n_bootstrap=args.bootstrap_samples,
        n_permutation=args.permutation_samples,
        seed=args.seed,
        title=args.title,
    )
    print(summary)


if __name__ == "__main__":
    main()

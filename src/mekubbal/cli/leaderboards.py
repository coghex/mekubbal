from __future__ import annotations

import argparse

from mekubbal.leaderboards import generate_confidence_leaderboards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-symbol leaderboards with bootstrap confidence metrics."
    )
    parser.add_argument(
        "--reports-root",
        required=True,
        help="Directory containing multi_symbol_summary.csv and walkforward_<symbol>.csv files",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for leaderboard CSV/HTML files (default: reports root)",
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
        help="Paired permutation sample count (default: 20000, exact for <=16 folds)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Bootstrap RNG seed")
    args = parser.parse_args()

    summary = generate_confidence_leaderboards(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        confidence_level=args.confidence_level,
        n_bootstrap=args.bootstrap_samples,
        n_permutation=args.permutation_samples,
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()

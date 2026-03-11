from __future__ import annotations

import argparse

from mekubbal.profile_selection import run_profile_promotion


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote candidate profiles to active profiles per symbol using profile matrix outputs."
        )
    )
    parser.add_argument(
        "--profile-symbol-summary",
        default="logs/profile_matrix/reports/profile_symbol_summary.csv",
        help="Profile symbol summary CSV from profile-matrix output",
    )
    parser.add_argument(
        "--state",
        default="logs/profile_matrix/reports/profile_selection_state.json",
        help="Output JSON state path for active profile decisions",
    )
    parser.add_argument("--base-profile", default="base", help="Base profile name")
    parser.add_argument("--candidate-profile", default="candidate", help="Candidate profile name")
    parser.add_argument(
        "--min-candidate-gap-vs-base",
        type=float,
        default=0.0,
        help="Minimum candidate minus base average equity gap required for promotion",
    )
    parser.add_argument(
        "--max-candidate-rank",
        type=int,
        default=1,
        help="Maximum within-symbol candidate rank allowed for promotion (1=best)",
    )
    parser.add_argument(
        "--require-candidate-significant",
        action="store_true",
        help="Require candidate to be significantly better than base in per-symbol pairwise test",
    )
    parser.add_argument(
        "--allow-base-significant-better",
        action="store_true",
        help="Allow promotion even if base is significantly better (disabled by default).",
    )
    parser.add_argument(
        "--no-prefer-previous-active",
        action="store_true",
        help="Do not keep previous active profile when promotion fails.",
    )
    parser.add_argument(
        "--fallback-profile",
        default="base",
        help="Fallback profile when promotion fails and previous active is unavailable",
    )
    args = parser.parse_args()
    summary = run_profile_promotion(
        profile_symbol_summary_path=args.profile_symbol_summary,
        state_path=args.state,
        base_profile=args.base_profile,
        candidate_profile=args.candidate_profile,
        min_candidate_gap_vs_base=args.min_candidate_gap_vs_base,
        max_candidate_rank=args.max_candidate_rank,
        require_candidate_significant=args.require_candidate_significant,
        forbid_base_significant_better=not args.allow_base_significant_better,
        prefer_previous_active=not args.no_prefer_previous_active,
        fallback_profile=args.fallback_profile,
    )
    print(summary)


if __name__ == "__main__":
    main()

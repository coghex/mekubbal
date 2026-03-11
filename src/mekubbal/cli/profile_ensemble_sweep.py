from __future__ import annotations

import argparse

from mekubbal.profile_ensemble_sweep import run_profile_ensemble_sweep


def _parse_float_grid(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Float grid must include at least one value.")
    return [float(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep v3 ensemble parameters and rank combinations by selected-gap, "
            "confidence, and stability tradeoffs."
        )
    )
    parser.add_argument(
        "--profile-symbol-summary",
        required=True,
        help="Path to profile_symbol_summary.csv",
    )
    parser.add_argument(
        "--selection-state",
        required=True,
        help="Path to profile_selection_state.json",
    )
    parser.add_argument(
        "--health-history",
        required=True,
        help="Path to active_profile_health_history.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="logs/profile_matrix/reports/profile_ensemble_sweep.csv",
        help="Output CSV path for ranked sweep results",
    )
    parser.add_argument(
        "--output-html",
        default="logs/profile_matrix/reports/profile_ensemble_sweep.html",
        help="Output HTML path for ranked sweep results",
    )
    parser.add_argument(
        "--recommendation-json",
        default="logs/profile_matrix/reports/profile_ensemble_recommendation.json",
        help="Output JSON path with top-ranked ensemble_v3 recommendation block",
    )
    parser.add_argument("--base-profile", default="base", help="Base profile name")
    parser.add_argument("--candidate-profile", default="candidate", help="Candidate profile name")
    parser.add_argument("--lookback-runs", type=int, default=3, help="Regime lookback runs")
    parser.add_argument(
        "--min-regime-confidence-grid",
        default="0.5,0.6,0.7",
        help="Comma-separated regime confidence thresholds",
    )
    parser.add_argument(
        "--rank-weight-grid",
        default="0.45,0.55,0.65",
        help="Comma-separated rank score weights",
    )
    parser.add_argument(
        "--gap-weight-grid",
        default="0.35,0.45,0.55",
        help="Comma-separated gap score weights",
    )
    parser.add_argument(
        "--significance-bonus-grid",
        default="0.0,0.1",
        help="Comma-separated significance bonus values",
    )
    parser.add_argument(
        "--candidate-weight-grid",
        default="1.0,1.15,1.3",
        help="Comma-separated candidate profile weight values",
    )
    parser.add_argument(
        "--trending-candidate-multiplier-grid",
        default="1.0,1.1,1.2",
        help="Comma-separated trending-regime candidate multipliers",
    )
    parser.add_argument(
        "--high-vol-candidate-multiplier-grid",
        default="0.75,0.85,1.0",
        help="Comma-separated high-vol-regime candidate multipliers",
    )
    parser.add_argument(
        "--high-vol-gap-std-threshold",
        type=float,
        default=0.03,
        help="High-vol threshold for active gap standard deviation",
    )
    parser.add_argument(
        "--high-vol-rank-std-threshold",
        type=float,
        default=0.75,
        help="High-vol threshold for active rank standard deviation",
    )
    parser.add_argument(
        "--trending-min-gap-improvement",
        type=float,
        default=0.01,
        help="Minimum gap improvement to classify trending regime",
    )
    parser.add_argument(
        "--trending-min-rank-improvement",
        type=float,
        default=0.25,
        help="Minimum rank improvement to classify trending regime",
    )
    parser.add_argument(
        "--min-history-runs",
        type=int,
        default=5,
        help="Minimum distinct run timestamps required to accept recommendation",
    )
    parser.add_argument(
        "--min-history-runs-per-symbol",
        type=int,
        default=5,
        help="Minimum run timestamps per symbol required to accept recommendation",
    )
    parser.add_argument("--fallback-profile", default="base", help="Fallback profile name")
    parser.add_argument("--title", default="Profile Ensemble Sweep", help="HTML report title")
    args = parser.parse_args()

    summary = run_profile_ensemble_sweep(
        profile_symbol_summary_path=args.profile_symbol_summary,
        selection_state_path=args.selection_state,
        health_history_path=args.health_history,
        output_csv_path=args.output_csv,
        output_html_path=args.output_html,
        recommendation_json_path=args.recommendation_json,
        base_profile=args.base_profile,
        candidate_profile=args.candidate_profile,
        lookback_runs=args.lookback_runs,
        min_regime_confidence_grid=_parse_float_grid(args.min_regime_confidence_grid),
        rank_weight_grid=_parse_float_grid(args.rank_weight_grid),
        gap_weight_grid=_parse_float_grid(args.gap_weight_grid),
        significance_bonus_grid=_parse_float_grid(args.significance_bonus_grid),
        candidate_weight_grid=_parse_float_grid(args.candidate_weight_grid),
        trending_candidate_multiplier_grid=_parse_float_grid(
            args.trending_candidate_multiplier_grid
        ),
        high_vol_candidate_multiplier_grid=_parse_float_grid(args.high_vol_candidate_multiplier_grid),
        high_vol_gap_std_threshold=args.high_vol_gap_std_threshold,
        high_vol_rank_std_threshold=args.high_vol_rank_std_threshold,
        trending_min_gap_improvement=args.trending_min_gap_improvement,
        trending_min_rank_improvement=args.trending_min_rank_improvement,
        min_history_runs=args.min_history_runs,
        min_history_runs_per_symbol=args.min_history_runs_per_symbol,
        fallback_profile=args.fallback_profile,
        title=args.title,
    )
    print(summary)


if __name__ == "__main__":
    main()

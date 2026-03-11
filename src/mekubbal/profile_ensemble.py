from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _clamp(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def classify_symbol_regimes(
    health_history: pd.DataFrame,
    *,
    lookback_runs: int,
    high_vol_gap_std_threshold: float,
    high_vol_rank_std_threshold: float,
    trending_min_gap_improvement: float,
    trending_min_rank_improvement: float,
) -> pd.DataFrame:
    if int(lookback_runs) < 1:
        raise ValueError("lookback_runs must be >= 1.")
    if float(high_vol_gap_std_threshold) <= 0:
        raise ValueError("high_vol_gap_std_threshold must be > 0.")
    if float(high_vol_rank_std_threshold) <= 0:
        raise ValueError("high_vol_rank_std_threshold must be > 0.")
    if float(trending_min_gap_improvement) < 0:
        raise ValueError("trending_min_gap_improvement must be >= 0.")
    if float(trending_min_rank_improvement) < 0:
        raise ValueError("trending_min_rank_improvement must be >= 0.")

    required = {"symbol", "run_timestamp_utc", "active_gap", "active_rank"}
    missing = sorted(required - set(health_history.columns))
    if missing:
        raise ValueError(f"Health history missing required columns: {missing}")

    frame = health_history.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["active_gap"] = pd.to_numeric(frame["active_gap"], errors="coerce")
    frame["active_rank"] = pd.to_numeric(frame["active_rank"], errors="coerce")
    frame = frame.dropna(subset=["active_gap", "active_rank"])

    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol"):
        ordered = group.sort_values("run_timestamp_utc").tail(int(lookback_runs)).reset_index(drop=True)
        observations = int(len(ordered))
        if observations < 2:
            rows.append(
                {
                    "symbol": symbol,
                    "regime": "unknown",
                    "regime_confidence": 0.0,
                    "gap_std": 0.0,
                    "rank_std": 0.0,
                    "gap_improvement": 0.0,
                    "rank_improvement": 0.0,
                    "observations": observations,
                }
            )
            continue

        gap_std = float(ordered["active_gap"].std(ddof=0))
        rank_std = float(ordered["active_rank"].std(ddof=0))
        latest = ordered.iloc[-1]
        baseline = ordered.iloc[:-1]
        baseline_gap = float(baseline["active_gap"].mean())
        baseline_rank = float(baseline["active_rank"].mean())
        gap_improvement = float(latest["active_gap"] - baseline_gap)
        rank_improvement = float(baseline_rank - float(latest["active_rank"]))

        if gap_std >= float(high_vol_gap_std_threshold) or rank_std >= float(high_vol_rank_std_threshold):
            regime = "high_vol"
            volatility_ratio = max(
                gap_std / float(high_vol_gap_std_threshold),
                rank_std / float(high_vol_rank_std_threshold),
            )
            confidence = _clamp(volatility_ratio / 2.0, low=0.0, high=1.0)
        elif gap_improvement >= float(trending_min_gap_improvement) and rank_improvement >= float(
            trending_min_rank_improvement
        ):
            regime = "trending"
            gap_ratio = 1.0 if float(trending_min_gap_improvement) == 0 else (
                gap_improvement / (float(trending_min_gap_improvement) * 2.0)
            )
            rank_ratio = 1.0 if float(trending_min_rank_improvement) == 0 else (
                rank_improvement / (float(trending_min_rank_improvement) * 2.0)
            )
            confidence = _clamp(max(gap_ratio, rank_ratio), low=0.0, high=1.0)
        else:
            regime = "stable"
            instability = max(
                gap_std / float(high_vol_gap_std_threshold),
                rank_std / float(high_vol_rank_std_threshold),
            )
            confidence = _clamp(1.0 - instability, low=0.0, high=1.0)

        rows.append(
            {
                "symbol": symbol,
                "regime": regime,
                "regime_confidence": confidence,
                "gap_std": gap_std,
                "rank_std": rank_std,
                "gap_improvement": gap_improvement,
                "rank_improvement": rank_improvement,
                "observations": observations,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "regime",
                "regime_confidence",
                "gap_std",
                "rank_std",
                "gap_improvement",
                "rank_improvement",
                "observations",
            ]
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def _load_pairwise_significance(pairwise_csv_path: str | Path | None) -> dict[tuple[str, str], bool]:
    if pairwise_csv_path is None:
        return {}
    path = Path(pairwise_csv_path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    required = {
        "profile_a",
        "profile_b",
        "profile_a_better_significant",
        "profile_b_better_significant",
    }
    if frame.empty or not required.issubset(frame.columns):
        return {}
    lookup: dict[tuple[str, str], bool] = {}
    for _, row in frame.iterrows():
        profile_a = str(row["profile_a"])
        profile_b = str(row["profile_b"])
        if bool(row["profile_a_better_significant"]):
            lookup[(profile_a, profile_b)] = True
        if bool(row["profile_b_better_significant"]):
            lookup[(profile_b, profile_a)] = True
    return lookup


def _normalized_gap_scores(group: pd.DataFrame) -> dict[str, float]:
    by_profile = {
        str(row["profile"]): float(row["avg_equity_gap"])
        for _, row in group.iterrows()
    }
    min_gap = min(by_profile.values())
    max_gap = max(by_profile.values())
    if max_gap - min_gap <= 1e-12:
        return {profile: 0.5 for profile in by_profile}
    return {
        profile: (gap - min_gap) / (max_gap - min_gap)
        for profile, gap in by_profile.items()
    }


def compute_regime_gated_ensemble(
    symbol_summary: pd.DataFrame,
    selection_state: dict[str, Any],
    health_history: pd.DataFrame,
    *,
    lookback_runs: int,
    min_regime_confidence: float,
    rank_weight: float,
    gap_weight: float,
    significance_bonus: float,
    fallback_profile: str,
    profile_weights: dict[str, Any],
    regime_multipliers: dict[str, Any],
    high_vol_gap_std_threshold: float,
    high_vol_rank_std_threshold: float,
    trending_min_gap_improvement: float,
    trending_min_rank_improvement: float,
) -> dict[str, Any]:
    if float(min_regime_confidence) < 0 or float(min_regime_confidence) > 1:
        raise ValueError("min_regime_confidence must be in [0, 1].")
    if float(rank_weight) < 0 or float(gap_weight) < 0:
        raise ValueError("rank_weight and gap_weight must be >= 0.")
    if float(rank_weight) + float(gap_weight) <= 0:
        raise ValueError("rank_weight and gap_weight cannot both be zero.")
    if float(significance_bonus) < 0:
        raise ValueError("significance_bonus must be >= 0.")

    required = {"symbol", "profile", "symbol_rank", "avg_equity_gap"}
    missing = sorted(required - set(symbol_summary.columns))
    if missing:
        raise ValueError(f"Profile symbol summary missing required columns: {missing}")
    if symbol_summary.empty:
        raise ValueError("Profile symbol summary is empty.")

    regimes = classify_symbol_regimes(
        health_history,
        lookback_runs=int(lookback_runs),
        high_vol_gap_std_threshold=float(high_vol_gap_std_threshold),
        high_vol_rank_std_threshold=float(high_vol_rank_std_threshold),
        trending_min_gap_improvement=float(trending_min_gap_improvement),
        trending_min_rank_improvement=float(trending_min_rank_improvement),
    )
    regime_by_symbol = {
        str(row["symbol"]).upper(): {
            "regime": str(row["regime"]),
            "regime_confidence": float(row["regime_confidence"]),
            "gap_std": float(row["gap_std"]),
            "rank_std": float(row["rank_std"]),
            "gap_improvement": float(row["gap_improvement"]),
            "rank_improvement": float(row["rank_improvement"]),
            "observations": int(row["observations"]),
        }
        for _, row in regimes.iterrows()
    }

    active_profiles = selection_state.get("active_profiles", {})
    if not isinstance(active_profiles, dict):
        raise ValueError("profile_selection_state.active_profiles must be an object.")

    frame = symbol_summary.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["symbol_rank"] = pd.to_numeric(frame["symbol_rank"], errors="coerce")
    frame["avg_equity_gap"] = pd.to_numeric(frame["avg_equity_gap"], errors="coerce")
    frame = frame.dropna(subset=["symbol_rank", "avg_equity_gap"])
    if frame.empty:
        raise ValueError("No valid symbol rows found in profile summary.")

    decision_rows: list[dict[str, Any]] = []
    ensembled_profiles: dict[str, str] = {}
    for symbol, group in frame.groupby("symbol"):
        ranked = group.sort_values("symbol_rank").reset_index(drop=True)
        available_profiles = [str(value) for value in ranked["profile"].tolist()]
        selected_profile = str(active_profiles.get(symbol) or "")
        if selected_profile not in available_profiles:
            selected_profile = available_profiles[0]
        fallback = str(fallback_profile)
        if fallback not in available_profiles:
            fallback = available_profiles[0]

        pairwise_csv = None
        if "symbol_pairwise_csv_path" in ranked.columns:
            pairwise_rows = [
                str(value)
                for value in ranked["symbol_pairwise_csv_path"].tolist()
                if isinstance(value, str) and value.strip()
            ]
            pairwise_csv = pairwise_rows[0] if pairwise_rows else None
        significant_lookup = _load_pairwise_significance(pairwise_csv)
        gap_scores = _normalized_gap_scores(ranked)

        profile_scores: dict[str, float] = {}
        score_bits: list[str] = []
        regime_meta = regime_by_symbol.get(
            symbol,
            {
                "regime": "unknown",
                "regime_confidence": 0.0,
                "gap_std": 0.0,
                "rank_std": 0.0,
                "gap_improvement": 0.0,
                "rank_improvement": 0.0,
                "observations": 0,
            },
        )
        regime = str(regime_meta["regime"])
        regime_confidence = float(regime_meta["regime_confidence"])

        for _, row in ranked.iterrows():
            profile = str(row["profile"])
            rank = float(row["symbol_rank"])
            rank_score = 1.0 / rank
            gap_score = float(gap_scores[profile])
            base_score = float(rank_weight) * rank_score + float(gap_weight) * gap_score
            if significant_lookup.get((profile, fallback), False):
                base_score += float(significance_bonus)

            profile_weight = float(profile_weights.get(profile, 1.0))
            by_regime = regime_multipliers.get(regime, {})
            regime_multiplier = (
                float(by_regime.get(profile, 1.0))
                if isinstance(by_regime, dict)
                else 1.0
            )
            final_score = base_score * profile_weight * regime_multiplier
            profile_scores[profile] = final_score
            score_bits.append(f"{profile}:{final_score:.4f}")

        sorted_scores = sorted(
            profile_scores.items(),
            key=lambda item: (item[1], -available_profiles.index(item[0])),
            reverse=True,
        )
        winner, winner_score = sorted_scores[0]
        second_score = float(sorted_scores[1][1]) if len(sorted_scores) > 1 else 0.0
        total_positive = sum(max(value, 0.0) for value in profile_scores.values())
        if total_positive > 0:
            ensemble_confidence = max(winner_score, 0.0) / total_positive
        else:
            ensemble_confidence = 0.0

        gated = regime_confidence < float(min_regime_confidence)
        if gated:
            ensemble_profile = selected_profile
            decision_reason = "regime_confidence_below_threshold"
        else:
            ensemble_profile = winner
            decision_reason = "ensemble_scoring"
        ensembled_profiles[symbol] = ensemble_profile

        decision_rows.append(
            {
                "symbol": symbol,
                "regime": regime,
                "regime_confidence": regime_confidence,
                "selected_profile": selected_profile,
                "ensemble_profile": ensemble_profile,
                "ensemble_confidence": float(_clamp(ensemble_confidence, low=0.0, high=1.0)),
                "winner_profile": winner,
                "winner_score": float(winner_score),
                "runner_up_score": float(second_score),
                "score_breakdown": ";".join(score_bits),
                "gated_by_regime": bool(gated),
                "decision_reason": decision_reason,
                "observations": int(regime_meta["observations"]),
                "pairwise_csv_path": pairwise_csv,
            }
        )

    decisions = pd.DataFrame(decision_rows).sort_values("symbol").reset_index(drop=True)
    return {
        "decisions": decisions,
        "ensembled_profiles": ensembled_profiles,
        "regimes": regime_by_symbol,
    }

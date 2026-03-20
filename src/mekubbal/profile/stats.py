from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from mekubbal.statistics import bootstrap_mean_confidence, paired_permutation_stats

def _pairwise_profile_rows(
    profile_symbol_gaps: dict[str, dict[str, float]],
    *,
    confidence_level: float,
    n_bootstrap: int,
    n_permutation: int,
    seed: int,
) -> pd.DataFrame:
    alpha = 1.0 - float(confidence_level)
    rows: list[dict[str, Any]] = []
    profiles = sorted(profile_symbol_gaps)
    for idx, (profile_a, profile_b) in enumerate(combinations(profiles, 2)):
        symbols = sorted(set(profile_symbol_gaps[profile_a]).intersection(profile_symbol_gaps[profile_b]))
        if not symbols:
            continue
        diffs = np.asarray(
            [profile_symbol_gaps[profile_a][symbol] - profile_symbol_gaps[profile_b][symbol] for symbol in symbols],
            dtype=float,
        )
        perm = paired_permutation_stats(
            diffs,
            n_permutation=n_permutation,
            seed=seed + 307 * idx,
        )
        conf = bootstrap_mean_confidence(
            diffs,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed + 101 * idx,
            empty_message="Cannot compute confidence from empty paired differences.",
        )
        mean_diff = float(perm["mean_diff"])
        p_a = float(perm["p_observed_or_higher"])
        p_b = float(perm["p_observed_or_lower"])
        rows.append(
            {
                "profile_a": profile_a,
                "profile_b": profile_b,
                "paired_symbol_count": int(perm["pair_count"]),
                "mean_gap_diff_a_minus_b": mean_diff,
                "diff_ci_low": float(conf["ci_low"]),
                "diff_ci_high": float(conf["ci_high"]),
                "diff_ci_width": float(conf["ci_width"]),
                "p_value_two_sided": float(perm["p_two_sided"]),
                "p_value_profile_a_better": p_a,
                "p_value_profile_b_better": p_b,
                "profile_a_better_significant": bool(mean_diff > 0 and p_a <= alpha),
                "profile_b_better_significant": bool(mean_diff < 0 and p_b <= alpha),
            }
        )
    if not rows:
        raise ValueError("No overlapping symbols found across profile results.")
    return pd.DataFrame(rows).sort_values(
        ["p_value_two_sided", "p_value_profile_a_better", "mean_gap_diff_a_minus_b"],
        ascending=[True, True, False],
    )

def _aggregate_profile_rows(symbol_profile_rows: pd.DataFrame, pairwise_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        symbol_profile_rows.groupby("profile", as_index=False)
        .agg(
            symbols_covered=("symbol", "nunique"),
            mean_equity_gap=("avg_equity_gap", "mean"),
            median_equity_gap=("avg_equity_gap", "median"),
            std_equity_gap=("avg_equity_gap", "std"),
            mean_rank=("symbol_rank", "mean"),
            median_rank=("symbol_rank", "median"),
            win_count=("symbol_rank", lambda values: int((values == 1).sum())),
        )
        .copy()
    )
    grouped["std_equity_gap"] = grouped["std_equity_gap"].fillna(0.0)
    grouped["win_rate"] = grouped["win_count"] / grouped["symbols_covered"].clip(lower=1)

    wins: defaultdict[str, int] = defaultdict(int)
    losses: defaultdict[str, int] = defaultdict(int)
    for _, row in pairwise_rows.iterrows():
        profile_a = str(row["profile_a"])
        profile_b = str(row["profile_b"])
        if bool(row["profile_a_better_significant"]):
            wins[profile_a] += 1
            losses[profile_b] += 1
        if bool(row["profile_b_better_significant"]):
            wins[profile_b] += 1
            losses[profile_a] += 1

    grouped["significant_wins"] = grouped["profile"].map(lambda name: wins.get(str(name), 0))
    grouped["significant_losses"] = grouped["profile"].map(lambda name: losses.get(str(name), 0))
    grouped["net_significant_wins"] = grouped["significant_wins"] - grouped["significant_losses"]
    ranked = grouped.sort_values(
        ["win_rate", "median_rank", "mean_equity_gap", "net_significant_wins", "profile"],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked

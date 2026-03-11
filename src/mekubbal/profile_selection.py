from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _load_existing_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _load_symbol_pairwise_flags(
    pairwise_csv_path: str | Path | None,
    *,
    base_profile: str,
    candidate_profile: str,
) -> tuple[bool | None, bool | None]:
    if pairwise_csv_path is None:
        return None, None
    path = Path(pairwise_csv_path)
    if not path.exists():
        return None, None
    frame = pd.read_csv(path)
    if frame.empty:
        return None, None
    required = {
        "profile_a",
        "profile_b",
        "profile_a_better_significant",
        "profile_b_better_significant",
    }
    if not required.issubset(frame.columns):
        return None, None
    for _, row in frame.iterrows():
        profile_a = str(row["profile_a"])
        profile_b = str(row["profile_b"])
        if profile_a == candidate_profile and profile_b == base_profile:
            return bool(row["profile_a_better_significant"]), bool(row["profile_b_better_significant"])
        if profile_a == base_profile and profile_b == candidate_profile:
            return bool(row["profile_b_better_significant"]), bool(row["profile_a_better_significant"])
    return None, None


def run_profile_promotion(
    profile_symbol_summary_path: str | Path,
    state_path: str | Path,
    *,
    base_profile: str = "base",
    candidate_profile: str = "candidate",
    min_candidate_gap_vs_base: float = 0.0,
    max_candidate_rank: int = 1,
    require_candidate_significant: bool = False,
    forbid_base_significant_better: bool = True,
    prefer_previous_active: bool = True,
    fallback_profile: str = "base",
) -> dict[str, Any]:
    if max_candidate_rank < 1:
        raise ValueError("max_candidate_rank must be >= 1.")

    summary_path = Path(profile_symbol_summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Profile symbol summary does not exist: {summary_path}")
    summary = pd.read_csv(summary_path)
    required_cols = {"symbol", "profile", "symbol_rank", "avg_equity_gap"}
    missing = sorted(required_cols - set(summary.columns))
    if missing:
        raise ValueError(f"Profile symbol summary missing required columns: {missing}")
    if summary.empty:
        raise ValueError("Profile symbol summary has no rows.")

    state_file = Path(state_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    existing_state = _load_existing_state(state_file)
    existing_active = existing_state.get("active_profiles", {}) if isinstance(existing_state, dict) else {}
    if not isinstance(existing_active, dict):
        existing_active = {}

    decisions: list[dict[str, Any]] = []
    active_profiles: dict[str, str] = {}
    symbols = sorted({str(value).upper() for value in summary["symbol"].tolist()})
    for symbol in symbols:
        symbol_rows = summary[summary["symbol"].astype(str).str.upper() == symbol].copy()
        if symbol_rows.empty:
            continue
        symbol_rows["symbol_rank"] = pd.to_numeric(symbol_rows["symbol_rank"], errors="coerce")
        symbol_rows["avg_equity_gap"] = pd.to_numeric(symbol_rows["avg_equity_gap"], errors="coerce")
        symbol_rows = symbol_rows.dropna(subset=["symbol_rank", "avg_equity_gap"])
        if symbol_rows.empty:
            continue

        by_profile = {
            str(row["profile"]): row for _, row in symbol_rows.sort_values("symbol_rank").iterrows()
        }
        candidate_row = by_profile.get(candidate_profile)
        base_row = by_profile.get(base_profile)
        previous_active = str(existing_active.get(symbol)) if symbol in existing_active else None
        reasons: list[str] = []

        candidate_better_significant: bool | None = None
        base_better_significant: bool | None = None
        pairwise_csv = None
        if "symbol_pairwise_csv_path" in symbol_rows.columns:
            non_null_pairwise = [
                str(value)
                for value in symbol_rows["symbol_pairwise_csv_path"].tolist()
                if isinstance(value, str) and value.strip()
            ]
            pairwise_csv = non_null_pairwise[0] if non_null_pairwise else None
        candidate_better_significant, base_better_significant = _load_symbol_pairwise_flags(
            pairwise_csv,
            base_profile=base_profile,
            candidate_profile=candidate_profile,
        )

        promote = True
        candidate_rank = None
        candidate_gap = None
        base_rank = None
        base_gap = None
        delta = None
        if candidate_row is None:
            promote = False
            reasons.append("candidate_missing")
        else:
            candidate_rank = int(candidate_row["symbol_rank"])
            candidate_gap = float(candidate_row["avg_equity_gap"])
            if candidate_rank > max_candidate_rank:
                promote = False
                reasons.append("candidate_rank_too_low")

        if base_row is None:
            promote = False
            reasons.append("base_missing")
        else:
            base_rank = int(base_row["symbol_rank"])
            base_gap = float(base_row["avg_equity_gap"])

        if candidate_gap is not None and base_gap is not None:
            delta = float(candidate_gap - base_gap)
            if delta < float(min_candidate_gap_vs_base):
                promote = False
                reasons.append("candidate_gap_delta_below_threshold")

        if require_candidate_significant and candidate_better_significant is not True:
            promote = False
            reasons.append("candidate_not_significantly_better")
        if forbid_base_significant_better and base_better_significant is True:
            promote = False
            reasons.append("base_significantly_better")

        if promote:
            active_profile = candidate_profile
        else:
            if prefer_previous_active and previous_active in by_profile:
                active_profile = previous_active
                reasons.append("kept_previous_active")
            elif fallback_profile in by_profile:
                active_profile = fallback_profile
                reasons.append("using_fallback_profile")
            else:
                active_profile = str(symbol_rows.sort_values("symbol_rank").iloc[0]["profile"])
                reasons.append("using_best_ranked_profile")

        active_profiles[symbol] = active_profile
        decisions.append(
            {
                "symbol": symbol,
                "promoted": bool(promote),
                "active_profile": active_profile,
                "previous_active_profile": previous_active,
                "candidate_profile": candidate_profile,
                "base_profile": base_profile,
                "candidate_rank": candidate_rank,
                "base_rank": base_rank,
                "candidate_gap": candidate_gap,
                "base_gap": base_gap,
                "candidate_minus_base_gap": delta,
                "candidate_better_significant": candidate_better_significant,
                "base_better_significant": base_better_significant,
                "pairwise_csv_path": pairwise_csv,
                "reasons": reasons,
            }
        )

    state = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_profile_symbol_summary_path": str(summary_path.resolve()),
        "promotion_rule": {
            "base_profile": base_profile,
            "candidate_profile": candidate_profile,
            "min_candidate_gap_vs_base": float(min_candidate_gap_vs_base),
            "max_candidate_rank": int(max_candidate_rank),
            "require_candidate_significant": bool(require_candidate_significant),
            "forbid_base_significant_better": bool(forbid_base_significant_better),
            "prefer_previous_active": bool(prefer_previous_active),
            "fallback_profile": fallback_profile,
        },
        "active_profiles": active_profiles,
        "symbols": decisions,
    }
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "state_path": str(state_file),
        "symbols_evaluated": int(len(decisions)),
        "promoted_count": int(sum(1 for row in decisions if bool(row["promoted"]))),
        "active_profiles": active_profiles,
    }

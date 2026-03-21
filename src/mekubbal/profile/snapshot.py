from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_profile_selection_state(state_path: str | Path) -> dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile selection state does not exist: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Profile selection state must decode to a JSON object.")
    return loaded


def build_active_snapshot(
    symbol_summary: pd.DataFrame,
    selection_state: dict[str, Any],
    *,
    run_timestamp_utc: str,
    active_profiles_override: dict[str, str] | None = None,
    ensemble_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"symbol", "profile", "symbol_rank", "avg_equity_gap"}
    missing = sorted(required - set(symbol_summary.columns))
    if missing:
        raise ValueError(f"Profile symbol summary missing required columns: {missing}")
    if symbol_summary.empty:
        raise ValueError("Profile symbol summary is empty.")

    selected_profiles = selection_state.get("active_profiles", {})
    if not isinstance(selected_profiles, dict):
        raise ValueError("profile_selection_state.active_profiles must be an object.")
    promotion_rule = selection_state.get("promotion_rule", {})
    base_profile = str(promotion_rule.get("base_profile", "base"))
    candidate_profile = str(promotion_rule.get("candidate_profile", "candidate"))
    decision_rows = selection_state.get("symbols", [])
    by_symbol_decision: dict[str, dict[str, Any]] = {}
    if isinstance(decision_rows, list):
        for item in decision_rows:
            if not isinstance(item, dict) or "symbol" not in item:
                continue
            by_symbol_decision[str(item["symbol"]).upper()] = item
    by_symbol_ensemble: dict[str, dict[str, Any]] = {}
    if ensemble_decisions is not None and not ensemble_decisions.empty:
        for _, item in ensemble_decisions.iterrows():
            symbol = str(item.get("symbol", "")).upper()
            if not symbol:
                continue
            by_symbol_ensemble[symbol] = {
                "regime": item.get("regime"),
                "regime_confidence": item.get("regime_confidence"),
                "ensemble_confidence": item.get("ensemble_confidence"),
                "decision_reason": item.get("decision_reason"),
                "gated_by_regime": item.get("gated_by_regime"),
            }

    frame = symbol_summary.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["symbol_rank"] = pd.to_numeric(frame["symbol_rank"], errors="coerce")
    frame["avg_equity_gap"] = pd.to_numeric(frame["avg_equity_gap"], errors="coerce")
    frame = frame.dropna(subset=["symbol_rank", "avg_equity_gap"])

    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol"):
        sorted_group = group.sort_values("symbol_rank").reset_index(drop=True)
        by_profile = {str(row["profile"]): row for _, row in sorted_group.iterrows()}
        symbol_category = None
        if "symbol_category" in sorted_group.columns:
            category_values = [str(value).strip() for value in sorted_group["symbol_category"].tolist() if pd.notna(value)]
            if category_values:
                symbol_category = category_values[0] or None
        selected = str(selected_profiles.get(symbol) or "")
        if not selected or selected not in by_profile:
            selected = str(sorted_group.iloc[0]["profile"])
        active = selected
        active_source = "selection_state"
        if active_profiles_override is not None:
            override_profile = str(active_profiles_override.get(symbol) or "")
            if override_profile and override_profile in by_profile:
                active = override_profile
                if active != selected:
                    active_source = "ensemble_v3"
        active_row = by_profile[active]
        selected_row = by_profile[selected]
        base_row = by_profile.get(base_profile)
        candidate_row = by_profile.get(candidate_profile)
        decision = by_symbol_decision.get(symbol, {})
        ensemble_meta = by_symbol_ensemble.get(symbol, {})

        base_gap = float(base_row["avg_equity_gap"]) if base_row is not None else None
        active_gap = float(active_row["avg_equity_gap"])
        rows.append(
            {
                "run_timestamp_utc": run_timestamp_utc,
                "symbol": symbol,
                "symbol_category": symbol_category,
                "selected_profile": selected,
                "selected_rank": int(selected_row["symbol_rank"]),
                "selected_gap": float(selected_row["avg_equity_gap"]),
                "active_profile": active,
                "active_profile_source": active_source,
                "active_rank": int(active_row["symbol_rank"]),
                "active_gap": active_gap,
                "ensemble_regime": ensemble_meta.get("regime"),
                "ensemble_regime_confidence": ensemble_meta.get("regime_confidence"),
                "ensemble_confidence": ensemble_meta.get("ensemble_confidence"),
                "ensemble_decision_reason": ensemble_meta.get("decision_reason"),
                "ensemble_gated_by_regime": ensemble_meta.get("gated_by_regime"),
                "base_profile": base_profile if base_row is not None else None,
                "base_rank": int(base_row["symbol_rank"]) if base_row is not None else None,
                "base_gap": base_gap,
                "active_minus_base_gap": (active_gap - base_gap) if base_gap is not None else None,
                "candidate_profile": candidate_profile if candidate_row is not None else None,
                "candidate_rank": int(candidate_row["symbol_rank"]) if candidate_row is not None else None,
                "candidate_gap": float(candidate_row["avg_equity_gap"]) if candidate_row is not None else None,
                "promoted": bool(decision.get("promoted", False)),
                "promotion_reasons": ";".join(str(value) for value in decision.get("reasons", [])),
            }
        )

    if not rows:
        raise ValueError("No valid symbol rows found in profile summary.")
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

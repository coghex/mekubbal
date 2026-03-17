from __future__ import annotations

from copy import deepcopy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.profile_ensemble import compute_regime_gated_ensemble


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_profile_selection_state(state_path: str | Path) -> dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile selection state does not exist: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Profile selection state must decode to a JSON object.")
    return loaded


def _build_active_snapshot(
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


def _append_history(snapshot: pd.DataFrame, history_path: str | Path) -> pd.DataFrame:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        merged = pd.concat([existing, snapshot], ignore_index=True)
    else:
        merged = snapshot.copy()
    merged.to_csv(path, index=False)
    return merged


def _alert_rows(
    history: pd.DataFrame,
    *,
    lookback_runs: int,
    max_gap_drop: float,
    max_rank_worsening: float,
    min_active_minus_base_gap: float,
) -> pd.DataFrame:
    columns = [
        "symbol",
        "run_timestamp_utc",
        "active_profile",
        "latest_active_gap",
        "baseline_active_gap",
        "gap_drop",
        "latest_active_rank",
        "baseline_active_rank",
        "rank_worsening",
        "latest_active_minus_base_gap",
        "reasons",
    ]
    if lookback_runs < 1:
        raise ValueError("lookback_runs must be >= 1.")

    frame = history.copy()
    frame["run_timestamp_utc"] = frame["run_timestamp_utc"].astype(str)
    frame["active_gap"] = pd.to_numeric(frame["active_gap"], errors="coerce")
    frame["active_rank"] = pd.to_numeric(frame["active_rank"], errors="coerce")
    frame["active_minus_base_gap"] = pd.to_numeric(frame["active_minus_base_gap"], errors="coerce")
    frame = frame.dropna(subset=["active_gap", "active_rank"])

    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol"):
        ordered = group.sort_values("run_timestamp_utc").reset_index(drop=True)
        if len(ordered) <= lookback_runs:
            continue
        for idx in range(int(lookback_runs), len(ordered)):
            latest = ordered.iloc[idx]
            baseline = ordered.iloc[idx - int(lookback_runs) : idx]
            baseline_gap = float(baseline["active_gap"].mean())
            baseline_rank = float(baseline["active_rank"].mean())
            latest_gap = float(latest["active_gap"])
            latest_rank = float(latest["active_rank"])
            latest_minus_base = (
                float(latest["active_minus_base_gap"])
                if pd.notna(latest["active_minus_base_gap"])
                else None
            )

            gap_drop = baseline_gap - latest_gap
            rank_worsening = latest_rank - baseline_rank
            reasons: list[str] = []
            if gap_drop > max_gap_drop:
                reasons.append("gap_drop_exceeded")
            if rank_worsening > max_rank_worsening:
                reasons.append("rank_worsening_exceeded")
            if latest_minus_base is not None and latest_minus_base < min_active_minus_base_gap:
                reasons.append("active_below_base_threshold")
            if not reasons:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "run_timestamp_utc": str(latest["run_timestamp_utc"]),
                    "active_profile": str(latest["active_profile"]),
                    "latest_active_gap": latest_gap,
                    "baseline_active_gap": baseline_gap,
                    "gap_drop": gap_drop,
                    "latest_active_rank": latest_rank,
                    "baseline_active_rank": baseline_rank,
                    "rank_worsening": rank_worsening,
                    "latest_active_minus_base_gap": latest_minus_base,
                    "reasons": ";".join(reasons),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["symbol", "run_timestamp_utc"])
        .reset_index(drop=True)
    )


def compute_drift_alert_history(
    health_history: pd.DataFrame,
    *,
    lookback_runs: int,
    max_gap_drop: float,
    max_rank_worsening: float,
    min_active_minus_base_gap: float,
) -> pd.DataFrame:
    return _alert_rows(
        history=health_history,
        lookback_runs=lookback_runs,
        max_gap_drop=max_gap_drop,
        max_rank_worsening=max_rank_worsening,
        min_active_minus_base_gap=min_active_minus_base_gap,
    )


def compute_ensemble_alert_history(
    health_history: pd.DataFrame,
    *,
    low_confidence_threshold: float,
) -> pd.DataFrame:
    columns = [
        "symbol",
        "run_timestamp_utc",
        "active_profile",
        "selected_profile",
        "ensemble_regime",
        "ensemble_regime_confidence",
        "ensemble_confidence",
        "reasons",
    ]
    required = {
        "symbol",
        "run_timestamp_utc",
        "active_profile",
        "selected_profile",
        "ensemble_regime",
        "ensemble_confidence",
    }
    if float(low_confidence_threshold) < 0 or float(low_confidence_threshold) > 1:
        raise ValueError("low_confidence_threshold must be in [0, 1].")
    if not required.issubset(health_history.columns):
        return pd.DataFrame(columns=columns)

    frame = health_history.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["run_timestamp_utc"] = frame["run_timestamp_utc"].astype(str)
    frame["ensemble_confidence"] = pd.to_numeric(frame["ensemble_confidence"], errors="coerce")
    frame["ensemble_regime_confidence"] = pd.to_numeric(
        frame.get("ensemble_regime_confidence"),
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        reasons: list[str] = []
        ensemble_confidence = (
            float(row["ensemble_confidence"]) if pd.notna(row["ensemble_confidence"]) else None
        )
        regime = str(row.get("ensemble_regime") or "")
        selected = str(row.get("selected_profile") or "")
        active = str(row.get("active_profile") or "")
        if ensemble_confidence is not None and ensemble_confidence < float(low_confidence_threshold):
            reasons.append("low_ensemble_confidence")
        if regime == "high_vol" and selected and active and selected != active:
            reasons.append("high_vol_profile_disagreement")
        if not reasons:
            continue
        rows.append(
            {
                "symbol": str(row["symbol"]).upper(),
                "run_timestamp_utc": str(row["run_timestamp_utc"]),
                "active_profile": active,
                "selected_profile": selected,
                "ensemble_regime": regime,
                "ensemble_regime_confidence": (
                    float(row["ensemble_regime_confidence"])
                    if pd.notna(row["ensemble_regime_confidence"])
                    else None
                ),
                "ensemble_confidence": ensemble_confidence,
                "reasons": ";".join(reasons),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["symbol", "run_timestamp_utc"]).reset_index(
        drop=True
    )


def _html_table(title: str, note: str, frame: pd.DataFrame) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f5f5f5; }}
    .note {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 12px; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="note">{note}</div>
  {frame.to_html(index=False)}
</body>
</html>
"""


def _format_pct_points(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:+.2f}%"


def _reason_to_label(reason: str) -> str:
    mapping = {
        "gap_drop_exceeded": "recent performance drop",
        "rank_worsening_exceeded": "profile rank worsened",
        "active_below_base_threshold": "active profile below base threshold",
    }
    return mapping.get(reason, reason)


def _streak_length(values: list[float], *, positive: bool) -> int:
    streak = 0
    for value in reversed(values):
        if positive and value > 0:
            streak += 1
            continue
        if (not positive) and value < 0:
            streak += 1
            continue
        break
    return streak


def _build_ticker_history_context(history: pd.DataFrame, *, recent_window: int = 3) -> dict[str, dict[str, Any]]:
    required = {"symbol", "run_timestamp_utc", "active_gap", "active_rank"}
    if history.empty or not required.issubset(history.columns):
        return {}

    frame = history.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["run_timestamp_utc"] = frame["run_timestamp_utc"].astype(str)
    frame["active_gap"] = pd.to_numeric(frame["active_gap"], errors="coerce")
    frame["active_rank"] = pd.to_numeric(frame["active_rank"], errors="coerce")
    frame["active_minus_base_gap"] = pd.to_numeric(frame.get("active_minus_base_gap"), errors="coerce")
    frame = frame.dropna(subset=["active_gap", "active_rank"])
    if frame.empty:
        return {}

    contexts: dict[str, dict[str, Any]] = {}
    for symbol, group in frame.groupby("symbol"):
        ordered = group.sort_values("run_timestamp_utc").reset_index(drop=True)
        gaps = [float(value) for value in ordered["active_gap"].tolist() if pd.notna(value)]
        if not gaps:
            continue
        recent = ordered.tail(min(int(recent_window), len(ordered))).copy()
        recent_gaps = [float(value) for value in recent["active_gap"].tolist() if pd.notna(value)]
        recent_base_deltas = [
            float(value) for value in recent["active_minus_base_gap"].tolist() if pd.notna(value)
        ]
        recent_ranks = [float(value) for value in recent["active_rank"].tolist() if pd.notna(value)]
        sign_flip_count = 0
        if len(recent_gaps) >= 2:
            for previous, current in zip(recent_gaps[:-1], recent_gaps[1:]):
                if (previous > 0 and current < 0) or (previous < 0 and current > 0):
                    sign_flip_count += 1

        positive_runs = sum(1 for value in gaps if value > 0)
        positive_streak = _streak_length(gaps, positive=True)
        negative_streak = _streak_length(gaps, positive=False)
        recent_avg_gap = float(sum(recent_gaps) / len(recent_gaps)) if recent_gaps else None
        recent_gap_std = (
            float(pd.Series(recent_gaps, dtype=float).std(ddof=0)) if len(recent_gaps) > 1 else 0.0
        )
        recent_avg_rank = float(sum(recent_ranks) / len(recent_ranks)) if recent_ranks else None
        recent_avg_base_delta = (
            float(sum(recent_base_deltas) / len(recent_base_deltas)) if recent_base_deltas else None
        )
        momentum = float(gaps[-1] - gaps[-2]) if len(gaps) > 1 else None

        if len(gaps) < 3:
            stability_label = "Early"
        elif negative_streak >= 2 or (recent_avg_gap is not None and recent_avg_gap < 0):
            stability_label = "Weakening"
        elif (
            positive_streak >= 2
            and recent_avg_gap is not None
            and recent_avg_gap > 0
            and recent_gap_std <= 0.015
            and sign_flip_count == 0
        ):
            stability_label = "Stable"
        elif recent_avg_gap is not None and recent_avg_gap > 0:
            stability_label = "Building"
        else:
            stability_label = "Choppy"

        contexts[str(symbol)] = {
            "runs_observed": int(len(gaps)),
            "positive_runs": int(positive_runs),
            "positive_run_share": float(positive_runs / len(gaps)),
            "positive_streak": int(positive_streak),
            "negative_streak": int(negative_streak),
            "recent_window": int(len(recent_gaps)),
            "recent_avg_gap": recent_avg_gap,
            "recent_gap_std": recent_gap_std,
            "recent_avg_rank": recent_avg_rank,
            "recent_avg_base_delta": recent_avg_base_delta,
            "momentum": momentum,
            "sign_flip_count": int(sign_flip_count),
            "stability_label": stability_label,
        }
    return contexts


def _comparison_phrase(reference: str, value: float | None) -> str:
    if value is None:
        return f"comparison versus {reference} is unavailable"
    magnitude = abs(float(value)) * 100.0
    if magnitude < 0.005:
        return f"roughly flat versus {reference}"
    direction = "ahead of" if value > 0 else "behind"
    return f"{magnitude:.2f}% {direction} {reference}"


def _regime_context(regime: str) -> str | None:
    cleaned = str(regime).strip().lower()
    if not cleaned:
        return None
    mapping = {
        "stable": "The current backdrop looks relatively steady.",
        "calm": "The current backdrop looks relatively steady.",
        "trending": "The market backdrop looks directional rather than range-bound.",
        "high_vol": "The market backdrop looks choppier and higher-volatility than usual.",
        "turbulent": "The market backdrop looks choppier and higher-volatility than usual.",
    }
    return mapping.get(cleaned, f"Current market backdrop: {cleaned.replace('_', ' ')}.")


def _confidence_label(
    *,
    ensemble_confidence: float | None,
    has_alert: bool,
    severe_alert: bool,
    active_gap: float,
    active_rank: int,
    active_minus_base: float | None,
    history_context: dict[str, Any],
) -> str:
    if ensemble_confidence is not None:
        confidence_floor = (
            "High"
            if ensemble_confidence >= 0.75
            else "Medium"
            if ensemble_confidence >= 0.55
            else "Low"
        )
    else:
        confidence_floor = None

    runs_observed = int(history_context.get("runs_observed", 0))
    positive_streak = int(history_context.get("positive_streak", 0))
    negative_streak = int(history_context.get("negative_streak", 0))
    recent_avg_gap = history_context.get("recent_avg_gap")
    stability_label = str(history_context.get("stability_label", "Early"))

    if severe_alert or negative_streak >= 2:
        return "Low"
    if recent_avg_gap is not None and float(recent_avg_gap) < 0:
        return "Low"
    if confidence_floor == "Low":
        return "Low"
    if has_alert:
        return "Medium" if active_gap > 0 and positive_streak >= 1 else "Low"
    if (
        stability_label == "Stable"
        and runs_observed >= 3
        and positive_streak >= 2
        and active_gap > 0
        and active_rank == 1
        and (active_minus_base is None or active_minus_base >= 0)
        and confidence_floor != "Medium"
    ):
        return "High"
    if confidence_floor == "High" and runs_observed >= 2 and active_gap > 0:
        return "High"
    if active_gap > 0 and recent_avg_gap is not None and float(recent_avg_gap) > 0:
        return "Medium"
    if active_gap > 0 and active_rank == 1 and runs_observed <= 1:
        return "Medium"
    if active_gap > 0:
        return "Medium"
    return "Low"


def _recommendation_fields(
    *,
    symbol: str,
    active_profile: str,
    base_profile_text: str,
    active_gap: float,
    active_minus_base: float | None,
    active_rank: int,
    reason_labels: str,
    has_alert: bool,
    severe_alert: bool,
    promoted: bool,
    regime_note: str | None,
    selected_profile: str,
    confidence: str,
    history_context: dict[str, Any],
) -> dict[str, str]:
    runs_observed = int(history_context.get("runs_observed", 0))
    positive_runs = int(history_context.get("positive_runs", 0))
    positive_run_share = float(history_context.get("positive_run_share", 0.0))
    positive_streak = int(history_context.get("positive_streak", 0))
    negative_streak = int(history_context.get("negative_streak", 0))
    recent_window = int(history_context.get("recent_window", 0))
    recent_avg_gap = history_context.get("recent_avg_gap")
    recent_avg_base_delta = history_context.get("recent_avg_base_delta")
    recent_gap_std = history_context.get("recent_gap_std")
    stability_label = str(history_context.get("stability_label", "Early"))

    strong_setup = (
        not has_alert
        and runs_observed >= 3
        and positive_streak >= 2
        and positive_run_share >= (2.0 / 3.0)
        and recent_avg_gap is not None
        and float(recent_avg_gap) > 0.01
        and (active_minus_base is None or active_minus_base >= 0)
        and active_rank == 1
        and stability_label == "Stable"
    )
    constructive_setup = (
        active_gap > 0
        and recent_avg_gap is not None
        and float(recent_avg_gap) > 0
        and (recent_avg_base_delta is None or float(recent_avg_base_delta) >= -0.0025)
        and negative_streak == 0
        and not severe_alert
    )
    mixed_setup = (
        recent_avg_gap is not None
        and float(recent_avg_gap) >= -0.0025
        and active_gap >= -0.005
        and not severe_alert
    )

    if severe_alert or active_gap < 0 or (recent_avg_gap is not None and float(recent_avg_gap) < -0.005):
        recommendation = "Avoid for now"
        recommendation_subtitle = "edge is weak or deteriorating"
        recommended_action = (
            "Avoid leaning on this ticker for now until several daily updates improve the trend."
        )
    elif has_alert:
        recommendation = "Caution"
        recommendation_subtitle = "promise is there, but warning flags are active"
        recommended_action = (
            "Keep this ticker on watch, but wait for the warning flags to clear before giving it more weight."
        )
    elif strong_setup:
        recommendation = "Bullish setup"
        recommendation_subtitle = "positive and stable"
        recommended_action = (
            "This looks like a bullish setup: repeated positive evidence is still beating the simpler benchmarks."
        )
    elif constructive_setup:
        recommendation = "Improving trend"
        recommendation_subtitle = "positive, but still proving itself"
        recommended_action = (
            "The setup is positive, but it still needs a little more stability before it becomes a top pick."
        )
    elif mixed_setup:
        recommendation = "Mixed trend"
        recommendation_subtitle = "signals are mixed and not confirmed"
        recommended_action = (
            "Signals are mixed right now, so this ticker is better treated as a watchlist name than a priority."
        )
    else:
        recommendation = "Avoid for now"
        recommendation_subtitle = "edge is weak or deteriorating"
        recommended_action = (
            "The current edge is weak, so wait for stronger and more consistent daily evidence before leaning on this ticker."
        )

    if runs_observed <= 1:
        evidence_text = "This view is based on only 1 recorded run so far."
    else:
        evidence_text = (
            f"Over the last {runs_observed} runs, it stayed ahead of buy-and-hold in "
            f"{positive_runs}/{runs_observed} runs."
        )
    if positive_streak >= 2:
        stability_text = f"It has held a positive edge for {positive_streak} runs in a row."
    elif negative_streak >= 2:
        stability_text = f"It has been negative for {negative_streak} runs in a row."
    elif stability_label == "Early":
        stability_text = "Recent evidence is still limited."
    elif stability_label == "Stable":
        stability_text = "The recent signal has been steady rather than choppy."
    elif stability_label == "Building":
        stability_text = "The recent signal is positive, but still building proof."
    elif stability_label == "Weakening":
        stability_text = "The recent signal has weakened."
    else:
        stability_text = "The recent signal has been choppy."

    summary_parts = [
        f"{symbol} is {_comparison_phrase('buy-and-hold', active_gap)}.",
        f"It is {_comparison_phrase(f'your {base_profile_text} setup', active_minus_base)}.",
        evidence_text,
        stability_text,
    ]
    if selected_profile != active_profile:
        summary_parts.append(
            f"The active model is coming from an override that switched from {selected_profile} to {active_profile}."
        )
    elif promoted:
        summary_parts.append(f"{active_profile} became the active setup on this run.")
    else:
        summary_parts.append(f"{active_profile} remains the active setup.")
    if has_alert:
        summary_parts.append(f"Current warning signs: {reason_labels}.")
    else:
        summary_parts.append("There are no active drift warnings on this run.")
    if regime_note:
        summary_parts.append(regime_note)
    if recent_avg_gap is not None:
        summary_parts.append(
            f"Recent average edge over the last {max(recent_window, 1)} run(s) is {_format_pct_points(recent_avg_gap)}."
        )

    if severe_alert or active_gap < 0 or negative_streak >= 2:
        what_to_watch = (
            "Watch for the model to move back ahead of buy-and-hold, rebuild a positive streak, and clear its current warning flags."
        )
    elif has_alert:
        what_to_watch = (
            "Watch the next run closely: this setup needs its warning flags to clear before confidence improves."
        )
    elif runs_observed < 3 and active_gap > 0:
        what_to_watch = (
            "Watch for a few more solid daily runs before treating this signal as fully proven."
        )
    elif recent_gap_std is not None and float(recent_gap_std) > 0.015:
        what_to_watch = (
            "Watch whether the recent signal becomes steadier, because the edge has still been moving around from run to run."
        )
    elif active_gap > 0 and (active_minus_base is None or active_minus_base >= 0):
        what_to_watch = (
            "Watch whether the gap versus buy-and-hold stays positive over the next few daily updates."
        )
    else:
        what_to_watch = (
            "Watch for clearer separation from the baseline setup before upgrading this ticker."
        )

    if confidence == "Low" and not has_alert and active_gap > 0:
        what_to_watch += " Confidence is still limited, so treat the signal as early rather than proven."

    return {
        "recommendation": recommendation,
        "recommendation_subtitle": recommendation_subtitle,
        "recommended_action": recommended_action,
        "summary": " ".join(summary_parts),
        "what_to_watch": what_to_watch,
    }


def _build_ticker_summary(snapshot: pd.DataFrame, alerts: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    alert_map: dict[str, list[str]] = {}
    if not alerts.empty:
        for _, row in alerts.iterrows():
            symbol = str(row["symbol"]).upper()
            raw = str(row.get("reasons", ""))
            reasons = [value.strip() for value in raw.split(";") if value.strip()]
            alert_map[symbol] = reasons
    history_context_by_symbol = _build_ticker_history_context(history)

    rows: list[dict[str, Any]] = []
    for _, row in snapshot.sort_values("symbol").iterrows():
        symbol = str(row["symbol"]).upper()
        reasons = alert_map.get(symbol, [])
        selected_profile = str(row.get("selected_profile") or row.get("active_profile"))
        active_profile = str(row["active_profile"])
        regime = str(row.get("ensemble_regime") or "")
        ensemble_confidence = (
            float(row["ensemble_confidence"]) if pd.notna(row.get("ensemble_confidence")) else None
        )
        regime_confidence = (
            float(row["ensemble_regime_confidence"])
            if pd.notna(row.get("ensemble_regime_confidence"))
            else None
        )
        active_gap = float(row["active_gap"])
        active_minus_base = (
            float(row["active_minus_base_gap"])
            if pd.notna(row["active_minus_base_gap"])
            else None
        )
        promoted = bool(row.get("promoted", False))
        has_alert = bool(reasons)
        severe_alert = bool(
            "active_below_base_threshold" in reasons
            or (active_minus_base is not None and active_minus_base < 0)
        )
        reason_labels = ", ".join(_reason_to_label(value) for value in reasons) if reasons else "none"
        regime_note = _regime_context(regime)

        if severe_alert and has_alert:
            status = "Critical"
        elif has_alert:
            status = "Watch"
        else:
            status = "Healthy"
        base_profile = row.get("base_profile")
        base_profile_text = str(base_profile) if pd.notna(base_profile) else "base"
        active_rank = int(row["active_rank"])
        history_context = history_context_by_symbol.get(symbol, {})
        confidence = _confidence_label(
            ensemble_confidence=ensemble_confidence,
            has_alert=has_alert,
            severe_alert=severe_alert,
            active_gap=active_gap,
            active_rank=active_rank,
            active_minus_base=active_minus_base,
            history_context=history_context,
        )
        recommendation_fields = _recommendation_fields(
            symbol=symbol,
            active_profile=active_profile,
            base_profile_text=base_profile_text,
            active_gap=active_gap,
            active_minus_base=active_minus_base,
            active_rank=active_rank,
            reason_labels=reason_labels,
            has_alert=has_alert,
            severe_alert=severe_alert,
            promoted=promoted,
            regime_note=regime_note,
            selected_profile=selected_profile,
            confidence=confidence,
            history_context=history_context,
        )
        rows.append(
            {
                "symbol": symbol,
                "status": status,
                "recommendation": recommendation_fields["recommendation"],
                "recommendation_subtitle": recommendation_fields["recommendation_subtitle"],
                "confidence": confidence,
                "selected_profile": selected_profile,
                "active_profile": active_profile,
                "active_profile_source": str(row.get("active_profile_source") or "selection_state"),
                "ensemble_regime": regime or None,
                "ensemble_regime_confidence": regime_confidence,
                "ensemble_confidence": ensemble_confidence,
                "ensemble_decision_reason": (
                    str(row.get("ensemble_decision_reason"))
                    if pd.notna(row.get("ensemble_decision_reason"))
                    else None
                ),
                "active_rank": int(row["active_rank"]),
                "active_vs_buy_and_hold": _format_pct_points(active_gap),
                "active_vs_base": _format_pct_points(active_minus_base),
                "runs_observed": int(history_context.get("runs_observed", 0)),
                "positive_run_share": (
                    round(float(history_context["positive_run_share"]), 4)
                    if "positive_run_share" in history_context
                    else None
                ),
                "positive_streak": int(history_context.get("positive_streak", 0)),
                "signal_stability": str(history_context.get("stability_label", "Early")),
                "recent_avg_edge": _format_pct_points(history_context.get("recent_avg_gap")),
                "promoted_this_run": bool(promoted),
                "alert_reasons": reason_labels,
                "recommended_action": recommendation_fields["recommended_action"],
                "summary": recommendation_fields["summary"],
                "what_to_watch": recommendation_fields["what_to_watch"],
            }
        )
    return pd.DataFrame(rows)


def run_profile_monitor(
    *,
    profile_symbol_summary_path: str | Path,
    selection_state_path: str | Path,
    health_snapshot_path: str | Path,
    health_history_path: str | Path,
    drift_alerts_csv_path: str | Path,
    drift_alerts_html_path: str | Path,
    drift_alerts_history_path: str | Path | None = None,
    ticker_summary_csv_path: str | Path | None = None,
    ticker_summary_html_path: str | Path | None = None,
    lookback_runs: int = 3,
    max_gap_drop: float = 0.03,
    max_rank_worsening: float = 0.75,
    min_active_minus_base_gap: float = -0.01,
    run_timestamp_utc: str | None = None,
    ensemble_v3_config: dict[str, Any] | None = None,
    ensemble_decisions_csv_path: str | Path | None = None,
    ensemble_history_path: str | Path | None = None,
    ensemble_effective_selection_state_path: str | Path | None = None,
    ensemble_alerts_csv_path: str | Path | None = None,
    ensemble_alerts_html_path: str | Path | None = None,
    ensemble_alerts_history_path: str | Path | None = None,
    ensemble_low_confidence_threshold: float = 0.55,
) -> dict[str, Any]:
    summary_path = Path(profile_symbol_summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Profile symbol summary does not exist: {summary_path}")
    symbol_summary = pd.read_csv(summary_path)
    selection_state = _load_profile_selection_state(selection_state_path)
    run_time = run_timestamp_utc or _now_utc_iso()

    ensemble_summary: dict[str, Any] | None = None
    active_profiles_override: dict[str, str] | None = None
    effective_selection_state_written: str | None = None
    ensemble_decisions_for_snapshot: pd.DataFrame | None = None
    if ensemble_v3_config is not None and bool(ensemble_v3_config.get("enabled", False)):
        if (
            ensemble_decisions_csv_path is None
            or ensemble_history_path is None
            or ensemble_effective_selection_state_path is None
        ):
            raise ValueError(
                "ensemble_decisions_csv_path, ensemble_history_path, and "
                "ensemble_effective_selection_state_path are required when ensemble_v3 is enabled."
            )

        history_for_regime = (
            pd.read_csv(health_history_path)
            if Path(health_history_path).exists()
            else pd.DataFrame(
                columns=["symbol", "run_timestamp_utc", "active_gap", "active_rank"]
            )
        )
        ensemble_result = compute_regime_gated_ensemble(
            symbol_summary,
            selection_state,
            history_for_regime,
            lookback_runs=int(ensemble_v3_config["lookback_runs"]),
            min_regime_confidence=float(ensemble_v3_config["min_regime_confidence"]),
            rank_weight=float(ensemble_v3_config["rank_weight"]),
            gap_weight=float(ensemble_v3_config["gap_weight"]),
            significance_bonus=float(ensemble_v3_config["significance_bonus"]),
            fallback_profile=str(ensemble_v3_config["fallback_profile"]),
            profile_weights=dict(ensemble_v3_config["profile_weights"]),
            regime_multipliers=dict(ensemble_v3_config["regime_multipliers"]),
            high_vol_gap_std_threshold=float(ensemble_v3_config["high_vol_gap_std_threshold"]),
            high_vol_rank_std_threshold=float(ensemble_v3_config["high_vol_rank_std_threshold"]),
            trending_min_gap_improvement=float(ensemble_v3_config["trending_min_gap_improvement"]),
            trending_min_rank_improvement=float(ensemble_v3_config["trending_min_rank_improvement"]),
        )
        active_profiles_override = dict(ensemble_result["ensembled_profiles"])
        ensemble_decisions = ensemble_result["decisions"].copy()
        ensemble_decisions_for_snapshot = ensemble_decisions.copy()
        ensemble_decisions.insert(0, "run_timestamp_utc", run_time)

        decisions_path = Path(ensemble_decisions_csv_path)
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble_decisions.to_csv(decisions_path, index=False)

        history_path = Path(ensemble_history_path)
        history_written = _append_history(ensemble_decisions, history_path)

        effective_state = deepcopy(selection_state)
        effective_state["active_profiles"] = active_profiles_override
        effective_state["ensemble_v3"] = {
            "enabled": True,
            "run_timestamp_utc": run_time,
            "source_selection_state_path": str(Path(selection_state_path).resolve()),
            "decisions_csv_path": str(decisions_path),
        }
        effective_state_path = Path(ensemble_effective_selection_state_path)
        effective_state_path.parent.mkdir(parents=True, exist_ok=True)
        effective_state_path.write_text(
            json.dumps(effective_state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        effective_selection_state_written = str(effective_state_path)
        ensemble_summary = {
            "enabled": True,
            "decisions_csv_path": str(decisions_path),
            "history_path": str(history_path),
            "history_rows": int(len(history_written)),
            "effective_selection_state_path": effective_selection_state_written,
            "symbols_ensembled": int(len(active_profiles_override)),
            "regimes": ensemble_result["regimes"],
        }

    snapshot = _build_active_snapshot(
        symbol_summary,
        selection_state,
        run_timestamp_utc=run_time,
        active_profiles_override=active_profiles_override,
        ensemble_decisions=ensemble_decisions_for_snapshot,
    )
    snapshot_path = Path(health_snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(snapshot_path, index=False)

    history = _append_history(snapshot, health_history_path)
    ensemble_alert_history_written = None
    ensemble_alerts_csv_written = None
    ensemble_alerts_html_written = None
    ensemble_alerts_count = 0
    ensemble_alerts_history_count = 0
    if (
        ensemble_alerts_csv_path is not None
        or ensemble_alerts_html_path is not None
        or ensemble_alerts_history_path is not None
    ):
        if ensemble_alerts_csv_path is None or ensemble_alerts_html_path is None:
            raise ValueError(
                "ensemble_alerts_csv_path and ensemble_alerts_html_path must be provided together."
            )
        all_ensemble_alerts = compute_ensemble_alert_history(
            history,
            low_confidence_threshold=float(ensemble_low_confidence_threshold),
        )
        current_ensemble_alerts = all_ensemble_alerts[
            all_ensemble_alerts["run_timestamp_utc"].astype(str) == str(run_time)
        ].copy()
        ensemble_csv = Path(ensemble_alerts_csv_path)
        ensemble_html = Path(ensemble_alerts_html_path)
        ensemble_csv.parent.mkdir(parents=True, exist_ok=True)
        ensemble_html.parent.mkdir(parents=True, exist_ok=True)
        current_ensemble_alerts.to_csv(ensemble_csv, index=False)
        ensemble_html.write_text(
            _html_table(
                "Ensemble Ops Alerts",
                (
                    "Alerts fire for low ensemble confidence or high-vol profile disagreement. "
                    f"low_confidence_threshold={float(ensemble_low_confidence_threshold)}."
                ),
                current_ensemble_alerts,
            ),
            encoding="utf-8",
        )
        if ensemble_alerts_history_path is not None:
            ensemble_history_file = Path(ensemble_alerts_history_path)
            ensemble_history_file.parent.mkdir(parents=True, exist_ok=True)
            all_ensemble_alerts.to_csv(ensemble_history_file, index=False)
            ensemble_alert_history_written = str(ensemble_history_file)
        ensemble_alerts_csv_written = str(ensemble_csv)
        ensemble_alerts_html_written = str(ensemble_html)
        ensemble_alerts_count = int(len(current_ensemble_alerts))
        ensemble_alerts_history_count = int(len(all_ensemble_alerts))

    all_alerts = compute_drift_alert_history(
        history,
        lookback_runs=int(lookback_runs),
        max_gap_drop=float(max_gap_drop),
        max_rank_worsening=float(max_rank_worsening),
        min_active_minus_base_gap=float(min_active_minus_base_gap),
    )
    alerts = all_alerts[all_alerts["run_timestamp_utc"].astype(str) == str(run_time)].copy()

    alerts_csv = Path(drift_alerts_csv_path)
    alerts_html = Path(drift_alerts_html_path)
    alerts_csv.parent.mkdir(parents=True, exist_ok=True)
    alerts_html.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(alerts_csv, index=False)
    note = (
        "Alerts fire when active-profile degradation exceeds configured thresholds "
        f"(lookback_runs={int(lookback_runs)}, max_gap_drop={float(max_gap_drop)}, "
        f"max_rank_worsening={float(max_rank_worsening)}, "
        f"min_active_minus_base_gap={float(min_active_minus_base_gap)})."
    )
    alerts_html.write_text(
        _html_table("Profile Drift Alerts", note, alerts),
        encoding="utf-8",
    )
    alert_history_written = None
    if drift_alerts_history_path is not None:
        alert_history = Path(drift_alerts_history_path)
        alert_history.parent.mkdir(parents=True, exist_ok=True)
        all_alerts.to_csv(alert_history, index=False)
        alert_history_written = str(alert_history)

    ticker_summary_written_csv = None
    ticker_summary_written_html = None
    ticker_status_counts: dict[str, int] = {}
    if ticker_summary_csv_path is not None or ticker_summary_html_path is not None:
        if ticker_summary_csv_path is None or ticker_summary_html_path is None:
            raise ValueError(
                "ticker_summary_csv_path and ticker_summary_html_path must be provided together."
            )
        ticker_summary = _build_ticker_summary(snapshot, alerts, history)
        summary_csv = Path(ticker_summary_csv_path)
        summary_html = Path(ticker_summary_html_path)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_html.parent.mkdir(parents=True, exist_ok=True)
        ticker_summary.to_csv(summary_csv, index=False)
        summary_html.write_text(
            _html_table(
                "Ticker Health Summary",
                "Plain-language status per ticker from active profile health and current run alerts.",
                ticker_summary,
            ),
            encoding="utf-8",
        )
        ticker_summary_written_csv = str(summary_csv)
        ticker_summary_written_html = str(summary_html)
        ticker_status_counts = {
            str(status): int(count)
            for status, count in ticker_summary["status"].value_counts().to_dict().items()
        }

    return {
        "run_timestamp_utc": run_time,
        "health_snapshot_path": str(snapshot_path),
        "health_history_path": str(Path(health_history_path)),
        "drift_alerts_csv_path": str(alerts_csv),
        "drift_alerts_html_path": str(alerts_html),
        "drift_alerts_history_path": alert_history_written,
        "ticker_summary_csv_path": ticker_summary_written_csv,
        "ticker_summary_html_path": ticker_summary_written_html,
        "ticker_status_counts": ticker_status_counts,
        "symbols_in_snapshot": int(len(snapshot)),
        "history_rows": int(len(history)),
        "alerts_count": int(len(alerts)),
        "alerts_history_count": int(len(all_alerts)),
        "ensemble_alerts_csv_path": ensemble_alerts_csv_written,
        "ensemble_alerts_html_path": ensemble_alerts_html_written,
        "ensemble_alerts_history_path": ensemble_alert_history_written,
        "ensemble_alerts_count": ensemble_alerts_count,
        "ensemble_alerts_history_count": ensemble_alerts_history_count,
        "ensemble_v3_summary": ensemble_summary,
        "ensemble_effective_selection_state_path": effective_selection_state_written,
    }

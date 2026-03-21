from __future__ import annotations

from typing import Any

import pandas as pd


def _format_pct_points(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:+.2f}%"


def _reason_to_label(reason: str) -> str:
    mapping = {
        "gap_drop_exceeded": "recent performance drop",
        "rank_worsening_exceeded": "ranking slipped",
        "active_below_base_threshold": "current setup is trailing the base setup",
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
        "stable": "Market conditions look fairly calm.",
        "calm": "Market conditions look fairly calm.",
        "trending": "Market conditions look directional right now.",
        "high_vol": "Market conditions look choppy and higher-volatility than usual.",
        "turbulent": "Market conditions look choppy and higher-volatility than usual.",
    }
    return mapping.get(cleaned, f"Market conditions: {cleaned.replace('_', ' ')}.")


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
        recommendation_subtitle = "results are weak right now"
        recommended_action = (
            "Leave this ticker alone for now until several more updates show a clearer recovery."
        )
    elif has_alert:
        recommendation = "Caution"
        recommendation_subtitle = "some good signs, but warnings are active"
        recommended_action = (
            "Keep this on the watchlist, but wait for the warning signs to clear before trusting it more."
        )
    elif strong_setup:
        recommendation = "Bullish setup"
        recommendation_subtitle = "looking strong and steady"
        recommended_action = (
            "This is one of the stronger tickers right now: it has been holding up well and beating the simpler alternatives."
        )
    elif constructive_setup:
        recommendation = "Improving trend"
        recommendation_subtitle = "improving, but not proven yet"
        recommended_action = (
            "This is moving in the right direction, but it still needs a bit more consistency before it becomes a top pick."
        )
    elif mixed_setup:
        recommendation = "Mixed trend"
        recommendation_subtitle = "the picture is still mixed"
        recommended_action = (
            "Treat this as a watchlist ticker for now rather than something to lean on heavily."
        )
    else:
        recommendation = "Avoid for now"
        recommendation_subtitle = "results are weak right now"
        recommended_action = (
            "Wait for stronger and more consistent daily results before relying on this ticker."
        )

    if runs_observed <= 1:
        evidence_text = "This view is based on only 1 recorded run so far."
    else:
        evidence_text = (
            f"Over the last {runs_observed} runs, it stayed ahead of buy-and-hold in "
            f"{positive_runs}/{runs_observed} runs."
        )
    if positive_streak >= 2:
        stability_text = f"It has stayed positive for {positive_streak} runs in a row."
    elif negative_streak >= 2:
        stability_text = f"It has stayed negative for {negative_streak} runs in a row."
    elif stability_label == "Early":
        stability_text = "There is still only a small amount of recent evidence."
    elif stability_label == "Stable":
        stability_text = "Recent results have been steady rather than jumpy."
    elif stability_label == "Building":
        stability_text = "Recent results are positive, but still building proof."
    elif stability_label == "Weakening":
        stability_text = "Recent results have softened."
    else:
        stability_text = "Recent results have been choppy."

    summary_parts = [
        f"{symbol} is {_comparison_phrase('buy-and-hold', active_gap)}.",
        f"It is {_comparison_phrase(f'your {base_profile_text} setup', active_minus_base)}.",
        evidence_text,
        stability_text,
    ]
    if selected_profile != active_profile:
        summary_parts.append(
            f"The dashboard is currently using {active_profile} instead of the originally selected {selected_profile} setup."
        )
    elif promoted:
        summary_parts.append(f"{active_profile} became the live setup on this run.")
    else:
        summary_parts.append(f"{active_profile} is still the live setup.")
    if has_alert:
        summary_parts.append(f"Current issues: {reason_labels}.")
    else:
        summary_parts.append("There are no active warning flags right now.")
    if regime_note:
        summary_parts.append(regime_note)
    if recent_avg_gap is not None:
        summary_parts.append(
            f"Average recent edge over the last {max(recent_window, 1)} run(s): {_format_pct_points(recent_avg_gap)}."
        )

    if severe_alert or active_gap < 0 or negative_streak >= 2:
        what_to_watch = (
            "Watch for this ticker to get back ahead of buy-and-hold, rebuild a positive streak, and clear its warning flags."
        )
    elif has_alert:
        what_to_watch = (
            "Watch the next update closely: the warning signs need to clear before confidence improves."
        )
    elif runs_observed < 3 and active_gap > 0:
        what_to_watch = (
            "Watch for a few more solid runs before treating this as fully proven."
        )
    elif recent_gap_std is not None and float(recent_gap_std) > 0.015:
        what_to_watch = (
            "Watch whether the recent results become steadier, because the edge is still moving around from run to run."
        )
    elif active_gap > 0 and (active_minus_base is None or active_minus_base >= 0):
        what_to_watch = (
            "Watch whether it stays ahead of buy-and-hold over the next few updates."
        )
    else:
        what_to_watch = (
            "Watch for clearer separation from the base setup before upgrading this ticker."
        )

    if confidence == "Low" and not has_alert and active_gap > 0:
        what_to_watch += " Confidence is still limited, so treat this as early rather than proven."

    return {
        "recommendation": recommendation,
        "recommendation_subtitle": recommendation_subtitle,
        "recommended_action": recommended_action,
        "summary": " ".join(summary_parts),
        "what_to_watch": what_to_watch,
    }


def build_ticker_summary(snapshot: pd.DataFrame, alerts: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
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
            float(row["active_minus_base_gap"]) if pd.notna(row["active_minus_base_gap"]) else None
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
                "symbol_category": str(row.get("symbol_category")).strip().lower()
                if pd.notna(row.get("symbol_category"))
                else None,
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

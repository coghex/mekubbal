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


def _build_ticker_summary(snapshot: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    alert_map: dict[str, list[str]] = {}
    if not alerts.empty:
        for _, row in alerts.iterrows():
            symbol = str(row["symbol"]).upper()
            raw = str(row.get("reasons", ""))
            reasons = [value.strip() for value in raw.split(";") if value.strip()]
            alert_map[symbol] = reasons

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

        if has_alert and (
            "active_below_base_threshold" in reasons
            or (active_minus_base is not None and active_minus_base < 0)
        ):
            status = "Critical"
            recommended_action = "Review rollback recommendation and consider reverting to base."
        elif has_alert:
            status = "Watch"
            recommended_action = "Monitor next run and re-check promotion/monitor thresholds."
        else:
            status = "Healthy"
            recommended_action = "Keep current active profile."

        reason_labels = ", ".join(_reason_to_label(value) for value in reasons) if reasons else "none"
        base_profile = row.get("base_profile")
        base_profile_text = str(base_profile) if pd.notna(base_profile) else "base"
        summary_text = (
            f"{active_profile} vs buy-and-hold {_format_pct_points(active_gap)}; "
            f"vs {base_profile_text} {_format_pct_points(active_minus_base)}."
        )
        if selected_profile != active_profile:
            summary_text += f" Ensemble override from {selected_profile} to {active_profile}."
        rows.append(
            {
                "symbol": symbol,
                "status": status,
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
                "promoted_this_run": bool(promoted),
                "alert_reasons": reason_labels,
                "recommended_action": recommended_action,
                "summary": summary_text,
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
        ticker_summary = _build_ticker_summary(snapshot, alerts)
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

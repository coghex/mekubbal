from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


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
) -> pd.DataFrame:
    required = {"symbol", "profile", "symbol_rank", "avg_equity_gap"}
    missing = sorted(required - set(symbol_summary.columns))
    if missing:
        raise ValueError(f"Profile symbol summary missing required columns: {missing}")
    if symbol_summary.empty:
        raise ValueError("Profile symbol summary is empty.")

    active_profiles = selection_state.get("active_profiles", {})
    if not isinstance(active_profiles, dict):
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

    frame = symbol_summary.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["symbol_rank"] = pd.to_numeric(frame["symbol_rank"], errors="coerce")
    frame["avg_equity_gap"] = pd.to_numeric(frame["avg_equity_gap"], errors="coerce")
    frame = frame.dropna(subset=["symbol_rank", "avg_equity_gap"])

    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol"):
        sorted_group = group.sort_values("symbol_rank").reset_index(drop=True)
        by_profile = {str(row["profile"]): row for _, row in sorted_group.iterrows()}
        active = str(active_profiles.get(symbol) or "")
        if not active or active not in by_profile:
            active = str(sorted_group.iloc[0]["profile"])
        active_row = by_profile[active]
        base_row = by_profile.get(base_profile)
        candidate_row = by_profile.get(candidate_profile)
        decision = by_symbol_decision.get(symbol, {})

        base_gap = float(base_row["avg_equity_gap"]) if base_row is not None else None
        active_gap = float(active_row["avg_equity_gap"])
        rows.append(
            {
                "run_timestamp_utc": run_timestamp_utc,
                "symbol": symbol,
                "active_profile": active,
                "active_rank": int(active_row["symbol_rank"]),
                "active_gap": active_gap,
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
        latest = ordered.iloc[-1]
        baseline = ordered.iloc[-(lookback_runs + 1) : -1]
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


def run_profile_monitor(
    *,
    profile_symbol_summary_path: str | Path,
    selection_state_path: str | Path,
    health_snapshot_path: str | Path,
    health_history_path: str | Path,
    drift_alerts_csv_path: str | Path,
    drift_alerts_html_path: str | Path,
    lookback_runs: int = 3,
    max_gap_drop: float = 0.03,
    max_rank_worsening: float = 0.75,
    min_active_minus_base_gap: float = -0.01,
    run_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    summary_path = Path(profile_symbol_summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Profile symbol summary does not exist: {summary_path}")
    symbol_summary = pd.read_csv(summary_path)
    selection_state = _load_profile_selection_state(selection_state_path)
    run_time = run_timestamp_utc or _now_utc_iso()

    snapshot = _build_active_snapshot(
        symbol_summary,
        selection_state,
        run_timestamp_utc=run_time,
    )
    snapshot_path = Path(health_snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(snapshot_path, index=False)

    history = _append_history(snapshot, health_history_path)
    alerts = _alert_rows(
        history,
        lookback_runs=int(lookback_runs),
        max_gap_drop=float(max_gap_drop),
        max_rank_worsening=float(max_rank_worsening),
        min_active_minus_base_gap=float(min_active_minus_base_gap),
    )

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

    return {
        "run_timestamp_utc": run_time,
        "health_snapshot_path": str(snapshot_path),
        "health_history_path": str(Path(health_history_path)),
        "drift_alerts_csv_path": str(alerts_csv),
        "drift_alerts_html_path": str(alerts_html),
        "symbols_in_snapshot": int(len(snapshot)),
        "history_rows": int(len(history)),
        "alerts_count": int(len(alerts)),
    }

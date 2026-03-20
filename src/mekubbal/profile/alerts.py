from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def append_history(snapshot: pd.DataFrame, history_path: str | Path) -> pd.DataFrame:
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

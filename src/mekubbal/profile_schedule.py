from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import tomllib

from mekubbal.profile_matrix import load_profile_matrix_config, run_profile_matrix
from mekubbal.profile_monitor import run_profile_monitor
from mekubbal.profile_rollback import run_profile_rollback
from mekubbal.visualization import render_product_dashboard


def _default_profile_schedule_config() -> dict[str, Any]:
    return {
        "schedule": {
            "matrix_config": "configs/profile-matrix.toml",
            "symbols": [],
            "health_snapshot_path": "reports/active_profile_health.csv",
            "health_history_path": "reports/active_profile_health_history.csv",
            "drift_alerts_csv_path": "reports/profile_drift_alerts.csv",
            "drift_alerts_html_path": "reports/profile_drift_alerts.html",
            "drift_alerts_history_path": "reports/profile_drift_alerts_history.csv",
            "ensemble_alerts_csv_path": "reports/profile_ensemble_alerts.csv",
            "ensemble_alerts_html_path": "reports/profile_ensemble_alerts.html",
            "ensemble_alerts_history_path": "reports/profile_ensemble_alerts_history.csv",
            "ticker_summary_csv_path": "reports/ticker_health_summary.csv",
            "ticker_summary_html_path": "reports/ticker_health_summary.html",
            "product_dashboard_path": "reports/product_dashboard.html",
            "product_dashboard_title": "Mekubbal Market Pulse",
            "summary_json_path": "reports/profile_schedule_summary.json",
            "ops_journal_csv_path": "reports/profile_ops_journal.csv",
            "ops_digest_html_path": "reports/profile_ops_digest.html",
            "ops_digest_lookback_runs": 14,
        },
        "monitor": {
            "lookback_runs": 3,
            "max_gap_drop": 0.03,
            "max_rank_worsening": 0.75,
            "min_active_minus_base_gap": -0.01,
            "ensemble_low_confidence_threshold": 0.55,
        },
        "rollback": {
            "enabled": False,
            "rollback_state_path": "reports/profile_rollback_state.json",
            "min_consecutive_alert_runs": 2,
            "rollback_on_drift_alerts": True,
            "rollback_on_ensemble_events": False,
            "min_consecutive_ensemble_event_runs": 2,
            "rollback_profile": "base",
            "apply_rollback": False,
        },
        "ensemble_v3": {
            "enabled": False,
            "lookback_runs": 3,
            "min_regime_confidence": 0.55,
            "rank_weight": 0.55,
            "gap_weight": 0.45,
            "significance_bonus": 0.1,
            "fallback_profile": "base",
            "high_vol_gap_std_threshold": 0.03,
            "high_vol_rank_std_threshold": 0.75,
            "trending_min_gap_improvement": 0.01,
            "trending_min_rank_improvement": 0.25,
            "decision_csv_path": "reports/profile_ensemble_decisions.csv",
            "decision_history_path": "reports/profile_ensemble_history.csv",
            "effective_selection_state_path": "reports/profile_selection_state_ensemble.json",
            "profile_weights": {},
            "regime_multipliers": {},
        },
        "shadow": {
            "enabled": False,
            "production_state_path": "",
            "shadow_state_path": "reports/profile_selection_state_shadow.json",
            "window_runs": 5,
            "min_match_ratio": 1.0,
            "apply_promotion_after_shadow": False,
            "comparison_csv_path": "reports/profile_shadow_comparison.csv",
            "comparison_html_path": "reports/profile_shadow_comparison.html",
            "comparison_history_path": "reports/profile_shadow_comparison_history.csv",
            "gate_json_path": "reports/profile_shadow_gate.json",
            "suggestion_json_path": "reports/profile_shadow_suggestions.json",
            "suggestion_html_path": "reports/profile_shadow_suggestions.html",
            "suggestion_min_history_runs": 8,
            "health_snapshot_path": "reports/shadow_active_profile_health.csv",
            "health_history_path": "reports/shadow_active_profile_health_history.csv",
            "drift_alerts_csv_path": "reports/shadow_profile_drift_alerts.csv",
            "drift_alerts_html_path": "reports/shadow_profile_drift_alerts.html",
            "drift_alerts_history_path": "reports/shadow_profile_drift_alerts_history.csv",
            "ensemble_alerts_csv_path": "reports/shadow_profile_ensemble_alerts.csv",
            "ensemble_alerts_html_path": "reports/shadow_profile_ensemble_alerts.html",
            "ensemble_alerts_history_path": "reports/shadow_profile_ensemble_alerts_history.csv",
            "ticker_summary_csv_path": "reports/shadow_ticker_health_summary.csv",
            "ticker_summary_html_path": "reports/shadow_ticker_health_summary.html",
            "ensemble_decision_csv_path": "reports/shadow_profile_ensemble_decisions.csv",
            "ensemble_history_path": "reports/shadow_profile_ensemble_history.csv",
            "effective_selection_state_path": "reports/profile_selection_state_shadow_ensemble.json",
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_config_dir = (base_dir / path).resolve()
    if from_config_dir.exists():
        return from_config_dir
    return path.resolve()


def _parse_symbols(values: list[Any]) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("schedule.symbols must be a list.")
    seen: set[str] = set()
    symbols: list[str] = []
    for raw in values:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _validate_profile_schedule_config(config: dict[str, Any], *, config_dir: Path) -> None:
    schedule = config["schedule"]
    monitor = config["monitor"]
    rollback = config["rollback"]
    ensemble = config["ensemble_v3"]
    shadow = config["shadow"]
    if not schedule.get("matrix_config"):
        raise ValueError("schedule.matrix_config is required.")
    matrix_config = _resolve_path(config_dir, str(schedule["matrix_config"]))
    if not matrix_config.exists():
        raise FileNotFoundError(f"Matrix config does not exist: {matrix_config}")
    config["schedule"]["symbols"] = _parse_symbols(schedule.get("symbols", []))
    if int(monitor["lookback_runs"]) < 1:
        raise ValueError("monitor.lookback_runs must be >= 1.")
    if float(monitor["max_gap_drop"]) < 0:
        raise ValueError("monitor.max_gap_drop must be >= 0.")
    if float(monitor["max_rank_worsening"]) < 0:
        raise ValueError("monitor.max_rank_worsening must be >= 0.")
    if float(monitor["ensemble_low_confidence_threshold"]) < 0 or float(
        monitor["ensemble_low_confidence_threshold"]
    ) > 1:
        raise ValueError("monitor.ensemble_low_confidence_threshold must be in [0, 1].")
    if int(rollback["min_consecutive_alert_runs"]) < 1:
        raise ValueError("rollback.min_consecutive_alert_runs must be >= 1.")
    if int(rollback["min_consecutive_ensemble_event_runs"]) < 1:
        raise ValueError("rollback.min_consecutive_ensemble_event_runs must be >= 1.")
    if bool(rollback["enabled"]) and not bool(rollback["rollback_on_drift_alerts"]) and not bool(
        rollback["rollback_on_ensemble_events"]
    ):
        raise ValueError("rollback must enable rollback_on_drift_alerts or rollback_on_ensemble_events.")
    if int(ensemble["lookback_runs"]) < 1:
        raise ValueError("ensemble_v3.lookback_runs must be >= 1.")
    if float(ensemble["min_regime_confidence"]) < 0 or float(ensemble["min_regime_confidence"]) > 1:
        raise ValueError("ensemble_v3.min_regime_confidence must be in [0, 1].")
    if float(ensemble["rank_weight"]) < 0 or float(ensemble["gap_weight"]) < 0:
        raise ValueError("ensemble_v3.rank_weight and gap_weight must be >= 0.")
    if float(ensemble["rank_weight"]) + float(ensemble["gap_weight"]) <= 0:
        raise ValueError("ensemble_v3.rank_weight and gap_weight cannot both be zero.")
    if float(ensemble["significance_bonus"]) < 0:
        raise ValueError("ensemble_v3.significance_bonus must be >= 0.")
    if float(ensemble["high_vol_gap_std_threshold"]) <= 0:
        raise ValueError("ensemble_v3.high_vol_gap_std_threshold must be > 0.")
    if float(ensemble["high_vol_rank_std_threshold"]) <= 0:
        raise ValueError("ensemble_v3.high_vol_rank_std_threshold must be > 0.")
    if float(ensemble["trending_min_gap_improvement"]) < 0:
        raise ValueError("ensemble_v3.trending_min_gap_improvement must be >= 0.")
    if float(ensemble["trending_min_rank_improvement"]) < 0:
        raise ValueError("ensemble_v3.trending_min_rank_improvement must be >= 0.")
    if not isinstance(ensemble.get("profile_weights"), dict):
        raise ValueError("ensemble_v3.profile_weights must be a TOML table/object.")
    if not isinstance(ensemble.get("regime_multipliers"), dict):
        raise ValueError("ensemble_v3.regime_multipliers must be a TOML table/object.")
    if int(shadow["window_runs"]) < 1:
        raise ValueError("shadow.window_runs must be >= 1.")
    min_match_ratio = float(shadow["min_match_ratio"])
    if min_match_ratio < 0 or min_match_ratio > 1:
        raise ValueError("shadow.min_match_ratio must be in [0, 1].")
    if int(shadow["suggestion_min_history_runs"]) < 3:
        raise ValueError("shadow.suggestion_min_history_runs must be >= 3.")
    if int(schedule["ops_digest_lookback_runs"]) < 1:
        raise ValueError("schedule.ops_digest_lookback_runs must be >= 1.")


def _resolve_relative_to(base: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _append_history_rows(rows: pd.DataFrame, history_path: Path) -> pd.DataFrame:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        existing = pd.read_csv(history_path)
        merged = pd.concat([existing, rows], ignore_index=True)
    else:
        merged = rows.copy()
    merged.to_csv(history_path, index=False)
    return merged


def _html_table(title: str, note: str, frame: pd.DataFrame) -> str:
    safe_note = note.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    table_html = frame.to_html(index=False, border=0, escape=False, classes="dataframe")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:2rem;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:0.5rem;text-align:left;font-size:0.9rem;}"
        "th{background:#f6f8fa;}"
        "</style></head><body>"
        f"<h1>{title}</h1><p>{safe_note}</p>{table_html}</body></html>"
    )


def _build_shadow_comparison(
    *,
    run_timestamp_utc: str,
    production_snapshot_path: Path,
    shadow_snapshot_path: Path,
    comparison_csv_path: Path,
    comparison_history_path: Path,
    comparison_html_path: Path,
    gate_json_path: Path,
    window_runs: int,
    min_match_ratio: float,
) -> dict[str, Any]:
    production = pd.read_csv(production_snapshot_path)
    shadow = pd.read_csv(shadow_snapshot_path)
    required = {
        "symbol",
        "selected_profile",
        "active_profile",
        "active_profile_source",
        "active_rank",
        "active_gap",
    }
    missing_prod = sorted(required - set(production.columns))
    if missing_prod:
        raise ValueError(f"Production snapshot missing required columns: {missing_prod}")
    missing_shadow = sorted(required - set(shadow.columns))
    if missing_shadow:
        raise ValueError(f"Shadow snapshot missing required columns: {missing_shadow}")

    production = production[list(required)].copy()
    shadow = shadow[list(required)].copy()
    production = production.rename(
        columns={
            "selected_profile": "production_selected_profile",
            "active_profile": "production_active_profile",
            "active_profile_source": "production_active_profile_source",
            "active_rank": "production_active_rank",
            "active_gap": "production_active_gap",
        }
    )
    shadow = shadow.rename(
        columns={
            "selected_profile": "shadow_selected_profile",
            "active_profile": "shadow_active_profile",
            "active_profile_source": "shadow_active_profile_source",
            "active_rank": "shadow_active_rank",
            "active_gap": "shadow_active_gap",
        }
    )
    merged = production.merge(shadow, on="symbol", how="inner").sort_values("symbol").reset_index(drop=True)
    if merged.empty:
        raise ValueError("Shadow comparison found no overlapping symbols.")

    merged.insert(0, "run_timestamp_utc", run_timestamp_utc)
    merged["active_profile_match"] = (
        merged["production_active_profile"].astype(str) == merged["shadow_active_profile"].astype(str)
    )
    merged["active_rank_delta"] = pd.to_numeric(merged["shadow_active_rank"], errors="coerce") - pd.to_numeric(
        merged["production_active_rank"], errors="coerce"
    )
    merged["active_gap_delta"] = pd.to_numeric(merged["shadow_active_gap"], errors="coerce") - pd.to_numeric(
        merged["production_active_gap"], errors="coerce"
    )

    comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(comparison_csv_path, index=False)
    comparison_history = _append_history_rows(merged, comparison_history_path)

    gate_rows: list[dict[str, Any]] = []
    failing_symbols: list[str] = []
    for symbol, group in comparison_history.groupby("symbol"):
        ordered = group.sort_values("run_timestamp_utc")
        tail = ordered.tail(int(window_runs))
        run_count = int(len(tail))
        match_flags = (
            tail["active_profile_match"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
            .fillna(0.0)
        )
        match_ratio = float(match_flags.mean())
        has_window = run_count >= int(window_runs)
        passed = has_window and match_ratio >= float(min_match_ratio)
        if not passed:
            if not has_window:
                failing_symbols.append(f"{symbol}:insufficient_runs({run_count}/{int(window_runs)})")
            else:
                failing_symbols.append(
                    f"{symbol}:match_ratio({match_ratio:.3f}<{float(min_match_ratio):.3f})"
                )
        gate_rows.append(
            {
                "symbol": symbol,
                "window_runs_required": int(window_runs),
                "runs_in_window": run_count,
                "match_ratio": match_ratio,
                "min_match_ratio": float(min_match_ratio),
                "gate_passed": bool(passed),
            }
        )
    gate_frame = pd.DataFrame(gate_rows).sort_values("symbol").reset_index(drop=True)
    overall_gate_passed = bool(gate_frame["gate_passed"].all()) if not gate_frame.empty else False

    comparison_html_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_html_path.write_text(
        _html_table(
            "Shadow vs Production Comparison",
            (
                "Current-run active profile comparison between production state and shadow state. "
                "active_rank_delta = shadow - production; active_gap_delta = shadow - production."
            ),
            merged,
        ),
        encoding="utf-8",
    )
    gate_json_path.parent.mkdir(parents=True, exist_ok=True)
    gate_payload = {
        "run_timestamp_utc": run_timestamp_utc,
        "window_runs": int(window_runs),
        "min_match_ratio": float(min_match_ratio),
        "overall_gate_passed": overall_gate_passed,
        "failing_symbols": failing_symbols,
        "symbols": gate_frame.to_dict(orient="records"),
    }
    gate_json_path.write_text(json.dumps(gate_payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "comparison_csv_path": str(comparison_csv_path),
        "comparison_html_path": str(comparison_html_path),
        "comparison_history_path": str(comparison_history_path),
        "comparison_rows": int(len(merged)),
        "comparison_history_rows": int(len(comparison_history)),
        "gate_json_path": str(gate_json_path),
        "overall_gate_passed": overall_gate_passed,
        "failing_symbols": failing_symbols,
    }


def _suggest_shadow_thresholds(
    *,
    comparison_history_path: Path,
    suggestion_json_path: Path,
    suggestion_html_path: Path,
    min_history_runs: int,
) -> dict[str, Any]:
    required = {"run_timestamp_utc", "symbol", "active_profile_match"}
    if not comparison_history_path.exists():
        payload = {
            "accepted": False,
            "reasons": [f"history_missing:{comparison_history_path}"],
            "recommended_window_runs": None,
            "recommended_min_match_ratio": None,
            "grid_rows": 0,
            "symbol_run_counts": {},
        }
        suggestion_json_path.parent.mkdir(parents=True, exist_ok=True)
        suggestion_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        suggestion_html_path.parent.mkdir(parents=True, exist_ok=True)
        suggestion_html_path.write_text(
            _html_table("Shadow Threshold Suggestions", "No comparison history available.", pd.DataFrame()),
            encoding="utf-8",
        )
        return {
            "suggestion_json_path": str(suggestion_json_path),
            "suggestion_html_path": str(suggestion_html_path),
            "accepted": False,
            "reasons": payload["reasons"],
        }

    history = pd.read_csv(comparison_history_path)
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Shadow comparison history missing required columns: {missing}")

    frame = history.copy()
    frame["run_timestamp_utc"] = frame["run_timestamp_utc"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["active_profile_match"] = (
        frame["active_profile_match"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
    )
    frame = frame.dropna(subset=["active_profile_match"])
    if frame.empty:
        raise ValueError("Shadow comparison history has no valid active_profile_match rows.")
    frame["active_profile_match"] = frame["active_profile_match"].astype(int)

    run_counts = {
        str(symbol): int(count)
        for symbol, count in frame.groupby("symbol").size().to_dict().items()
    }
    min_symbol_runs = min(run_counts.values()) if run_counts else 0
    reasons: list[str] = []
    if min_symbol_runs < int(min_history_runs):
        reasons.append(f"insufficient_history_runs_per_symbol:{min_symbol_runs}<{int(min_history_runs)}")

    max_window = max(2, min(10, min_symbol_runs - 1))
    windows = list(range(2, max_window + 1))
    thresholds = [round(value, 2) for value in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]]
    rows: list[dict[str, Any]] = []
    for window in windows:
        for threshold in thresholds:
            total = 0
            pass_predictions = 0
            pass_correct = 0
            fail_predictions = 0
            fail_correct = 0
            for _, group in frame.groupby("symbol"):
                ordered = group.sort_values("run_timestamp_utc").reset_index(drop=True)
                matches = ordered["active_profile_match"].tolist()
                if len(matches) <= window:
                    continue
                for idx in range(window - 1, len(matches) - 1):
                    ratio = float(sum(matches[idx - window + 1 : idx + 1])) / float(window)
                    predict_pass = ratio >= float(threshold)
                    next_match = int(matches[idx + 1])
                    total += 1
                    if predict_pass:
                        pass_predictions += 1
                        if next_match == 1:
                            pass_correct += 1
                    else:
                        fail_predictions += 1
                        if next_match == 0:
                            fail_correct += 1
            if total == 0:
                continue
            pass_rate = float(pass_predictions) / float(total)
            pass_precision = float(pass_correct) / float(pass_predictions) if pass_predictions > 0 else 0.0
            fail_precision = float(fail_correct) / float(fail_predictions) if fail_predictions > 0 else 0.0
            score = (
                0.65 * pass_precision
                + 0.35 * fail_precision
                - 0.1 * abs(pass_rate - 0.5)
                + 0.02 * min(total, 200) / 200.0
            )
            rows.append(
                {
                    "window_runs": int(window),
                    "min_match_ratio": float(threshold),
                    "samples": int(total),
                    "pass_rate": pass_rate,
                    "pass_precision_next_run_match": pass_precision,
                    "fail_precision_next_run_mismatch": fail_precision,
                    "score": score,
                }
            )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        reasons.append("insufficient_comparison_samples_for_grid")
    else:
        ranking = ranking.sort_values(
            ["score", "pass_precision_next_run_match", "samples", "window_runs", "min_match_ratio"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))

    accepted = (not reasons) and (not ranking.empty)
    best = ranking.iloc[0].to_dict() if accepted else {}
    payload = {
        "accepted": bool(accepted),
        "reasons": reasons,
        "symbol_run_counts": run_counts,
        "min_symbol_runs": int(min_symbol_runs),
        "minimum_required_history_runs": int(min_history_runs),
        "recommended_window_runs": int(best["window_runs"]) if accepted else None,
        "recommended_min_match_ratio": float(best["min_match_ratio"]) if accepted else None,
        "recommendation_metrics": (
            {
                "samples": int(best["samples"]),
                "pass_rate": float(best["pass_rate"]),
                "pass_precision_next_run_match": float(best["pass_precision_next_run_match"]),
                "fail_precision_next_run_mismatch": float(best["fail_precision_next_run_mismatch"]),
                "score": float(best["score"]),
            }
            if accepted
            else {}
        ),
        "grid_rows": int(len(ranking)),
    }
    suggestion_json_path.parent.mkdir(parents=True, exist_ok=True)
    suggestion_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    suggestion_html_path.parent.mkdir(parents=True, exist_ok=True)
    top_rows = ranking.head(20) if not ranking.empty else pd.DataFrame()
    note = (
        "Auto-suggested shadow gate thresholds ranked by next-run prediction quality from historical "
        "shadow-vs-production agreement."
    )
    suggestion_html_path.write_text(
        _html_table("Shadow Threshold Suggestions", note, top_rows),
        encoding="utf-8",
    )
    return {
        "suggestion_json_path": str(suggestion_json_path),
        "suggestion_html_path": str(suggestion_html_path),
        "accepted": bool(accepted),
        "reasons": reasons,
        "recommended_window_runs": payload["recommended_window_runs"],
        "recommended_min_match_ratio": payload["recommended_min_match_ratio"],
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _update_ops_journal(
    *,
    run_timestamp_utc: str,
    journal_csv_path: Path,
    digest_html_path: Path,
    digest_lookback_runs: int,
    monitor_summary: dict[str, Any],
    shadow_summary: dict[str, Any] | None,
    rollback_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    shadow_gate = (
        shadow_summary.get("comparison_summary", {})
        if isinstance(shadow_summary, dict)
        else {}
    )
    row = pd.DataFrame(
        [
            {
                "run_timestamp_utc": str(run_timestamp_utc),
                "symbols_in_snapshot": _safe_int(monitor_summary.get("symbols_in_snapshot")),
                "health_history_rows": _safe_int(monitor_summary.get("history_rows")),
                "drift_alerts_count": _safe_int(monitor_summary.get("alerts_count")),
                "ensemble_alerts_count": _safe_int(monitor_summary.get("ensemble_alerts_count")),
                "shadow_enabled": bool(isinstance(shadow_summary, dict)),
                "shadow_gate_passed": (
                    bool(shadow_gate.get("overall_gate_passed"))
                    if isinstance(shadow_summary, dict)
                    else None
                ),
                "shadow_failing_symbols": (
                    ";".join(str(value) for value in shadow_gate.get("failing_symbols", []))
                    if isinstance(shadow_summary, dict)
                    else ""
                ),
                "shadow_promotion_applied": (
                    bool(shadow_summary.get("promotion_applied"))
                    if isinstance(shadow_summary, dict)
                    else False
                ),
                "rollback_enabled": bool(isinstance(rollback_summary, dict)),
                "rollback_recommended_count": (
                    _safe_int(rollback_summary.get("rollback_recommended_count"))
                    if isinstance(rollback_summary, dict)
                    else 0
                ),
                "rollback_applied_count": (
                    _safe_int(rollback_summary.get("rollback_applied_count"))
                    if isinstance(rollback_summary, dict)
                    else 0
                ),
            }
        ]
    )
    shadow_gate_failed = row["shadow_gate_passed"].map(lambda value: value is False)
    row["attention_needed"] = (
        (row["drift_alerts_count"] > 0)
        | (row["ensemble_alerts_count"] > 0)
        | (row["rollback_recommended_count"] > 0)
        | (row["rollback_applied_count"] > 0)
        | (row["shadow_enabled"].astype(bool) & shadow_gate_failed)
    )
    journal = _append_history_rows(row, journal_csv_path)
    latest = journal.sort_values("run_timestamp_utc").tail(int(digest_lookback_runs)).reset_index(drop=True)
    digest_html_path.parent.mkdir(parents=True, exist_ok=True)
    digest_html_path.write_text(
        _html_table(
            "Profile Ops Digest",
            (
                "Silent operations digest for automatic safeguards. "
                "Use this for periodic review instead of real-time alerts."
            ),
            latest,
        ),
        encoding="utf-8",
    )
    return {
        "journal_csv_path": str(journal_csv_path),
        "digest_html_path": str(digest_html_path),
        "journal_rows": int(len(journal)),
        "digest_rows": int(len(latest)),
        "latest_attention_needed": bool(latest.iloc[-1]["attention_needed"]) if not latest.empty else False,
    }


def load_profile_schedule_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must decode to a TOML table.")
    merged = _deep_merge(deepcopy(_default_profile_schedule_config()), loaded)
    _validate_profile_schedule_config(merged, config_dir=path.parent.resolve())
    return merged


def run_profile_schedule(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    config = load_profile_schedule_config(config_file)
    schedule_cfg = config["schedule"]
    monitor_cfg = config["monitor"]
    rollback_cfg = config["rollback"]
    ensemble_cfg = config["ensemble_v3"]
    shadow_cfg = config["shadow"]

    matrix_config = _resolve_path(config_dir, str(schedule_cfg["matrix_config"]))
    symbols = list(schedule_cfg["symbols"])
    shadow_enabled = bool(shadow_cfg.get("enabled", False))
    shadow_selection_state_path: Path | None = None
    production_selection_state_path: Path | None = None
    matrix_call_kwargs: dict[str, Any] = {
        "symbols_override": symbols if symbols else None,
    }
    if shadow_enabled:
        matrix_loaded = load_profile_matrix_config(matrix_config)
        matrix_output_root_hint = _resolve_relative_to(
            matrix_config.parent.resolve(), str(matrix_loaded["matrix"]["output_root"])
        )
        matrix_promotion_cfg = dict(matrix_loaded["promotion"])
        production_state_raw = str(shadow_cfg.get("production_state_path", "")).strip()
        if production_state_raw:
            production_selection_state_path = _resolve_relative_to(matrix_output_root_hint, production_state_raw)
        else:
            production_selection_state_path = _resolve_relative_to(
                matrix_output_root_hint, str(matrix_promotion_cfg["state_path"])
            )
        if not production_selection_state_path.exists():
            raise FileNotFoundError(
                f"Shadow evaluation requires existing production state: {production_selection_state_path}"
            )
        shadow_selection_state_path = _resolve_relative_to(
            matrix_output_root_hint, str(shadow_cfg["shadow_state_path"])
        )
        matrix_call_kwargs["promotion_override"] = {
            "enabled": True,
            "state_path": str(shadow_selection_state_path),
        }

    matrix_summary = run_profile_matrix(
        matrix_config,
        **matrix_call_kwargs,
    )
    matrix_output_root = Path(str(matrix_summary["output_root"])).resolve()

    if shadow_enabled:
        assert production_selection_state_path is not None
        assert shadow_selection_state_path is not None
        production_selection_state_path = _resolve_relative_to(
            matrix_output_root, str(production_selection_state_path)
        )
        shadow_selection_state_path = _resolve_relative_to(matrix_output_root, str(shadow_selection_state_path))
        if not shadow_selection_state_path.exists():
            profile_selection = matrix_summary.get("profile_selection")
            if isinstance(profile_selection, dict) and profile_selection.get("state_path"):
                shadow_selection_state_path = Path(str(profile_selection["state_path"])).resolve()
        if not shadow_selection_state_path.exists():
            raise ValueError(
                "Shadow evaluation enabled but shadow selection state was not generated by profile matrix."
            )

    selection_state_path = None
    if shadow_enabled:
        selection_state_path = production_selection_state_path
    else:
        profile_selection = matrix_summary.get("profile_selection")
        if isinstance(profile_selection, dict) and profile_selection.get("state_path"):
            selection_state_path = Path(str(profile_selection["state_path"])).resolve()
        else:
            fallback_state = matrix_output_root / "reports" / "profile_selection_state.json"
            if fallback_state.exists():
                selection_state_path = fallback_state
    if selection_state_path is None:
        raise ValueError(
            "Profile selection state not found. Enable promotion in profile-matrix config "
            "or run mekubbal-profile-select before schedule monitoring."
        )

    monitor_summary = run_profile_monitor(
        profile_symbol_summary_path=matrix_summary["symbol_summary_path"],
        selection_state_path=selection_state_path,
        health_snapshot_path=matrix_output_root / str(schedule_cfg["health_snapshot_path"]),
        health_history_path=matrix_output_root / str(schedule_cfg["health_history_path"]),
        drift_alerts_csv_path=matrix_output_root / str(schedule_cfg["drift_alerts_csv_path"]),
        drift_alerts_html_path=matrix_output_root / str(schedule_cfg["drift_alerts_html_path"]),
        drift_alerts_history_path=matrix_output_root / str(schedule_cfg["drift_alerts_history_path"]),
        ensemble_alerts_csv_path=matrix_output_root / str(schedule_cfg["ensemble_alerts_csv_path"]),
        ensemble_alerts_html_path=matrix_output_root / str(schedule_cfg["ensemble_alerts_html_path"]),
        ensemble_alerts_history_path=matrix_output_root / str(
            schedule_cfg["ensemble_alerts_history_path"]
        ),
        ticker_summary_csv_path=matrix_output_root / str(schedule_cfg["ticker_summary_csv_path"]),
        ticker_summary_html_path=matrix_output_root / str(schedule_cfg["ticker_summary_html_path"]),
        lookback_runs=int(monitor_cfg["lookback_runs"]),
        max_gap_drop=float(monitor_cfg["max_gap_drop"]),
        max_rank_worsening=float(monitor_cfg["max_rank_worsening"]),
        min_active_minus_base_gap=float(monitor_cfg["min_active_minus_base_gap"]),
        ensemble_low_confidence_threshold=float(monitor_cfg["ensemble_low_confidence_threshold"]),
        ensemble_v3_config=ensemble_cfg,
        ensemble_decisions_csv_path=matrix_output_root / str(ensemble_cfg["decision_csv_path"]),
        ensemble_history_path=matrix_output_root / str(ensemble_cfg["decision_history_path"]),
        ensemble_effective_selection_state_path=matrix_output_root
        / str(ensemble_cfg["effective_selection_state_path"]),
    )
    shadow_summary = None
    if shadow_enabled:
        assert shadow_selection_state_path is not None
        shadow_monitor_summary = run_profile_monitor(
            profile_symbol_summary_path=matrix_summary["symbol_summary_path"],
            selection_state_path=shadow_selection_state_path,
            health_snapshot_path=matrix_output_root / str(shadow_cfg["health_snapshot_path"]),
            health_history_path=matrix_output_root / str(shadow_cfg["health_history_path"]),
            drift_alerts_csv_path=matrix_output_root / str(shadow_cfg["drift_alerts_csv_path"]),
            drift_alerts_html_path=matrix_output_root / str(shadow_cfg["drift_alerts_html_path"]),
            drift_alerts_history_path=matrix_output_root / str(shadow_cfg["drift_alerts_history_path"]),
            ensemble_alerts_csv_path=matrix_output_root / str(shadow_cfg["ensemble_alerts_csv_path"]),
            ensemble_alerts_html_path=matrix_output_root / str(shadow_cfg["ensemble_alerts_html_path"]),
            ensemble_alerts_history_path=matrix_output_root / str(shadow_cfg["ensemble_alerts_history_path"]),
            ticker_summary_csv_path=matrix_output_root / str(shadow_cfg["ticker_summary_csv_path"]),
            ticker_summary_html_path=matrix_output_root / str(shadow_cfg["ticker_summary_html_path"]),
            lookback_runs=int(monitor_cfg["lookback_runs"]),
            max_gap_drop=float(monitor_cfg["max_gap_drop"]),
            max_rank_worsening=float(monitor_cfg["max_rank_worsening"]),
            min_active_minus_base_gap=float(monitor_cfg["min_active_minus_base_gap"]),
            run_timestamp_utc=monitor_summary["run_timestamp_utc"],
            ensemble_low_confidence_threshold=float(monitor_cfg["ensemble_low_confidence_threshold"]),
            ensemble_v3_config=ensemble_cfg,
            ensemble_decisions_csv_path=matrix_output_root / str(shadow_cfg["ensemble_decision_csv_path"]),
            ensemble_history_path=matrix_output_root / str(shadow_cfg["ensemble_history_path"]),
            ensemble_effective_selection_state_path=matrix_output_root
            / str(shadow_cfg["effective_selection_state_path"]),
        )
        shadow_comparison_summary = _build_shadow_comparison(
            run_timestamp_utc=str(monitor_summary["run_timestamp_utc"]),
            production_snapshot_path=Path(str(monitor_summary["health_snapshot_path"])),
            shadow_snapshot_path=Path(str(shadow_monitor_summary["health_snapshot_path"])),
            comparison_csv_path=matrix_output_root / str(shadow_cfg["comparison_csv_path"]),
            comparison_history_path=matrix_output_root / str(shadow_cfg["comparison_history_path"]),
            comparison_html_path=matrix_output_root / str(shadow_cfg["comparison_html_path"]),
            gate_json_path=matrix_output_root / str(shadow_cfg["gate_json_path"]),
            window_runs=int(shadow_cfg["window_runs"]),
            min_match_ratio=float(shadow_cfg["min_match_ratio"]),
        )
        shadow_suggestion_summary = _suggest_shadow_thresholds(
            comparison_history_path=Path(str(shadow_comparison_summary["comparison_history_path"])),
            suggestion_json_path=matrix_output_root / str(shadow_cfg["suggestion_json_path"]),
            suggestion_html_path=matrix_output_root / str(shadow_cfg["suggestion_html_path"]),
            min_history_runs=int(shadow_cfg["suggestion_min_history_runs"]),
        )
        shadow_promotion_applied = False
        if bool(shadow_cfg["apply_promotion_after_shadow"]) and bool(
            shadow_comparison_summary["overall_gate_passed"]
        ):
            shadow_state = json.loads(shadow_selection_state_path.read_text(encoding="utf-8"))
            if not isinstance(shadow_state, dict):
                raise ValueError("Shadow selection state must decode to a JSON object.")
            metadata = dict(shadow_state.get("shadow_gate", {})) if isinstance(
                shadow_state.get("shadow_gate", {}), dict
            ) else {}
            metadata.update(
                {
                    "applied_at_utc": str(monitor_summary["run_timestamp_utc"]),
                    "window_runs": int(shadow_cfg["window_runs"]),
                    "min_match_ratio": float(shadow_cfg["min_match_ratio"]),
                    "source_state_path": str(shadow_selection_state_path),
                }
            )
            shadow_state["shadow_gate"] = metadata
            selection_state_path.write_text(
                json.dumps(shadow_state, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            shadow_promotion_applied = True
        shadow_summary = {
            "enabled": True,
            "production_state_path": str(selection_state_path),
            "shadow_state_path": str(shadow_selection_state_path),
            "monitor_summary": shadow_monitor_summary,
            "comparison_summary": shadow_comparison_summary,
            "suggestion_summary": shadow_suggestion_summary,
            "promotion_applied": shadow_promotion_applied,
        }

    rollback_summary = None
    if bool(rollback_cfg["enabled"]):
        rollback_selection_state_path: Path = selection_state_path
        if not bool(rollback_cfg["apply_rollback"]):
            ensemble_state = monitor_summary.get("ensemble_effective_selection_state_path")
            if isinstance(ensemble_state, str) and ensemble_state.strip():
                rollback_selection_state_path = Path(ensemble_state).resolve()
        rollback_summary = run_profile_rollback(
            selection_state_path=rollback_selection_state_path,
            health_history_path=monitor_summary["health_history_path"],
            rollback_state_path=matrix_output_root / str(rollback_cfg["rollback_state_path"]),
            lookback_runs=int(monitor_cfg["lookback_runs"]),
            max_gap_drop=float(monitor_cfg["max_gap_drop"]),
            max_rank_worsening=float(monitor_cfg["max_rank_worsening"]),
            min_active_minus_base_gap=float(monitor_cfg["min_active_minus_base_gap"]),
            min_consecutive_alert_runs=int(rollback_cfg["min_consecutive_alert_runs"]),
            rollback_on_drift_alerts=bool(rollback_cfg["rollback_on_drift_alerts"]),
            rollback_on_ensemble_events=bool(rollback_cfg["rollback_on_ensemble_events"]),
            ensemble_alerts_history_path=monitor_summary.get("ensemble_alerts_history_path"),
            min_consecutive_ensemble_event_runs=int(
                rollback_cfg["min_consecutive_ensemble_event_runs"]
            ),
            rollback_profile=rollback_cfg.get("rollback_profile"),
            apply_rollback=bool(rollback_cfg["apply_rollback"]),
            run_timestamp_utc=monitor_summary["run_timestamp_utc"],
        )
    ops_summary = _update_ops_journal(
        run_timestamp_utc=str(monitor_summary["run_timestamp_utc"]),
        journal_csv_path=matrix_output_root / str(schedule_cfg["ops_journal_csv_path"]),
        digest_html_path=matrix_output_root / str(schedule_cfg["ops_digest_html_path"]),
        digest_lookback_runs=int(schedule_cfg["ops_digest_lookback_runs"]),
        monitor_summary=monitor_summary,
        shadow_summary=shadow_summary,
        rollback_summary=rollback_summary,
    )
    summary = {
        "config_path": str(config_file),
        "matrix_summary": matrix_summary,
        "monitor_summary": monitor_summary,
        "shadow_summary": shadow_summary,
        "rollback_summary": rollback_summary,
        "ops_summary": ops_summary,
    }
    summary_path = matrix_output_root / str(schedule_cfg["summary_json_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    product_dashboard_path = render_product_dashboard(
        matrix_output_root / str(schedule_cfg["product_dashboard_path"]),
        ticker_summary_csv_path=monitor_summary["ticker_summary_csv_path"],
        health_history_path=monitor_summary["health_history_path"],
        symbol_summary_path=matrix_summary["symbol_summary_path"],
        title=str(schedule_cfg["product_dashboard_title"]),
        global_report_paths={
            "Product ticker summary": monitor_summary["ticker_summary_html_path"],
            "System matrix workspace": matrix_summary.get("dashboard_path", ""),
            "Cross-symbol aggregate": matrix_summary.get("profile_aggregate_html_path", ""),
            "Cross-symbol pairwise": matrix_summary.get("profile_pairwise_html_path", ""),
            "Drift alerts": monitor_summary["drift_alerts_html_path"],
            "Ensemble ops alerts": monitor_summary.get("ensemble_alerts_html_path", ""),
            "Shadow comparison": (
                shadow_summary["comparison_summary"]["comparison_html_path"]
                if isinstance(shadow_summary, dict)
                else ""
            ),
            "Shadow gate JSON": (
                shadow_summary["comparison_summary"]["gate_json_path"]
                if isinstance(shadow_summary, dict)
                else ""
            ),
            "Shadow suggestions": (
                shadow_summary["suggestion_summary"]["suggestion_html_path"]
                if isinstance(shadow_summary, dict)
                else ""
            ),
            "Shadow suggestion JSON": (
                shadow_summary["suggestion_summary"]["suggestion_json_path"]
                if isinstance(shadow_summary, dict)
                else ""
            ),
            "Rollback state JSON": (
                rollback_summary["rollback_state_path"] if isinstance(rollback_summary, dict) else ""
            ),
            "Ops digest": ops_summary["digest_html_path"],
            "Ops journal CSV": ops_summary["journal_csv_path"],
            "Schedule summary JSON": summary_path,
        },
    )
    summary["product_dashboard_path"] = str(product_dashboard_path)
    summary["summary_json_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

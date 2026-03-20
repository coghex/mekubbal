from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .ticker_summary import _build_ticker_rankings


def render_product_dashboard(
    output_path: str | Path,
    *,
    ticker_summary_csv_path: str | Path,
    health_history_path: str | Path,
    symbol_summary_path: str | Path,
    title: str = "Dashboard",
    global_report_paths: dict[str, str | Path] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_dir = output.parent.resolve()

    ticker_summary = pd.read_csv(ticker_summary_csv_path)
    health_history = pd.read_csv(health_history_path)
    symbol_summary = pd.read_csv(symbol_summary_path)
    if ticker_summary.empty:
        raise ValueError("Ticker summary is empty.")

    def _normalize_path(path_like: str | Path) -> str | None:
        value = str(path_like).strip()
        if not value:
            return None
        if "://" in value:
            return value
        file_path = Path(value).expanduser()
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
        else:
            file_path = file_path.resolve()
        if not file_path.exists():
            return None
        try:
            return file_path.relative_to(base_dir).as_posix()
        except ValueError:
            return Path(os.path.relpath(file_path, start=base_dir)).as_posix()

    def _parse_pct_text(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip().replace("%", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    summary = ticker_summary.copy()
    summary["symbol"] = summary["symbol"].astype(str).str.upper()
    health = health_history.copy()
    health["symbol"] = health["symbol"].astype(str).str.upper()
    health["run_timestamp_utc"] = health["run_timestamp_utc"].astype(str)
    health["active_gap"] = pd.to_numeric(health.get("active_gap"), errors="coerce")
    health["selected_gap"] = pd.to_numeric(health.get("selected_gap"), errors="coerce")
    health["active_rank"] = pd.to_numeric(health.get("active_rank"), errors="coerce")
    symbol_perf = symbol_summary.copy()
    symbol_perf["symbol"] = symbol_perf["symbol"].astype(str).str.upper()
    symbol_perf["symbol_rank"] = pd.to_numeric(symbol_perf.get("symbol_rank"), errors="coerce")
    symbol_perf["avg_equity_gap"] = pd.to_numeric(symbol_perf.get("avg_equity_gap"), errors="coerce")

    ticker_payload: dict[str, Any] = {}
    nodes: list[dict[str, Any]] = []
    for _, row in summary.sort_values("symbol").iterrows():
        symbol = str(row["symbol"]).upper()
        status = str(row.get("status", "Healthy"))
        active_profile = str(row.get("active_profile", ""))
        selected_profile = str(row.get("selected_profile", active_profile))
        source = str(row.get("active_profile_source", "selection_state"))
        regime = str(row.get("ensemble_regime", "") or "")
        action = str(row.get("recommended_action", ""))
        summary_text = str(row.get("summary", ""))
        recommendation = str(row.get("recommendation", "") or "").strip()
        recommendation_subtitle = str(row.get("recommendation_subtitle", "") or "").strip()
        confidence = str(row.get("confidence", "") or "").strip()
        what_to_watch = str(row.get("what_to_watch", "") or "").strip()
        active_vs_buy = str(row.get("active_vs_buy_and_hold", "n/a"))
        active_vs_base = str(row.get("active_vs_base", "n/a"))
        active_rank = int(row.get("active_rank")) if pd.notna(row.get("active_rank")) else None
        ensemble_confidence = (
            float(row.get("ensemble_confidence"))
            if pd.notna(row.get("ensemble_confidence"))
            else None
        )
        active_vs_buy_value = _parse_pct_text(active_vs_buy)
        active_vs_base_value = _parse_pct_text(active_vs_base)

        if not recommendation:
            if status == "Healthy" and (active_vs_buy_value is not None and active_vs_buy_value > 0):
                recommendation = "Improving trend"
            elif status == "Watch":
                recommendation = "Caution"
            elif status == "Critical":
                recommendation = "Avoid for now"
            else:
                recommendation = "Mixed trend"
        if not recommendation_subtitle:
            recommendation_subtitle = (
                "positive and stable"
                if recommendation == "Bullish setup"
                else "positive, but still proving itself"
                if recommendation == "Improving trend"
                else "promise is there, but warning flags are active"
                if recommendation == "Caution"
                else "signals are mixed and not confirmed"
                if recommendation == "Mixed trend"
                else "edge is weak or deteriorating"
            )
        if not confidence:
            if ensemble_confidence is not None:
                confidence = (
                    "High"
                    if ensemble_confidence >= 0.75
                    else "Medium"
                    if ensemble_confidence >= 0.55
                    else "Low"
                )
            elif status == "Healthy":
                confidence = "Medium"
            elif status == "Watch":
                confidence = "Medium"
            else:
                confidence = "Low"
        if not what_to_watch:
            what_to_watch = action or "Watch the next daily update for clearer confirmation."

        symbol_health = health[health["symbol"] == symbol].sort_values("run_timestamp_utc")
        gap_series = [
            float(value)
            for value in symbol_health["active_gap"].tolist()
            if pd.notna(value)
        ]
        latest_gap = gap_series[-1] if gap_series else None
        prev_gap = gap_series[-2] if len(gap_series) > 1 else None
        momentum = (latest_gap - prev_gap) if latest_gap is not None and prev_gap is not None else None
        if latest_gap is None:
            outlook = "Insufficient data"
        elif latest_gap > 0 and (momentum is None or momentum >= 0):
            outlook = "Bullish continuation"
        elif latest_gap > 0:
            outlook = "Cooling upside"
        elif momentum is not None and momentum > 0:
            outlook = "Recovery watch"
        else:
            outlook = "Risk-off"

        perf_rows = symbol_perf[symbol_perf["symbol"] == symbol].sort_values(
            ["symbol_rank", "profile"]
        )
        profiles: list[dict[str, Any]] = []
        for _, perf in perf_rows.iterrows():
            profile_name = str(perf.get("profile", ""))
            profiles.append(
                {
                    "profile": profile_name,
                    "rank": int(perf["symbol_rank"]) if pd.notna(perf["symbol_rank"]) else None,
                    "gap_pct": (
                        float(perf["avg_equity_gap"]) * 100.0
                        if pd.notna(perf["avg_equity_gap"])
                        else None
                    ),
                    "visual_report": _normalize_path(perf.get("visual_report_path")),
                    "pairwise_report": _normalize_path(perf.get("symbol_pairwise_html_path")),
                }
            )

        ticker_payload[symbol] = {
            "symbol": symbol,
            "status": status,
            "recommendation": recommendation,
            "recommendation_subtitle": recommendation_subtitle,
            "confidence": confidence,
            "selected_profile": selected_profile,
            "active_profile": active_profile,
            "active_profile_source": source,
            "regime": regime or None,
            "ensemble_confidence": ensemble_confidence,
            "active_rank": active_rank,
            "active_vs_buy_pct_text": active_vs_buy,
            "active_vs_buy_pct_value": active_vs_buy_value,
            "active_vs_base_pct_text": active_vs_base,
            "active_vs_base_pct_value": active_vs_base_value,
            "action": action,
            "summary": summary_text,
            "what_to_watch": what_to_watch,
            "outlook": outlook,
            "momentum": momentum,
            "history": [
                {
                    "run_timestamp_utc": str(item["run_timestamp_utc"]),
                    "active_gap": (
                        float(item["active_gap"]) if pd.notna(item["active_gap"]) else None
                    ),
                    "selected_gap": (
                        float(item["selected_gap"]) if pd.notna(item["selected_gap"]) else None
                    ),
                    "active_rank": (
                        float(item["active_rank"]) if pd.notna(item["active_rank"]) else None
                    ),
                }
                for _, item in symbol_health.iterrows()
            ],
            "profiles": profiles,
        }

        nodes.append(
            {
                "symbol": symbol,
                "status": status,
                "active_vs_buy_pct": _parse_pct_text(active_vs_buy),
                "confidence": ensemble_confidence,
            }
        )

    ranking_payload = _build_ticker_rankings(ticker_payload)

    dense_links: list[dict[str, str]] = []
    report_label_to_raw: dict[str, str | Path] = {}
    if global_report_paths:
        for label, raw in global_report_paths.items():
            label_text = str(label).strip()
            report_label_to_raw[label_text.lower()] = raw
            normalized = _normalize_path(raw)
            if normalized is None:
                continue
            dense_links.append({"label": label_text, "url": normalized})
    dense_links.sort(key=lambda item: item["label"])

    shadow_gate_payload: dict[str, Any] = {
        "enabled": False,
        "overall_gate_passed": None,
        "window_runs": None,
        "min_match_ratio": None,
        "failing_symbols": [],
        "symbols": [],
        "gate_json_url": _normalize_path(report_label_to_raw.get("shadow gate json", "")),
        "comparison_url": _normalize_path(report_label_to_raw.get("shadow comparison", "")),
    }
    shadow_suggestion_payload: dict[str, Any] = {
        "enabled": False,
        "accepted": False,
        "recommended_window_runs": None,
        "recommended_min_match_ratio": None,
        "reasons": [],
        "suggestion_json_url": _normalize_path(report_label_to_raw.get("shadow suggestion json", "")),
        "suggestion_html_url": _normalize_path(report_label_to_raw.get("shadow suggestions", "")),
        "suggestion_history_url": _normalize_path(report_label_to_raw.get("shadow suggestion history csv", "")),
        "state_json_url": _normalize_path(report_label_to_raw.get("shadow suggestion state json", "")),
        "state_active_window_runs": None,
        "state_active_min_match_ratio": None,
        "state_updated_at_utc": None,
        "recommendation_metrics": {},
    }
    shadow_gate_raw = report_label_to_raw.get("shadow gate json")
    if shadow_gate_raw is not None:
        shadow_gate_value = str(shadow_gate_raw).strip()
        if shadow_gate_value and "://" not in shadow_gate_value:
            shadow_gate_path = Path(shadow_gate_value).expanduser()
            if not shadow_gate_path.is_absolute():
                shadow_gate_path = (Path.cwd() / shadow_gate_path).resolve()
            else:
                shadow_gate_path = shadow_gate_path.resolve()
            if shadow_gate_path.exists():
                try:
                    loaded_shadow_gate = json.loads(shadow_gate_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    loaded_shadow_gate = None
                if isinstance(loaded_shadow_gate, dict):
                    symbol_rows: list[dict[str, Any]] = []
                    for item in loaded_shadow_gate.get("symbols", []):
                        if not isinstance(item, dict):
                            continue
                        symbol_rows.append(
                            {
                                "symbol": str(item.get("symbol", "")).upper(),
                                "gate_passed": bool(item.get("gate_passed", False)),
                                "runs_in_window": int(item.get("runs_in_window", 0)),
                                "window_runs_required": int(item.get("window_runs_required", 0)),
                                "match_ratio": (
                                    float(item.get("match_ratio"))
                                    if item.get("match_ratio") is not None
                                    else None
                                ),
                                "min_match_ratio": (
                                    float(item.get("min_match_ratio"))
                                    if item.get("min_match_ratio") is not None
                                    else None
                                ),
                            }
                        )
                    shadow_gate_payload = {
                        "enabled": True,
                        "overall_gate_passed": bool(loaded_shadow_gate.get("overall_gate_passed", False)),
                        "window_runs": (
                            int(loaded_shadow_gate["window_runs"])
                            if loaded_shadow_gate.get("window_runs") is not None
                            else None
                        ),
                        "min_match_ratio": (
                            float(loaded_shadow_gate["min_match_ratio"])
                            if loaded_shadow_gate.get("min_match_ratio") is not None
                            else None
                        ),
                        "failing_symbols": [
                            str(value) for value in loaded_shadow_gate.get("failing_symbols", [])
                        ],
                        "symbols": symbol_rows,
                        "gate_json_url": _normalize_path(shadow_gate_value),
                        "comparison_url": _normalize_path(report_label_to_raw.get("shadow comparison", "")),
                    }
    shadow_suggestion_raw = report_label_to_raw.get("shadow suggestion json")
    if shadow_suggestion_raw is not None:
        shadow_suggestion_value = str(shadow_suggestion_raw).strip()
        if shadow_suggestion_value and "://" not in shadow_suggestion_value:
            shadow_suggestion_path = Path(shadow_suggestion_value).expanduser()
            if not shadow_suggestion_path.is_absolute():
                shadow_suggestion_path = (Path.cwd() / shadow_suggestion_path).resolve()
            else:
                shadow_suggestion_path = shadow_suggestion_path.resolve()
            if shadow_suggestion_path.exists():
                try:
                    loaded_shadow_suggestion = json.loads(
                        shadow_suggestion_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    loaded_shadow_suggestion = None
                if isinstance(loaded_shadow_suggestion, dict):
                    shadow_suggestion_payload = {
                        "enabled": True,
                        "accepted": bool(loaded_shadow_suggestion.get("accepted", False)),
                        "recommended_window_runs": loaded_shadow_suggestion.get("recommended_window_runs"),
                        "recommended_min_match_ratio": loaded_shadow_suggestion.get(
                            "recommended_min_match_ratio"
                        ),
                        "reasons": [
                            str(value)
                            for value in loaded_shadow_suggestion.get("reasons", [])
                        ],
                        "suggestion_json_url": _normalize_path(shadow_suggestion_value),
                        "suggestion_html_url": _normalize_path(
                            report_label_to_raw.get("shadow suggestions", "")
                        ),
                        "suggestion_history_url": _normalize_path(
                            report_label_to_raw.get("shadow suggestion history csv", "")
                        ),
                        "state_json_url": _normalize_path(
                            report_label_to_raw.get("shadow suggestion state json", "")
                        ),
                        "state_active_window_runs": None,
                        "state_active_min_match_ratio": None,
                        "state_updated_at_utc": None,
                        "recommendation_metrics": (
                            loaded_shadow_suggestion.get("recommendation_metrics")
                            if isinstance(
                                loaded_shadow_suggestion.get("recommendation_metrics"), dict
                            )
                            else {}
                        ),
                    }
    shadow_suggestion_state_raw = report_label_to_raw.get("shadow suggestion state json")
    if shadow_suggestion_state_raw is not None:
        shadow_suggestion_state_value = str(shadow_suggestion_state_raw).strip()
        if shadow_suggestion_state_value and "://" not in shadow_suggestion_state_value:
            shadow_suggestion_state_path = Path(shadow_suggestion_state_value).expanduser()
            if not shadow_suggestion_state_path.is_absolute():
                shadow_suggestion_state_path = (Path.cwd() / shadow_suggestion_state_path).resolve()
            else:
                shadow_suggestion_state_path = shadow_suggestion_state_path.resolve()
            if shadow_suggestion_state_path.exists():
                try:
                    loaded_shadow_state = json.loads(
                        shadow_suggestion_state_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    loaded_shadow_state = None
                if isinstance(loaded_shadow_state, dict):
                    shadow_suggestion_payload["state_json_url"] = _normalize_path(
                        shadow_suggestion_state_value
                    )
                    shadow_suggestion_payload["state_active_window_runs"] = loaded_shadow_state.get(
                        "active_window_runs"
                    )
                    shadow_suggestion_payload["state_active_min_match_ratio"] = loaded_shadow_state.get(
                        "active_min_match_ratio"
                    )
                    shadow_suggestion_payload["state_updated_at_utc"] = loaded_shadow_state.get(
                        "updated_at_utc"
                    )

    rollback_payload: dict[str, Any] = {
        "available": False,
        "state_json_url": _normalize_path(report_label_to_raw.get("rollback state json", "")),
        "rollback_recommended_count": 0,
        "rollback_applied_count": 0,
        "symbols_evaluated": 0,
        "symbols": [],
    }
    rollback_state_raw = report_label_to_raw.get("rollback state json")
    if rollback_state_raw is not None:
        rollback_state_value = str(rollback_state_raw).strip()
        if rollback_state_value and "://" not in rollback_state_value:
            rollback_state_path = Path(rollback_state_value).expanduser()
            if not rollback_state_path.is_absolute():
                rollback_state_path = (Path.cwd() / rollback_state_path).resolve()
            else:
                rollback_state_path = rollback_state_path.resolve()
            if rollback_state_path.exists():
                try:
                    loaded_rollback_state = json.loads(rollback_state_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    loaded_rollback_state = None
                if isinstance(loaded_rollback_state, dict):
                    rollback_summary = (
                        loaded_rollback_state.get("summary")
                        if isinstance(loaded_rollback_state.get("summary"), dict)
                        else {}
                    )
                    rollback_symbols: list[dict[str, Any]] = []
                    for item in loaded_rollback_state.get("symbols", []):
                        if not isinstance(item, dict):
                            continue
                        rollback_symbols.append(
                            {
                                "symbol": str(item.get("symbol", "")).upper(),
                                "should_rollback": bool(item.get("should_rollback", False)),
                                "action": str(item.get("action", "") or ""),
                                "rollback_profile": str(item.get("rollback_profile", "") or ""),
                                "current_active_profile": str(item.get("current_active_profile", "") or ""),
                            }
                        )
                    rollback_payload = {
                        "available": True,
                        "state_json_url": _normalize_path(rollback_state_value),
                        "rollback_recommended_count": int(
                            rollback_summary.get("rollback_recommended_count", 0) or 0
                        ),
                        "rollback_applied_count": int(
                            rollback_summary.get("rollback_applied_count", 0) or 0
                        ),
                        "symbols_evaluated": int(rollback_summary.get("symbols_evaluated", 0) or 0),
                        "symbols": rollback_symbols,
                    }

    run_delta_payload: dict[str, Any] = {
        "has_previous": False,
        "latest_run_timestamp": None,
        "previous_run_timestamp": None,
        "symbols_compared": 0,
        "profile_change_count": 0,
        "source_change_count": 0,
        "gap_up_count": 0,
        "gap_down_count": 0,
        "rank_improved_count": 0,
        "rank_worsened_count": 0,
        "largest_gap_up_symbol": None,
        "largest_gap_up_delta": None,
        "largest_gap_down_symbol": None,
        "largest_gap_down_delta": None,
        "symbol_changes": [],
        "shadow_match_ratio_latest": None,
        "shadow_match_ratio_previous": None,
        "shadow_match_ratio_delta": None,
        "shadow_recovered_matches": 0,
        "shadow_new_mismatches": 0,
    }
    health_for_delta = health.copy()
    if "active_profile" in health_for_delta.columns:
        health_for_delta["active_profile"] = health_for_delta["active_profile"].astype(str)
    if "active_profile_source" in health_for_delta.columns:
        health_for_delta["active_profile_source"] = health_for_delta["active_profile_source"].astype(str)
    run_values = sorted(
        {
            str(value)
            for value in health_for_delta.get("run_timestamp_utc", []).tolist()
            if str(value).strip()
        }
    )
    if len(run_values) >= 2:
        latest_run = run_values[-1]
        previous_run = run_values[-2]
        latest_rows = health_for_delta[health_for_delta["run_timestamp_utc"].astype(str) == latest_run].copy()
        previous_rows = health_for_delta[
            health_for_delta["run_timestamp_utc"].astype(str) == previous_run
        ].copy()
        keep_cols = [
            "symbol",
            "active_profile",
            "active_profile_source",
            "active_rank",
            "active_gap",
        ]
        latest_rows = latest_rows[[column for column in keep_cols if column in latest_rows.columns]].copy()
        previous_rows = previous_rows[
            [column for column in keep_cols if column in previous_rows.columns]
        ].copy()
        latest_rows = latest_rows.rename(
            columns={column: f"{column}_latest" for column in latest_rows.columns if column != "symbol"}
        )
        previous_rows = previous_rows.rename(
            columns={column: f"{column}_previous" for column in previous_rows.columns if column != "symbol"}
        )
        merged_delta = latest_rows.merge(previous_rows, on="symbol", how="inner")
        if not merged_delta.empty:
            merged_delta["active_rank_latest"] = pd.to_numeric(
                merged_delta.get("active_rank_latest"), errors="coerce"
            )
            merged_delta["active_rank_previous"] = pd.to_numeric(
                merged_delta.get("active_rank_previous"), errors="coerce"
            )
            merged_delta["active_gap_latest"] = pd.to_numeric(
                merged_delta.get("active_gap_latest"), errors="coerce"
            )
            merged_delta["active_gap_previous"] = pd.to_numeric(
                merged_delta.get("active_gap_previous"), errors="coerce"
            )
            merged_delta["rank_delta"] = (
                merged_delta["active_rank_latest"] - merged_delta["active_rank_previous"]
            )
            merged_delta["gap_delta"] = (
                merged_delta["active_gap_latest"] - merged_delta["active_gap_previous"]
            )
            latest_profile_series = (
                merged_delta["active_profile_latest"].astype(str)
                if "active_profile_latest" in merged_delta.columns
                else pd.Series([""] * len(merged_delta))
            )
            previous_profile_series = (
                merged_delta["active_profile_previous"].astype(str)
                if "active_profile_previous" in merged_delta.columns
                else pd.Series([""] * len(merged_delta))
            )
            latest_source_series = (
                merged_delta["active_profile_source_latest"].astype(str)
                if "active_profile_source_latest" in merged_delta.columns
                else pd.Series([""] * len(merged_delta))
            )
            previous_source_series = (
                merged_delta["active_profile_source_previous"].astype(str)
                if "active_profile_source_previous" in merged_delta.columns
                else pd.Series([""] * len(merged_delta))
            )
            merged_delta["profile_changed"] = latest_profile_series != previous_profile_series
            merged_delta["source_changed"] = latest_source_series != previous_source_series
            run_delta_payload.update(
                {
                    "has_previous": True,
                    "latest_run_timestamp": latest_run,
                    "previous_run_timestamp": previous_run,
                    "symbols_compared": int(len(merged_delta)),
                    "profile_change_count": int(merged_delta["profile_changed"].sum()),
                    "source_change_count": int(merged_delta["source_changed"].sum()),
                    "gap_up_count": int((merged_delta["gap_delta"] > 0).sum()),
                    "gap_down_count": int((merged_delta["gap_delta"] < 0).sum()),
                    "rank_improved_count": int((merged_delta["rank_delta"] < 0).sum()),
                    "rank_worsened_count": int((merged_delta["rank_delta"] > 0).sum()),
                }
            )
            if (merged_delta["gap_delta"] > 0).any():
                best = merged_delta.loc[merged_delta["gap_delta"].idxmax()]
                run_delta_payload["largest_gap_up_symbol"] = str(best["symbol"])
                run_delta_payload["largest_gap_up_delta"] = float(best["gap_delta"])
            if (merged_delta["gap_delta"] < 0).any():
                worst = merged_delta.loc[merged_delta["gap_delta"].idxmin()]
                run_delta_payload["largest_gap_down_symbol"] = str(worst["symbol"])
                run_delta_payload["largest_gap_down_delta"] = float(worst["gap_delta"])
            changes = merged_delta[
                merged_delta["profile_changed"]
                | merged_delta["source_changed"]
                | (merged_delta["gap_delta"].abs() > 1e-12)
                | (merged_delta["rank_delta"].abs() > 1e-12)
            ].copy()
            if not changes.empty:
                changes = changes.sort_values(
                    ["profile_changed", "gap_delta"],
                    ascending=[False, False],
                    key=lambda column: column
                    if column.name != "gap_delta"
                    else column.abs(),
                )
            run_delta_payload["symbol_changes"] = [
                {
                    "symbol": str(item.get("symbol", "")),
                    "profile_changed": bool(item.get("profile_changed", False)),
                    "profile_latest": str(item.get("active_profile_latest", "")),
                    "profile_previous": str(item.get("active_profile_previous", "")),
                    "source_changed": bool(item.get("source_changed", False)),
                    "source_latest": str(item.get("active_profile_source_latest", "")),
                    "source_previous": str(item.get("active_profile_source_previous", "")),
                    "rank_delta": (
                        float(item.get("rank_delta")) if pd.notna(item.get("rank_delta")) else None
                    ),
                    "gap_delta": float(item.get("gap_delta")) if pd.notna(item.get("gap_delta")) else None,
                }
                for _, item in changes.head(12).iterrows()
            ]

    shadow_history_raw = report_label_to_raw.get("shadow comparison history csv")
    if shadow_history_raw is not None:
        shadow_history_value = str(shadow_history_raw).strip()
        if shadow_history_value and "://" not in shadow_history_value:
            shadow_history_path = Path(shadow_history_value).expanduser()
            if not shadow_history_path.is_absolute():
                shadow_history_path = (Path.cwd() / shadow_history_path).resolve()
            else:
                shadow_history_path = shadow_history_path.resolve()
            if shadow_history_path.exists():
                try:
                    shadow_history = pd.read_csv(shadow_history_path)
                except (OSError, pd.errors.ParserError):
                    shadow_history = pd.DataFrame()
                required_shadow_history = {"run_timestamp_utc", "symbol", "active_profile_match"}
                if not shadow_history.empty and required_shadow_history.issubset(shadow_history.columns):
                    shadow_history = shadow_history.copy()
                    shadow_history["run_timestamp_utc"] = shadow_history["run_timestamp_utc"].astype(str)
                    shadow_history["symbol"] = shadow_history["symbol"].astype(str).str.upper()
                    shadow_history["active_profile_match"] = (
                        shadow_history["active_profile_match"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map({"true": 1, "false": 0, "1": 1, "0": 0})
                    )
                    shadow_history = shadow_history.dropna(subset=["active_profile_match"])
                    if not shadow_history.empty:
                        shadow_runs = sorted(
                            {
                                str(value)
                                for value in shadow_history["run_timestamp_utc"].tolist()
                                if str(value).strip()
                            }
                        )
                        if len(shadow_runs) >= 2:
                            shadow_latest = shadow_runs[-1]
                            shadow_previous = shadow_runs[-2]
                            latest_match = shadow_history[
                                shadow_history["run_timestamp_utc"] == shadow_latest
                            ][["symbol", "active_profile_match"]].rename(
                                columns={"active_profile_match": "match_latest"}
                            )
                            previous_match = shadow_history[
                                shadow_history["run_timestamp_utc"] == shadow_previous
                            ][["symbol", "active_profile_match"]].rename(
                                columns={"active_profile_match": "match_previous"}
                            )
                            merged_match = latest_match.merge(previous_match, on="symbol", how="inner")
                            if not merged_match.empty:
                                merged_match["match_latest"] = pd.to_numeric(
                                    merged_match["match_latest"], errors="coerce"
                                )
                                merged_match["match_previous"] = pd.to_numeric(
                                    merged_match["match_previous"], errors="coerce"
                                )
                                merged_match = merged_match.dropna(
                                    subset=["match_latest", "match_previous"]
                                )
                                if not merged_match.empty:
                                    run_delta_payload["shadow_match_ratio_latest"] = float(
                                        merged_match["match_latest"].mean()
                                    )
                                    run_delta_payload["shadow_match_ratio_previous"] = float(
                                        merged_match["match_previous"].mean()
                                    )
                                    run_delta_payload["shadow_match_ratio_delta"] = float(
                                        run_delta_payload["shadow_match_ratio_latest"]
                                        - run_delta_payload["shadow_match_ratio_previous"]
                                    )
                                    run_delta_payload["shadow_recovered_matches"] = int(
                                        (
                                            (merged_match["match_previous"] < 0.5)
                                            & (merged_match["match_latest"] >= 0.5)
                                        ).sum()
                                    )
                                    run_delta_payload["shadow_new_mismatches"] = int(
                                        (
                                            (merged_match["match_previous"] >= 0.5)
                                            & (merged_match["match_latest"] < 0.5)
                                        ).sum()
                                    )

    tickers_sorted = [
        str(item["symbol"]).upper()
        for item in ranking_payload
        if str(item.get("symbol", "")).strip()
    ]
    if not tickers_sorted:
        tickers_sorted = sorted(ticker_payload)
    if not tickers_sorted:
        raise ValueError("No ticker rows found for product dashboard.")
    nav_buttons = "".join(
        (
            f"<button id='nav-{ticker}' class='nav-button nav-ticker-button' "
            f'title="{ticker}" aria-label="{ticker}" onclick="showTicker(\'{ticker}\')">{ticker}</button>'
        )
        for ticker in tickers_sorted
    )
    latest_health_run = None
    if "run_timestamp_utc" in health.columns:
        health_runs = sorted(
            {
                str(value)
                for value in health["run_timestamp_utc"].tolist()
                if str(value).strip()
            }
        )
        latest_health_run = health_runs[-1] if health_runs else None

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f4f6;
      --panel: #ffffff;
      --panel-soft: #f5f5f5;
      --border: #e4e4e7;
      --text: #0f172a;
      --muted: #475569;
      --nav: #09090b;
      --nav-soft: #18181b;
      --blue: #111111;
      --blue-soft: #f3f4f6;
      --green: #15803d;
      --green-soft: #dcfce7;
      --amber: #b45309;
      --amber-soft: #fef3c7;
      --red: #b91c1c;
      --red-soft: #fee2e2;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: var(--text); background: var(--bg); }}
    a {{ color: var(--blue); }}
    .layout {{ display: grid; grid-template-columns: 96px minmax(0, 1fr); min-height: 100vh; }}
    .side {{
      background: #09090b;
      color: #f4f4f5;
      padding: 10px 6px;
      border-right: 1px solid rgba(148, 163, 184, 0.12);
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-height: 100vh;
    }}
    .ticker-rail {{ display: flex; flex-direction: column; align-items: stretch; gap: 6px; overflow: auto; }}
    .rail-bottom {{ margin-top: auto; padding-top: 8px; display: flex; justify-content: stretch; }}
    .brand {{ font-size: 18px; font-weight: 700; letter-spacing: -0.02em; }}
    .brand-copy {{ margin-top: 8px; font-size: 13px; line-height: 1.5; color: #cbd5e1; }}
    .nav-section {{ margin-top: 18px; }}
    .nav-label {{
      margin: 0 0 8px 0;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #94a3b8;
    }}
    .nav-button {{
      width: 100%;
      min-width: 0;
      text-align: center;
      padding: 11px 8px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      margin: 0;
      cursor: pointer;
      color: #e4e4e7;
      background: #111111;
      font-weight: 700;
      letter-spacing: 0.01em;
      font-size: 11px;
      line-height: 1.2;
      transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
    }}
    .nav-ticker-button {{ min-height: 42px; }}
    .nav-overview-button {{
      background: #111111;
      border-color: #27272a;
      color: #e4e4e7;
      padding: 11px 8px;
      margin-bottom: 4px;
    }}
    .nav-overview-button.active {{
      background: #fafafa;
      color: #111111;
      border-color: #d4d4d8;
    }}
    .nav-system-button {{
      background: #111111;
      color: #e4e4e7;
      border-color: #27272a;
    }}
    .nav-button:hover {{ background: #1f1f23; border-color: #3f3f46; }}
    .nav-button.active {{
      background: #fafafa;
      color: #111111;
      border-color: #d4d4d8;
      box-shadow: none;
    }}
    .nav-button.active:hover {{ background: #fafafa; border-color: #d4d4d8; }}
    .nav-button:focus-visible,
    .row-item-button:focus-visible,
    .advanced-panel summary:focus-visible,
    .system-links a:focus-visible,
    .report-list a:focus-visible {{
      outline: 2px solid #2563eb;
      outline-offset: 2px;
    }}
    .main {{
      padding: 28px;
      overflow: auto;
      min-width: 0;
      background: linear-gradient(180deg, #f8fafc 0%, #f3f4f6 180px);
    }}
    .panel {{ display: none; }}
    .panel.active {{
      display: grid;
      gap: 18px;
      align-content: start;
      max-width: 1440px;
      margin: 0 auto;
    }}
    .workspace-head {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 0;
    }}
    .stat-block {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px 18px;
      box-shadow: var(--shadow);
      min-height: 84px;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
    }}
    .stat-block.wide {{ grid-column: span 2; }}
    .stat-label {{
      display: block;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
    }}
    .stat-value {{
      margin-top: 8px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .stat-copy {{ margin-top: 6px; font-size: 12px; line-height: 1.5; color: var(--muted); }}
    .overview-strip {{
      display: grid;
      grid-template-columns: minmax(200px, 0.9fr) minmax(160px, 0.6fr) minmax(260px, 1fr);
      gap: 0;
      align-items: stretch;
      margin-bottom: 0;
      overflow: hidden;
      padding: 0;
    }}
    .surface.overview-strip {{
      padding: 0;
      border-radius: 20px;
    }}
    .overview-strip-item {{
      padding: 20px 22px;
      border-right: 1px solid #e5e7eb;
      min-width: 0;
    }}
    .overview-strip-item:last-child {{ border-right: none; }}
    .overview-strip-value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .overview-strip-value.timestamp {{
      font-size: 18px;
      line-height: 1.3;
      word-break: break-word;
    }}
    .mini-stat-row {{
      display: flex;
      gap: 18px;
      align-items: flex-end;
      margin-top: 8px;
      flex-wrap: wrap;
    }}
    .mini-stat {{
      min-width: 56px;
    }}
    .mini-stat-label {{
      display: block;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #71717a;
    }}
    .mini-stat-value {{
      display: block;
      margin-top: 4px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.03em;
      color: #111111;
    }}
    .workspace-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 0;
    }}
    .workspace-grid-compact {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 18px;
      margin-top: 0;
    }}
    .workspace-grid-triple {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) minmax(280px, 0.85fr);
      gap: 18px;
      margin-top: 0;
    }}
    .section-title {{
      margin: 0 0 14px 0;
      font-size: 19px;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: #0f172a;
    }}
    .canvas-block {{
      width: 100%;
      height: 280px;
      display: block;
      background: #fff;
    }}
    .canvas-block.compact {{ height: 230px; }}
    .row-list {{
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .row-item {{
      display: grid;
      gap: 6px;
      padding: 14px 16px;
      border: 1px solid #e5edf7;
      border-radius: 14px;
      background: #fff;
    }}
    .row-list > * {{ min-width: 0; }}
    .row-item-main {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      font-size: 14px;
    }}
    .row-item-main strong {{ font-size: 15px; letter-spacing: 0.01em; }}
    .row-item-meta {{ color: #0f172a; font-weight: 700; white-space: nowrap; }}
    .row-item-note {{ color: var(--muted); font-size: 13px; line-height: 1.6; }}
    .summary-copy {{ margin: 0; color: var(--muted); line-height: 1.7; font-size: 14px; }}
    .row-item-button {{
      width: 100%;
      border: 0;
      background: transparent;
      padding: 0;
      text-align: left;
      cursor: pointer;
      color: inherit;
      font: inherit;
      border-radius: 12px;
    }}
    .row-item-button:hover .row-item {{
      background: #f8fafc;
      border-color: #d8e2f0;
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }}
    .signal-list {{
      display: grid;
      gap: 12px;
    }}
    .signal-line {{
      border-top: 1px solid #e5edf7;
      padding-top: 10px;
    }}
    .signal-list > :first-child {{ border-top: none; padding-top: 0; }}
    .signal-label {{
      display: block;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      margin-bottom: 4px;
    }}
    .signal-value {{ margin: 0; line-height: 1.6; color: var(--text); }}
    .rail-note {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      margin: 0 0 6px 0;
      text-align: center;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-start;
      background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
      color: #eff6ff;
      border-radius: 22px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      opacity: 0.85;
    }}
    .hero h1, .hero h2 {{ margin: 8px 0 10px 0; font-size: 30px; line-height: 1.1; }}
    .hero-copy {{ margin: 0; max-width: 760px; font-size: 15px; line-height: 1.6; color: rgba(239, 246, 255, 0.92); }}
    .hero-meta {{
      min-width: 220px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 18px;
      padding: 16px;
      font-size: 13px;
      line-height: 1.6;
    }}
    .hero-meta strong {{ display: block; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.8; }}
    .summary-grid, .detail-metrics, .system-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 14px;
      margin-top: 0;
    }}
    .surface {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .surface-soft {{ background: var(--panel-soft); }}
    .kicker {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }}
    .metric-value {{ margin-top: 8px; font-size: 28px; font-weight: 700; letter-spacing: -0.03em; }}
    .metric-sub {{ margin-top: 6px; font-size: 13px; line-height: 1.5; color: var(--muted); }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: end;
      margin-top: 0;
      margin-bottom: 14px;
    }}
    .section-head h3 {{ margin: 0; font-size: 22px; }}
    .section-head p {{ margin: 4px 0 0 0; color: var(--muted); }}
    .overview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 14px;
    }}
    .overview-toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      align-items: end;
      margin-bottom: 12px;
    }}
    .overview-filter-bar {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .filter-chip {{
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .filter-chip.active {{
      border-color: #93c5fd;
      background: #eff6ff;
      color: #1d4ed8;
    }}
    .search-group {{ min-width: min(280px, 100%); }}
    .search-input {{
      width: min(320px, 100%);
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
      color: var(--text);
    }}
    .overview-results-copy {{ margin: 0 0 14px 0; color: var(--muted); line-height: 1.6; }}
    .ranking-table-wrap {{ margin-top: 12px; }}
    .ranking-note {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .comparison-copy {{ margin: 0; color: #334155; line-height: 1.6; }}
    .ticker-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 20px;
      box-shadow: var(--shadow);
      display: grid;
      gap: 16px;
    }}
    .ticker-card-top {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
    }}
    .ticker-card-symbol {{ font-size: 24px; font-weight: 700; letter-spacing: -0.03em; }}
    .ticker-card-rank {{ margin-top: 4px; font-size: 13px; color: var(--muted); }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .badge-shortlist, .badge-healthy, .shadow-status.pass {{ background: var(--green-soft); color: var(--green); }}
    .badge-constructive, .badge-medium, .badge-overview {{ background: var(--blue-soft); color: var(--blue); }}
    .badge-watch, .badge-watch-closely, .shadow-status.fail {{ background: var(--amber-soft); color: var(--amber); }}
    .badge-hold-off, .badge-critical {{ background: var(--red-soft); color: var(--red); }}
    .badge-low, .shadow-status.inactive {{ background: #e2e8f0; color: #334155; }}
    .metric-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      background: #f8fafc;
      border: 1px solid #e5edf7;
      border-radius: 14px;
      padding: 14px;
    }}
    .metric-strip .item-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; }}
    .metric-strip .item-value {{ margin-top: 5px; font-size: 18px; font-weight: 700; }}
    .ticker-card-copy, .body-copy {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .ticker-card-signal {{ margin: 0; color: #111111; font-weight: 700; line-height: 1.5; }}
    .watch-box {{
      background: #f5f5f5;
      border: 1px solid #e4e4e7;
      color: #111111;
      border-radius: 8px;
      padding: 12px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .card-actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .primary-button, .ghost-button, .link-button {{
      border-radius: 12px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }}
    .primary-button {{ border: 1px solid var(--blue); background: var(--blue); color: #fff; }}
    .ghost-button {{ border: 1px solid var(--border); background: #fff; color: var(--text); }}
    .link-button {{ border: 1px solid #d4d4d8; background: #f5f5f5; color: #111111; }}
    .ticker-toolbar {{ display: block; margin-bottom: 0; }}
    .ticker-title {{ font-size: 32px; font-weight: 700; letter-spacing: -0.03em; margin: 8px 0 0 0; }}
    .ticker-subtitle {{ margin: 8px 0 0 0; max-width: 760px; line-height: 1.6; color: var(--muted); }}
    .signal-subtitle {{ margin: 10px 0 0 0; color: #111111; font-weight: 700; line-height: 1.5; }}
    .detail-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
      gap: 14px;
      margin-top: 14px;
    }}
    .chart-wrap {{ background: #fff; border: 1px solid var(--border); border-radius: 18px; padding: 16px; box-shadow: var(--shadow); }}
    #ticker-chart {{ width: 100%; height: 260px; }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid #e5edf7;
      padding: 11px 12px;
      text-align: left;
      font-size: 13px;
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover td {{ background: #fafcff; }}
    th {{ background: #f8fafc; }}
    .report-list {{ display: grid; gap: 10px; }}
    .report-list a {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
      color: #111111;
      text-decoration: none;
      transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }}
    .report-list a:hover,
    .system-links a:hover {{
      background: #f8fafc;
      border-color: #d8e2f0;
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }}
    .preview {{ margin-top: 12px; }}
    .preview iframe {{ width: 100%; height: 420px; border: 1px solid var(--border); border-radius: 14px; }}
    .select-input {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
    }}
    .details-note {{
      margin-top: 0;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 0;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .details-note > div {{
      padding: 16px 18px;
      background: #fafafa;
    }}
    details {{ background: #fff; border: 1px solid var(--border); border-radius: 14px; padding: 12px; }}
    details summary {{ cursor: pointer; font-weight: 700; }}
    .details-note summary {{
      list-style: none;
      padding: 16px 18px;
      font-size: 15px;
    }}
    .details-note summary::-webkit-details-marker {{ display: none; }}
    .details-note[open] summary {{ border-bottom: 1px solid #e5e7eb; }}
    #system-panel .hero {{ margin-bottom: 4px; }}
    .system-summary-grid {{ margin-top: 14px; }}
    .system-primary-card {{ margin-top: 14px; }}
    .advanced-stack {{ display: grid; gap: 12px; margin-top: 16px; }}
    .advanced-panel {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
      padding: 0;
    }}
    .advanced-panel summary {{
      list-style: none;
      padding: 16px 18px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }}
    .advanced-panel summary::-webkit-details-marker {{ display: none; }}
    .advanced-panel[open] summary {{ border-bottom: 1px solid #e5e7eb; }}
    .advanced-panel-body {{ padding: 16px 18px; background: #fafafa; }}
    .shadow-panel, .delta-panel {{ background: #fff; border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: var(--shadow); }}
    .shadow-head, .delta-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
    .shadow-status {{ font-size: 13px; font-weight: 700; border-radius: 999px; padding: 6px 12px; }}
    .shadow-meta, .shadow-failing, .shadow-suggestion, .delta-sub {{ margin-top: 8px; font-size: 13px; line-height: 1.6; color: var(--muted); }}
    .shadow-table-wrap, .delta-table-wrap {{ margin-top: 10px; }}
    .delta-grid {{ margin-top: 12px; display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 10px; }}
    .delta-item {{ background: #f8fafc; border: 1px solid #e5edf7; border-radius: 14px; padding: 12px; }}
    .delta-item .k {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }}
    .delta-item .v {{ margin-top: 6px; font-size: 15px; font-weight: 700; line-height: 1.4; }}
    #system-svg {{ width: 100%; height: 420px; background: #0b1220; border-radius: 18px; box-shadow: var(--shadow); }}
    .system-links {{ display: grid; gap: 10px; }}
    .system-links a {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
      color: #111111;
      text-decoration: none;
      transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }}
    .footnote {{ margin-top: 20px; font-size: 12px; color: #64748b; line-height: 1.6; }}
    @media (max-width: 700px) {{
      .layout {{ grid-template-columns: 88px minmax(0, 1fr); }}
      .hero, .ticker-toolbar, .section-head {{ flex-direction: column; }}
      .summary-grid, .detail-metrics, .system-grid, .detail-grid, .delta-grid, .workspace-head, .workspace-grid, .workspace-grid-compact, .workspace-grid-triple, .overview-strip {{ grid-template-columns: 1fr; }}
      .overview-strip-item {{ border-right: none; border-bottom: 1px solid #e5e7eb; }}
      .overview-strip-item:last-child {{ border-bottom: none; }}
    }}
  </style>
</head>
  <body>
  <div class="layout">
    <aside class="side">
      <button id="nav-overview" class="nav-button nav-overview-button active" title="Overview" aria-label="Overview" onclick="showOverview()">Overview</button>
      <div class="ticker-rail">
        {nav_buttons}
      </div>
      <div class="rail-bottom">
        <button id="nav-system" class="nav-button nav-system-button" title="System" aria-label="System" onclick="showSystem()">System</button>
      </div>
    </aside>
    <main class="main">
      <section id="overview-panel" class="panel active">
        <div class="surface overview-strip">
          <div class="overview-strip-item">
            <span class="stat-label">Last updated</span>
            <div id="overview-run-label" class="overview-strip-value timestamp">{html.escape(latest_health_run or "n/a")}</div>
          </div>
          <div class="overview-strip-item">
            <span class="stat-label">Top performer</span>
            <div id="overview-leader" class="overview-strip-value">n/a</div>
          </div>
          <div class="overview-strip-item">
            <span class="stat-label">Current mix</span>
            <div class="mini-stat-row">
              <div class="mini-stat">
                <span class="mini-stat-label">Strong</span>
                <span id="overview-shortlist-count" class="mini-stat-value">0</span>
              </div>
              <div class="mini-stat">
                <span class="mini-stat-label">Healthy</span>
                <span id="overview-healthy-count" class="mini-stat-value">0</span>
              </div>
              <div class="mini-stat">
                <span class="mini-stat-label">Avoid</span>
                <span id="overview-holdoff-count" class="mini-stat-value">0</span>
              </div>
            </div>
          </div>
        </div>
        <div class="workspace-grid">
          <div class="surface">
            <h2 class="section-title">Performance vs buy and hold</h2>
            <canvas id="overview-strength-chart" class="canvas-block" width="960" height="280"></canvas>
          </div>
          <div class="surface">
            <h2 class="section-title">Top performer over time</h2>
            <canvas id="overview-leader-chart" class="canvas-block" width="960" height="280"></canvas>
          </div>
        </div>
        <div class="workspace-grid-compact">
          <div class="surface">
            <h2 class="section-title">Highlighted tickers</h2>
            <div id="overview-leaders" class="row-list"></div>
          </div>
          <div class="surface">
            <h2 class="section-title">Recent changes</h2>
            <div id="overview-changes" class="row-list"></div>
          </div>
        </div>
        <div class="surface ranking-table-wrap">
          <h2 class="section-title">Current ranking</h2>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Symbol</th>
                <th>Recommendation</th>
                <th>Vs buy and hold</th>
                <th>Confidence</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody id="ranking-table-body"></tbody>
          </table>
        </div>
        <div class="surface">
          <h2 class="section-title">At a glance</h2>
          <p id="overview-summary" class="summary-copy">Waiting for ranked ticker data.</p>
        </div>
      </section>

      <section id="ticker-panel" class="panel">
        <div class="ticker-toolbar">
          <div class="workspace-head">
            <div class="stat-block wide">
              <span class="stat-label">Ticker</span>
              <div id="ticker-name" class="ticker-title"></div>
              <p id="ticker-subtitle" class="ticker-subtitle"></p>
              <p id="ticker-recommendation-subtitle" class="signal-subtitle"></p>
            </div>
            <div class="stat-block">
              <span class="stat-label">Recommendation</span>
              <div id="ticker-recommendation-badge" class="badge badge-overview">n/a</div>
            </div>
            <div class="stat-block">
              <span class="stat-label">Confidence</span>
              <div id="ticker-confidence-badge" class="badge badge-low">n/a</div>
            </div>
            <div class="stat-block">
              <span class="stat-label">Status</span>
              <div id="ticker-status-badge" class="badge badge-low">n/a</div>
            </div>
            <div class="stat-block">
              <span class="stat-label">Direction</span>
              <div id="ticker-outlook-badge" class="badge badge-overview">n/a</div>
            </div>
            <div class="stat-block">
              <span class="stat-label">Vs buy and hold</span>
              <div id="ticker-vs-buy" class="stat-value">n/a</div>
            </div>
            <div class="stat-block">
              <span class="stat-label">Vs base setup</span>
              <div id="ticker-vs-base" class="stat-value">n/a</div>
            </div>
            <div class="stat-block">
              <span class="stat-label">Rank</span>
              <div id="ticker-rank" class="stat-value">n/a</div>
            </div>
            <div class="stat-block">
              <span class="stat-label">Current setup</span>
              <div id="ticker-active-model" class="stat-copy">n/a</div>
              <div id="ticker-active-source" class="stat-copy">n/a</div>
            </div>
          </div>
        </div>
        <div class="workspace-grid">
          <div class="surface">
            <h2 class="section-title">Recent performance</h2>
            <canvas id="ticker-chart" class="canvas-block compact" width="960" height="230"></canvas>
          </div>
          <div class="surface">
            <h2 class="section-title">Recent ranking</h2>
            <canvas id="ticker-rank-chart" class="canvas-block compact" width="960" height="230"></canvas>
          </div>
        </div>
        <div class="workspace-grid-triple">
          <div class="surface">
            <h2 class="section-title">What this means</h2>
            <div class="signal-list">
              <div class="signal-line">
                <span class="signal-label">Suggested action</span>
                <p id="ticker-action" class="signal-value"></p>
              </div>
              <div class="signal-line">
                <span class="signal-label">Why</span>
                <p id="ticker-summary" class="signal-value"></p>
              </div>
              <div class="signal-line">
                <span class="signal-label">What to watch</span>
                <p id="ticker-watch" class="signal-value"></p>
              </div>
              <div class="signal-line">
                <span class="signal-label">Compared with peers</span>
                <p id="ticker-peer-comparison" class="signal-value"></p>
              </div>
            </div>
          </div>
          <div class="surface">
            <h2 class="section-title">Available setups</h2>
            <table>
              <thead><tr><th>Setup</th><th>Rank</th><th>Gap vs buy/hold</th></tr></thead>
              <tbody id="profile-table"></tbody>
            </table>
          </div>
          <div class="surface">
            <h2 class="section-title">Related reports</h2>
            <div class="report-list" id="report-links"></div>
          </div>
        </div>
        <details class="details-note">
          <summary>Technical details</summary>
          <div id="ops-internals" class="body-copy"></div>
        </details>
      </section>

      <section id="system-panel" class="panel">
        <div class="hero">
          <div>
            <div class="eyebrow">System overview</div>
            <h2>SYSTEM</h2>
            <p class="hero-copy">This page is the dashboard health summary. It should answer three questions quickly: is anything wrong, which tickers need attention, and what should you look at next.</p>
          </div>
          <div class="hero-meta">
            <strong>How to use it</strong>
              <div>Read the top summary first. Only open the advanced sections if you want the supporting evidence and technical detail behind the recommendations.</div>
          </div>
        </div>
        <div class="summary-grid system-summary-grid">
          <div class="surface">
            <div class="kicker">Overall status</div>
            <div class="badge-row" style="margin-top:10px;">
              <div id="system-overall-status" class="badge badge-overview">n/a</div>
            </div>
            <div id="system-overall-copy" class="metric-sub">Checking current dashboard health.</div>
          </div>
          <div class="surface">
            <div class="kicker">Action now</div>
            <div id="system-action-title" class="metric-value">n/a</div>
            <div id="system-action-copy" class="metric-sub">Working out the most useful next step.</div>
          </div>
          <div class="surface">
            <div class="kicker">Needs attention</div>
            <div id="system-attention-count" class="metric-value">0</div>
            <div id="system-attention-copy" class="metric-sub">No attention items yet.</div>
          </div>
          <div class="surface">
            <div class="kicker">Recent change</div>
            <div id="system-change-title" class="metric-value">n/a</div>
            <div id="system-change-copy" class="metric-sub">Waiting for another run to compare against.</div>
          </div>
        </div>
        <div class="surface system-primary-card">
          <div class="section-head" style="margin-top:0; margin-bottom:12px;">
            <div>
              <h3>What to do now</h3>
              <p>Short version first. Open the advanced sections only if you want the detailed evidence.</p>
            </div>
          </div>
          <div id="system-primary-action" class="watch-box">No action summary yet.</div>
          <div id="system-attention-list" class="row-list" style="margin-top:12px;"></div>
        </div>
        <div class="section-head">
          <div>
            <h3>Advanced details</h3>
            <p>These sections hold the deeper diagnostics and supporting artifacts.</p>
          </div>
        </div>
        <div class="advanced-stack">
          <details class="advanced-panel">
            <summary>Shadow check details</summary>
            <div class="advanced-panel-body">
              <div class="shadow-panel">
                <div class="shadow-head">
                  <div class="kicker">Shadow check</div>
                  <div id="shadow-status" class="shadow-status inactive">Inactive</div>
                </div>
                <div id="shadow-meta" class="shadow-meta"></div>
                <div id="shadow-failing" class="shadow-failing"></div>
                <div id="shadow-suggestion" class="shadow-suggestion"></div>
                <details id="shadow-details" class="shadow-table-wrap">
                  <summary>Per-ticker shadow agreement</summary>
                  <table>
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Result</th>
                        <th>Runs</th>
                        <th>Match rate</th>
                        <th>Needed</th>
                      </tr>
                    </thead>
                    <tbody id="shadow-table-body"></tbody>
                  </table>
                </details>
              </div>
            </div>
          </details>
          <details class="advanced-panel">
            <summary>Run change details</summary>
            <div class="advanced-panel-body">
              <div class="delta-panel">
                <div class="delta-head">
                  <div class="kicker">What changed since last run</div>
                  <div id="delta-run-range" class="badge badge-overview">n/a</div>
                </div>
                <div id="delta-sub" class="delta-sub"></div>
                <div class="delta-grid">
                  <div class="delta-item"><div class="k">Setup changes</div><div id="delta-profile-changes" class="v">n/a</div></div>
                  <div class="delta-item"><div class="k">Performance changes</div><div id="delta-score-deltas" class="v">n/a</div></div>
                  <div class="delta-item"><div class="k">Rank changes</div><div id="delta-rank-deltas" class="v">n/a</div></div>
                  <div class="delta-item"><div class="k">Shadow changes</div><div id="delta-shadow-deltas" class="v">n/a</div></div>
                </div>
                <details id="delta-details" class="delta-table-wrap">
                  <summary>Per-ticker changes</summary>
                  <table>
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Setup</th>
                        <th>Source</th>
                        <th>Gap Δ</th>
                        <th>Rank Δ</th>
                      </tr>
                    </thead>
                    <tbody id="delta-table-body"></tbody>
                  </table>
                </details>
              </div>
            </div>
          </details>
          <details class="advanced-panel">
            <summary>Supporting reports</summary>
            <div class="advanced-panel-body">
              <div class="surface surface-soft">
                <div class="section-head" style="margin-top:0; margin-bottom:12px;">
                  <div>
                    <h3>Supporting reports</h3>
                    <p>Open the deeper pages that sit behind this dashboard.</p>
                  </div>
                </div>
                <div id="system-report-links" class="system-links"></div>
              </div>
            </div>
          </details>
          <details class="advanced-panel">
            <summary>Ticker map</summary>
            <div class="advanced-panel-body">
              <div class="surface surface-soft">
                <div class="section-head" style="margin-top:0; margin-bottom:12px;">
                  <div>
                    <h3>Ticker map</h3>
                    <p>Each node shows how each ticker looks right now and how strong its edge is.</p>
                  </div>
                </div>
                <svg id="system-svg"></svg>
              </div>
            </div>
          </details>
        </div>
      </section>
    </main>
  </div>
  <script>
    const systemNodes = {json.dumps(nodes, sort_keys=True)};
    const tickerData = {json.dumps(ticker_payload, sort_keys=True)};
    const rankingData = {json.dumps(ranking_payload, sort_keys=True)};
    const globalReports = {json.dumps(dense_links, sort_keys=True)};
    const shadowGate = {json.dumps(shadow_gate_payload, sort_keys=True)};
    const shadowSuggestion = {json.dumps(shadow_suggestion_payload, sort_keys=True)};
    const rollbackState = {json.dumps(rollback_payload, sort_keys=True)};
    const runDelta = {json.dumps(run_delta_payload, sort_keys=True)};
    const tickerOrder = {json.dumps(tickers_sorted)};
    const latestHealthRun = {json.dumps(latest_health_run)};
    let currentTicker = null;

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function resetNav(activeId) {{
      document.querySelectorAll('.nav-button').forEach((btn) => btn.classList.remove('active'));
      const active = document.getElementById(activeId);
      if (active) active.classList.add('active');
    }}

    function formatTimestamp(value) {{
      const text = String(value || '').trim();
      if (!text) return 'n/a';
      const date = new Date(text);
      if (Number.isNaN(date.getTime())) return text;
      const yyyy = date.getUTCFullYear();
      const mm = String(date.getUTCMonth() + 1).padStart(2, '0');
      const dd = String(date.getUTCDate()).padStart(2, '0');
      const hh = String(date.getUTCHours()).padStart(2, '0');
      const min = String(date.getUTCMinutes()).padStart(2, '0');
      return `${{yyyy}}-${{mm}}-${{dd}} ${{hh}}:${{min}} UTC`;
    }}

    function showPanel(panelId) {{
      ['overview-panel', 'ticker-panel', 'system-panel'].forEach((id) => {{
        const panel = document.getElementById(id);
        panel.classList.toggle('active', id === panelId);
      }});
    }}

    function recommendationClass(label) {{
      const text = String(label || '').toLowerCase();
      if (text.includes('bullish')) return 'badge-shortlist';
      if (text.includes('improving') || text.includes('mixed')) return 'badge-constructive';
      if (text.includes('caution')) return 'badge-watch';
      if (text.includes('avoid')) return 'badge-hold-off';
      return 'badge-overview';
    }}

    function recommendationLabel(label) {{
      const text = String(label || '').trim().toLowerCase();
      if (text === 'bullish setup') return 'Strong setup';
      if (text === 'improving trend') return 'Getting better';
      if (text === 'caution') return 'Needs watching';
      if (text === 'mixed trend') return 'Mixed signals';
      return label || 'n/a';
    }}

    function sourceLabel(label) {{
      const text = String(label || '').trim().toLowerCase();
      if (text === 'selection_state') return 'Saved choice';
      if (text === 'ensemble_v3') return 'Auto-adjusted';
      if (text === 'shadow_gate') return 'Shadow-checked';
      return label || 'n/a';
    }}

    function formatSymbolList(symbols, limit = 3) {{
      const clean = Array.from(new Set((symbols || []).map((value) => String(value || '').trim()).filter(Boolean)));
      if (!clean.length) return '';
      if (clean.length <= limit) return clean.join(', ');
      return `${{clean.slice(0, limit).join(', ')}} +${{clean.length - limit}} more`;
    }}

    function setBadgeState(element, label, variant) {{
      if (!element) return;
      element.className = `badge ${{variant}}`;
      element.textContent = label;
    }}

    function confidenceClass(label) {{
      const text = String(label || '').toLowerCase();
      if (text.includes('high')) return 'badge-shortlist';
      if (text.includes('medium')) return 'badge-medium';
      return 'badge-low';
    }}

    function statusClass(label) {{
      const text = String(label || '').toLowerCase();
      if (text.includes('healthy')) return 'badge-healthy';
      if (text.includes('watch')) return 'badge-watch';
      if (text.includes('critical')) return 'badge-critical';
      return 'badge-low';
    }}

    function rankedTickers() {{
      return rankingData.map((item) => item.symbol).filter((symbol) => !!tickerData[symbol]);
    }}

    function rankingEntry(symbol) {{
      return rankingData.find((item) => item.symbol === symbol) || null;
    }}

    function collectTickerLinks(symbol, data) {{
      const links = [];
      globalReports.forEach((item) => links.push(item));
      (data.profiles || []).forEach((item) => {{
        if (item.visual_report) links.push({{ label: `${{symbol}} ${{item.profile}} report`, url: item.visual_report }});
        if (item.pairwise_report) links.push({{ label: `${{symbol}} pairwise`, url: item.pairwise_report }});
      }});
      const seen = new Set();
      return links.filter((item) => {{
        const key = `${{item.label}}::${{item.url}}`;
        if (!item.url || seen.has(key)) return false;
        seen.add(key);
        return true;
      }});
    }}

    function primaryTickerReport(symbol, data) {{
      const active = (data.profiles || []).find((item) => item.profile === data.active_profile && item.visual_report);
      if (active && active.visual_report) return {{ label: `${{symbol}} ${{active.profile}} report`, url: active.visual_report }};
      const fallback = (data.profiles || []).find((item) => item.visual_report);
      return fallback && fallback.visual_report ? {{ label: `${{symbol}} ${{fallback.profile}} report`, url: fallback.visual_report }} : null;
    }}

    function drawCanvasMessage(ctx, canvas, message) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#64748b';
      ctx.font = '14px Arial';
      ctx.fillText(message, 24, 32);
    }}

    function drawLineChart(canvasId, series, options = {{}}) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas && canvas.getContext ? canvas.getContext('2d') : null;
      if (!ctx || !canvas) return;

      const validSeries = series.filter((item) => Array.isArray(item.points) && item.points.some((point) => point != null));
      const allValues = validSeries.flatMap((item) => item.points.filter((point) => point != null));
      if (allValues.length < 1) {{
        drawCanvasMessage(ctx, canvas, options.emptyMessage || 'Need at least one point to draw this chart.');
        return;
      }}

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const minV = Math.min(...allValues, options.includeZero ? 0 : Math.min(...allValues));
      const maxV = Math.max(...allValues, options.includeZero ? 0 : Math.max(...allValues));
      const range = Math.max(maxV - minV, 1e-6);
      const pad = 34;
      const pointCount = Math.max(...validSeries.map((item) => item.points.length), 2);
      const toX = (index) => pad + (index * (canvas.width - pad * 2)) / Math.max(pointCount - 1, 1);
      const toY = options.invertY
        ? (value) => pad + ((value - minV) * (canvas.height - pad * 2)) / range
        : (value) => canvas.height - pad - ((value - minV) * (canvas.height - pad * 2)) / range;

      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      [0.25, 0.5, 0.75].forEach((ratio) => {{
        const y = pad + ratio * (canvas.height - pad * 2);
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(canvas.width - pad, y);
        ctx.stroke();
      }});
      if (options.includeZero && minV <= 0 && maxV >= 0) {{
        ctx.strokeStyle = '#cbd5e1';
        ctx.beginPath();
        ctx.moveTo(pad, toY(0));
        ctx.lineTo(canvas.width - pad, toY(0));
        ctx.stroke();
      }}

      validSeries.forEach((item) => {{
        let started = false;
        const pointCoords = [];
        ctx.beginPath();
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2.2;
        item.points.forEach((value, index) => {{
          if (value == null) return;
          const x = toX(index);
          const y = toY(value);
          pointCoords.push([x, y]);
          if (!started) {{
            ctx.moveTo(x, y);
            started = true;
          }} else {{
            ctx.lineTo(x, y);
          }}
        }});
        if (started) ctx.stroke();
        pointCoords.forEach(([x, y]) => {{
          ctx.fillStyle = item.color;
          ctx.beginPath();
          ctx.arc(x, y, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }});
      }});

      ctx.font = '12px Arial';
      validSeries.forEach((item, index) => {{
        const x = pad + index * 160;
        ctx.fillStyle = item.color;
        ctx.fillRect(x, 14, 16, 3);
        ctx.fillStyle = '#475569';
        ctx.fillText(item.label, x + 22, 19);
      }});
    }}

    function drawBarChart(canvasId, rows) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas && canvas.getContext ? canvas.getContext('2d') : null;
      if (!ctx || !canvas) return;

      const chartRows = rows.filter((row) => row.value != null);
      if (!chartRows.length) {{
        drawCanvasMessage(ctx, canvas, 'No current edge values available.');
        return;
      }}

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const pad = 36;
      const maxAbs = Math.max(...chartRows.map((row) => Math.abs(row.value)), 1);
      const zeroY = canvas.height / 2;
      const slot = (canvas.width - pad * 2) / chartRows.length;

      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, zeroY);
      ctx.lineTo(canvas.width - pad, zeroY);
      ctx.stroke();

      chartRows.forEach((row, index) => {{
        const barWidth = Math.min(42, slot * 0.58);
        const x = pad + index * slot + (slot - barWidth) / 2;
        const height = (Math.abs(row.value) / maxAbs) * (canvas.height / 2 - pad - 18);
        const y = row.value >= 0 ? zeroY - height : zeroY;
        ctx.fillStyle = row.value >= 0 ? '#111111' : '#71717a';
        ctx.fillRect(x, y, barWidth, height);
        ctx.fillStyle = '#475569';
        ctx.font = '11px Arial';
        ctx.fillText(row.symbol, x, canvas.height - 12);
      }});
    }}

    function renderSystemOverview() {{
      const tickers = Object.values(tickerData || {{}});
      const criticalSymbols = tickers.filter((item) => item.status === 'Critical').map((item) => item.symbol);
      const watchSymbols = tickers.filter((item) => item.status === 'Watch').map((item) => item.symbol);
      const rollbackRows = Array.isArray(rollbackState.symbols)
        ? rollbackState.symbols.filter((item) => item && item.should_rollback)
        : [];
      const rollbackSymbols = rollbackRows.map((item) => item.symbol).filter(Boolean);
      const shadowNeedsReview = shadowGate.enabled && shadowGate.overall_gate_passed === false;
      const urgentSymbols = Array.from(new Set([...rollbackSymbols, ...criticalSymbols]));
      const attentionSymbols = Array.from(new Set([...urgentSymbols, ...watchSymbols]));

      let overallLabel = 'Healthy';
      let overallVariant = 'badge-healthy';
      let overallCopy = 'No urgent issues are being flagged right now.';
      if (urgentSymbols.length) {{
        overallLabel = 'Needs attention';
        overallVariant = 'badge-critical';
        overallCopy = `${{formatSymbolList(urgentSymbols)}} ${{urgentSymbols.length === 1 ? 'needs' : 'need'}} immediate review.`;
      }} else if (watchSymbols.length) {{
        overallLabel = 'Watch list';
        overallVariant = 'badge-watch';
        overallCopy = `${{formatSymbolList(watchSymbols)}} ${{watchSymbols.length === 1 ? 'is' : 'are'}} worth watching, but there is nothing urgent.`;
      }}
      if (shadowNeedsReview) {{
        overallCopy += ' The shadow check is also asking for a closer look.';
      }}
      setBadgeState(document.getElementById('system-overall-status'), overallLabel, overallVariant);
      document.getElementById('system-overall-copy').textContent = overallCopy;

      let actionTitle = 'All clear';
      let actionCopy = 'No urgent action is needed right now.';
      if (rollbackSymbols.length) {{
        const first = rollbackRows[0] || null;
        const saferSetup = first && first.rollback_profile
          ? `safer ${{first.rollback_profile}} setup`
          : 'safer setup';
        actionTitle = rollbackSymbols.length === 1 ? `Review ${{rollbackSymbols[0]}}` : `Review ${{rollbackSymbols.length}} tickers`;
        actionCopy = rollbackSymbols.length === 1
          ? `${{rollbackSymbols[0]}} is currently the clearest candidate to move back to the ${{saferSetup}}.`
          : `Start with ${{formatSymbolList(rollbackSymbols)}}. These tickers are currently being flagged for a safer fallback.`;
      }} else if (criticalSymbols.length) {{
        actionTitle = criticalSymbols.length === 1 ? `Check ${{criticalSymbols[0]}}` : `Check ${{criticalSymbols.length}} critical tickers`;
        actionCopy = `Focus on ${{formatSymbolList(criticalSymbols)}} first. These names currently have the strongest warning signals.`;
      }} else if (shadowNeedsReview) {{
        actionTitle = 'Review shadow check';
        actionCopy = 'The dashboard is usable, but its shadow validation is not passing right now. Open the shadow details if you want the evidence.';
      }} else if (watchSymbols.length) {{
        actionTitle = 'Keep watching';
        actionCopy = `Keep an eye on ${{formatSymbolList(watchSymbols)}} over the next run or two, but there is no urgent action needed.`;
      }} else if (!runDelta.has_previous) {{
        actionTitle = 'Build history';
        actionCopy = 'This is still an early run. The dashboard will become more informative once it has another run to compare against.';
      }}
      document.getElementById('system-action-title').textContent = actionTitle;
      document.getElementById('system-action-copy').textContent = actionCopy;

      document.getElementById('system-attention-count').textContent = String(attentionSymbols.length);
      document.getElementById('system-attention-copy').textContent = attentionSymbols.length
        ? `${{formatSymbolList(attentionSymbols)}} ${{attentionSymbols.length === 1 ? 'is' : 'are'}} the main ticker focus right now.`
        : 'No ticker is being singled out right now.';

      let changeTitle = 'No baseline yet';
      let changeCopy = 'You need two runs before the dashboard can describe what changed.';
      if (runDelta.has_previous) {{
        const setupChanges = Number(runDelta.profile_change_count || 0) + Number(runDelta.source_change_count || 0);
        const movementCount = Number(runDelta.gap_up_count || 0) + Number(runDelta.gap_down_count || 0);
        if (setupChanges === 0 && movementCount === 0) {{
          changeTitle = 'Mostly steady';
          changeCopy = 'Very little changed from the previous run.';
        }} else {{
          changeTitle = 'Updated';
          const parts = [];
          if (setupChanges > 0) parts.push(`${{setupChanges}} setup change${{setupChanges === 1 ? '' : 's'}}`);
          if (movementCount > 0) parts.push(`${{movementCount}} performance move${{movementCount === 1 ? '' : 's'}}`);
          if (parts.length === 0) parts.push('small changes only');
          changeCopy = `Compared with the previous run, there were ${{parts.join(' and ')}}.`;
        }}
      }}
      document.getElementById('system-change-title').textContent = changeTitle;
      document.getElementById('system-change-copy').textContent = changeCopy;

      const primaryActionEl = document.getElementById('system-primary-action');
      primaryActionEl.textContent = actionCopy;

      const attentionListEl = document.getElementById('system-attention-list');
      attentionListEl.innerHTML = '';
      const items = [];
      rollbackRows.forEach((item) => {{
        items.push({{
          title: item.symbol || 'Ticker',
          meta: 'Safer fallback suggested',
          note: item['rollback_profile']
            ? `The system is recommending a move back to the ${{item['rollback_profile']}} setup.`
            : 'The system is recommending a safer fallback for this ticker.',
        }});
      }});
      criticalSymbols.filter((symbol) => !rollbackSymbols.includes(symbol)).forEach((symbol) => {{
        items.push({{
          title: symbol,
          meta: 'Urgent review',
          note: 'This ticker has the strongest current warning signals and should be checked first.',
        }});
      }});
      watchSymbols.filter((symbol) => !urgentSymbols.includes(symbol)).slice(0, 4).forEach((symbol) => {{
        items.push({{
          title: symbol,
          meta: 'Watch list',
          note: 'Keep an eye on this ticker, but it does not look urgent right now.',
        }});
      }});
      if (!items.length) {{
        items.push({{
          title: 'No urgent issues',
          meta: 'System looks stable',
          note: shadowNeedsReview
            ? 'Ticker recommendations look calm overall, but the shadow check is still worth reviewing.'
            : 'The dashboard looks stable right now. Open the advanced sections only if you want deeper detail.',
        }});
      }}
      items.slice(0, 6).forEach((item) => {{
        const row = document.createElement('div');
        row.className = 'row-item';
        row.innerHTML = `
          <div class="row-item-main">
            <strong>${{escapeHtml(item.title || '')}}</strong>
            <span class="row-item-meta">${{escapeHtml(item.meta || '')}}</span>
          </div>
          <div class="row-item-note">${{escapeHtml(item.note || '')}}</div>
        `;
        attentionListEl.appendChild(row);
      }});
    }}

    function showOverview() {{
      showPanel('overview-panel');
      resetNav('nav-overview');
      renderOverview();
    }}

    function showSystem() {{
      showPanel('system-panel');
      resetNav('nav-system');
      renderSystemOverview();
      renderShadowPanel();
      renderRunDelta();
      renderSystemReports();
      drawSystem();
    }}

    function showTicker(symbol) {{
      if (!tickerData[symbol]) return;
      currentTicker = symbol;
      showPanel('ticker-panel');
      resetNav(`nav-${{symbol}}`);
      renderTicker(symbol);
    }}

    function renderOverview() {{
      const rankedEntries = rankingData.filter((entry) => !!tickerData[entry.symbol]);
      const ranked = rankedEntries.map((entry) => tickerData[entry.symbol]).filter(Boolean);
      const shortlistCount = ranked.filter((item) => item.recommendation === 'Bullish setup').length;
      const healthyCount = ranked.filter((item) => item.status === 'Healthy').length;
      const holdoffCount = ranked.filter((item) => item.recommendation === 'Avoid for now' || item.status === 'Critical').length;
      const leaderEntry = rankedEntries[0] || null;
      const leader = leaderEntry ? tickerData[leaderEntry.symbol] : null;
      const leaderSignal = leader
        ? recommendationLabel(leader.recommendation) + (leader.recommendation_subtitle ? ` (${{leader.recommendation_subtitle}})` : '')
        : '';

      document.getElementById('overview-run-label').textContent = formatTimestamp(runDelta.latest_run_timestamp || latestHealthRun || 'n/a');
      document.getElementById('overview-shortlist-count').textContent = String(shortlistCount);
      document.getElementById('overview-healthy-count').textContent = String(healthyCount);
      document.getElementById('overview-holdoff-count').textContent = String(holdoffCount);
      document.getElementById('overview-leader').textContent = leader ? leader.symbol : 'n/a';
      document.getElementById('overview-summary').textContent = leader
        ? `${{leader.symbol}} is currently on top with ${{leaderSignal}}. It is at ${{leader.active_vs_buy_pct_text || 'n/a'}} versus buy-and-hold. ${{healthyCount}} tickers look healthy, ${{shortlistCount}} look strong, and ${{holdoffCount}} currently look best avoided. ${{(leaderEntry && leaderEntry.comparison) || ''}}`
        : 'Waiting for ranked ticker data.';

      const leadersWrap = document.getElementById('overview-leaders');
      leadersWrap.innerHTML = '';
      rankedEntries.slice(0, 6).forEach((entry) => {{
        const data = tickerData[entry.symbol];
        if (!data) return;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'row-item-button';
        button.innerHTML = `
          <div class="row-item">
            <div class="row-item-main">
              <strong>${{escapeHtml(entry.symbol)}}</strong>
              <span class="row-item-meta">${{escapeHtml(data.active_vs_buy_pct_text || 'n/a')}}</span>
            </div>
            <div class="row-item-note">${{escapeHtml(data.recommendation_subtitle || data.recommendation || '')}}</div>
            <div class="row-item-note">${{escapeHtml(entry.comparison || entry.reason || '')}}</div>
          </div>
        `;
        button.addEventListener('click', () => showTicker(entry.symbol));
        leadersWrap.appendChild(button);
      }});
      if (!leadersWrap.children.length) {{
        leadersWrap.innerHTML = '<div class="row-item"><div class="row-item-note">No ranked tickers yet.</div></div>';
      }}

      const changesWrap = document.getElementById('overview-changes');
      changesWrap.innerHTML = '';
      const changes = Array.isArray(runDelta.symbol_changes) ? runDelta.symbol_changes : [];
      if (!changes.length) {{
        changesWrap.innerHTML = '<div class="row-item"><div class="row-item-note">Need two runs to show what changed.</div></div>';
      }} else {{
        changes.slice(0, 8).forEach((item) => {{
          const row = document.createElement('div');
          const gapText = item.gap_delta == null ? 'n/a' : `${{(Number(item.gap_delta) * 100).toFixed(2)}}%`;
          const rankText = item.rank_delta == null ? 'n/a' : Number(item.rank_delta).toFixed(2);
          const profileText = item.profile_changed
            ? `${{item.profile_previous || 'n/a'}} → ${{item.profile_latest || 'n/a'}}`
            : (item.profile_latest || item.profile_previous || 'n/a');
          row.className = 'row-item';
          row.innerHTML = `
            <div class="row-item-main">
              <strong>${{escapeHtml(item.symbol || '')}}</strong>
              <span class="row-item-meta">${{escapeHtml(gapText)}}</span>
            </div>
            <div class="row-item-note">${{escapeHtml(profileText)}} · rank Δ ${{escapeHtml(rankText)}}</div>
          `;
          changesWrap.appendChild(row);
        }});
      }}

      const rankingBody = document.getElementById('ranking-table-body');
      rankingBody.innerHTML = '';
      if (!rankedEntries.length) {{
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="6">No ranked tickers available.</td>';
        rankingBody.appendChild(tr);
      }}
      rankedEntries.forEach((entry) => {{
        const data = tickerData[entry.symbol];
        if (!data) return;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>#${{entry.rank}}</td>
          <td>${{escapeHtml(entry.symbol)}}</td>
          <td>${{escapeHtml(recommendationLabel(data.recommendation))}}</td>
          <td>${{escapeHtml(data.active_vs_buy_pct_text || 'n/a')}}</td>
          <td>${{escapeHtml(data.confidence || 'n/a')}}</td>
          <td>${{escapeHtml(entry.comparison || entry.reason || '')}}</td>
        `;
        rankingBody.appendChild(tr);
      }});

      drawBarChart(
        'overview-strength-chart',
        rankedEntries.slice(0, 10).map((entry) => {{
          const data = tickerData[entry.symbol];
          return {{
            symbol: entry.symbol,
            value: data ? data.active_vs_buy_pct_value : null,
          }};
        }})
      );
      drawLineChart(
        'overview-leader-chart',
        leader
          ? [
              {{
                label: 'active edge',
                color: '#111111',
                points: (leader.history || []).map((item) => item.active_gap),
              }},
              {{
                label: 'selected edge',
                color: '#94a3b8',
                points: (leader.history || []).map((item) => item.selected_gap),
              }},
            ]
          : [],
        {{ includeZero: true, emptyMessage: 'No leader trend is available yet.' }}
      );
    }}

    function drawSystem() {{
      const svg = document.getElementById('system-svg');
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const width = svg.clientWidth || 900;
      const height = svg.clientHeight || 420;
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) * 0.3;
      const ns = 'http://www.w3.org/2000/svg';

      const hub = document.createElementNS(ns, 'circle');
      hub.setAttribute('cx', centerX.toString());
      hub.setAttribute('cy', centerY.toString());
      hub.setAttribute('r', '34');
      hub.setAttribute('fill', '#111111');
      svg.appendChild(hub);

      const hubLabel = document.createElementNS(ns, 'text');
      hubLabel.setAttribute('x', centerX.toString());
      hubLabel.setAttribute('y', (centerY + 5).toString());
      hubLabel.setAttribute('fill', '#eff6ff');
      hubLabel.setAttribute('font-size', '11');
      hubLabel.setAttribute('font-weight', '700');
      hubLabel.setAttribute('text-anchor', 'middle');
      hubLabel.textContent = 'Pulse';
      svg.appendChild(hubLabel);

      const count = Math.max(systemNodes.length, 1);
      systemNodes.forEach((node, idx) => {{
        const angle = (Math.PI * 2 * idx) / count;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        const line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', centerX.toString());
        line.setAttribute('y1', centerY.toString());
        line.setAttribute('x2', x.toString());
        line.setAttribute('y2', y.toString());
        line.setAttribute('stroke', '#334155');
        line.setAttribute('stroke-width', '2');
        line.setAttribute('opacity', '0.7');
        svg.appendChild(line);

        const group = document.createElementNS(ns, 'g');
        group.style.cursor = 'pointer';
        group.addEventListener('click', () => showTicker(node.symbol));
        const magnitude = Math.abs(node.active_vs_buy_pct || 0);
        const r = Math.max(20, Math.min(42, 18 + magnitude * 0.6));
        const statusColor = node.status === 'Critical'
          ? '#ef4444'
          : node.status === 'Watch'
          ? '#f59e0b'
          : '#22c55e';

        const circle = document.createElementNS(ns, 'circle');
        circle.setAttribute('cx', x.toString());
        circle.setAttribute('cy', y.toString());
        circle.setAttribute('r', r.toString());
        circle.setAttribute('fill', statusColor);
        circle.setAttribute('opacity', '0.95');
        group.appendChild(circle);

        const label = document.createElementNS(ns, 'text');
        label.setAttribute('x', x.toString());
        label.setAttribute('y', (y + 4).toString());
        label.setAttribute('fill', '#ffffff');
        label.setAttribute('font-size', '11');
        label.setAttribute('font-weight', '700');
        label.setAttribute('text-anchor', 'middle');
        label.textContent = node.symbol || '';
        group.appendChild(label);

        svg.appendChild(group);
      }});
    }}

    function renderShadowPanel() {{
      const statusEl = document.getElementById('shadow-status');
      const metaEl = document.getElementById('shadow-meta');
      const failingEl = document.getElementById('shadow-failing');
      const suggestionEl = document.getElementById('shadow-suggestion');
      const detailsEl = document.getElementById('shadow-details');
      const tableBody = document.getElementById('shadow-table-body');
      statusEl.classList.remove('pass', 'fail', 'inactive');
      tableBody.innerHTML = '';

      if (!shadowGate.enabled) {{
        statusEl.textContent = 'Inactive';
        statusEl.classList.add('inactive');
        const fallback = shadowGate.gate_json_url
          ? 'Shadow gate artifact available but not readable in this view.'
          : 'Shadow evaluation is not enabled for this run.';
        metaEl.textContent = fallback;
        failingEl.textContent = '';
        const stateRatio = shadowSuggestion.state_active_min_match_ratio == null
          ? null
          : `${{(Number(shadowSuggestion.state_active_min_match_ratio) * 100).toFixed(1)}}%`;
        const stateText = shadowSuggestion.state_active_window_runs == null
          ? ''
          : ` Active setting: window_runs=${{shadowSuggestion.state_active_window_runs}}, min_match_ratio=${{stateRatio ?? 'n/a'}}.`;
        if (shadowSuggestion.enabled && shadowSuggestion.accepted) {{
          const ratio = shadowSuggestion.recommended_min_match_ratio == null
            ? 'n/a'
            : `${{(Number(shadowSuggestion.recommended_min_match_ratio) * 100).toFixed(1)}}%`;
          suggestionEl.innerHTML = `Suggested config: window_runs=${{shadowSuggestion.recommended_window_runs ?? 'n/a'}}, min_match_ratio=${{ratio}}.${{stateText}}`;
        }} else {{
          suggestionEl.textContent = stateText.trim();
        }}
        detailsEl.style.display = 'none';
        return;
      }}

      const passed = shadowGate.overall_gate_passed === true;
      statusEl.textContent = passed ? 'PASS' : 'FAIL';
      statusEl.classList.add(passed ? 'pass' : 'fail');
      const windowRuns = shadowGate.window_runs == null ? 'n/a' : String(shadowGate.window_runs);
      const minRatio = shadowGate.min_match_ratio == null ? 'n/a' : `${{(shadowGate.min_match_ratio * 100).toFixed(1)}}%`;
      const links = [];
      if (shadowGate.comparison_url) links.push(`<a href="${{shadowGate.comparison_url}}" target="_blank" rel="noopener noreferrer">comparison</a>`);
      if (shadowGate.gate_json_url) links.push(`<a href="${{shadowGate.gate_json_url}}" target="_blank" rel="noopener noreferrer">gate json</a>`);
      const linkSuffix = links.length ? ` (${{links.join(' · ')}})` : '';
      metaEl.innerHTML = `Window: ${{windowRuns}} runs · Required agreement: ${{minRatio}}${{linkSuffix}}`;

      const failing = Array.isArray(shadowGate.failing_symbols) ? shadowGate.failing_symbols : [];
      failingEl.textContent = failing.length === 0
        ? 'No failing symbols in current shadow window.'
        : `Failing symbols: ${{failing.join(', ')}}`;

      if (!shadowSuggestion.enabled) {{
        suggestionEl.textContent = '';
      }} else if (shadowSuggestion.accepted) {{
        const ratio = shadowSuggestion.recommended_min_match_ratio == null
          ? 'n/a'
          : `${{(Number(shadowSuggestion.recommended_min_match_ratio) * 100).toFixed(1)}}%`;
        const bits = [
          `Suggested config: window_runs=${{shadowSuggestion.recommended_window_runs ?? 'n/a'}}`,
          `min_match_ratio=${{ratio}}`
        ];
        const links = [];
        if (shadowSuggestion.suggestion_html_url) links.push(`<a href="${{shadowSuggestion.suggestion_html_url}}" target="_blank" rel="noopener noreferrer">suggestion report</a>`);
        if (shadowSuggestion.suggestion_json_url) links.push(`<a href="${{shadowSuggestion.suggestion_json_url}}" target="_blank" rel="noopener noreferrer">suggestion json</a>`);
        if (shadowSuggestion.suggestion_history_url) links.push(`<a href="${{shadowSuggestion.suggestion_history_url}}" target="_blank" rel="noopener noreferrer">suggestion history</a>`);
        if (shadowSuggestion.state_json_url) links.push(`<a href="${{shadowSuggestion.state_json_url}}" target="_blank" rel="noopener noreferrer">active state</a>`);
        const suffix = links.length ? ` (${{links.join(' · ')}})` : '';
        const stateRatio = shadowSuggestion.state_active_min_match_ratio == null
          ? null
          : `${{(Number(shadowSuggestion.state_active_min_match_ratio) * 100).toFixed(1)}}%`;
        const stateText = shadowSuggestion.state_active_window_runs == null
          ? ''
          : ` Active setting: window_runs=${{shadowSuggestion.state_active_window_runs}}, min_match_ratio=${{stateRatio ?? 'n/a'}}.`;
        suggestionEl.innerHTML = bits.join(' · ') + suffix + stateText;
      }} else {{
        const reasons = Array.isArray(shadowSuggestion.reasons) && shadowSuggestion.reasons.length
          ? shadowSuggestion.reasons.join('; ')
          : 'insufficient history';
        const stateRatio = shadowSuggestion.state_active_min_match_ratio == null
          ? null
          : `${{(Number(shadowSuggestion.state_active_min_match_ratio) * 100).toFixed(1)}}%`;
        const stateText = shadowSuggestion.state_active_window_runs == null
          ? ''
          : ` Active setting: window_runs=${{shadowSuggestion.state_active_window_runs}}, min_match_ratio=${{stateRatio ?? 'n/a'}}.`;
        suggestionEl.textContent = `Auto-suggestion pending: ${{reasons}}.${{stateText}}`;
      }}

      const rows = Array.isArray(shadowGate.symbols) ? shadowGate.symbols : [];
      if (rows.length === 0) {{
        detailsEl.style.display = 'none';
        return;
      }}
      detailsEl.style.display = '';
      rows.forEach((row) => {{
        const tr = document.createElement('tr');
        const gate = row.gate_passed ? 'pass' : 'fail';
        const ratio = row.match_ratio == null ? 'n/a' : `${{(row.match_ratio * 100).toFixed(1)}}%`;
        const required = row.min_match_ratio == null ? 'n/a' : `${{(row.min_match_ratio * 100).toFixed(1)}}%`;
        const runs = `${{row.runs_in_window ?? 'n/a'}}/${{row.window_runs_required ?? 'n/a'}}`;
        tr.innerHTML = `<td>${{escapeHtml(row.symbol || '')}}</td><td>${{gate}}</td><td>${{runs}}</td><td>${{ratio}}</td><td>${{required}}</td>`;
        tableBody.appendChild(tr);
      }});
    }}

    function renderRunDelta() {{
      const rangeEl = document.getElementById('delta-run-range');
      const subEl = document.getElementById('delta-sub');
      const profileEl = document.getElementById('delta-profile-changes');
      const scoreEl = document.getElementById('delta-score-deltas');
      const rankEl = document.getElementById('delta-rank-deltas');
      const shadowEl = document.getElementById('delta-shadow-deltas');
      const detailsEl = document.getElementById('delta-details');
      const tableBody = document.getElementById('delta-table-body');
      tableBody.innerHTML = '';

      if (!runDelta.has_previous) {{
        rangeEl.textContent = 'Need 2 runs';
        subEl.textContent = 'Not enough history yet to compute run-to-run changes.';
        profileEl.textContent = 'n/a';
        scoreEl.textContent = 'n/a';
        rankEl.textContent = 'n/a';
        shadowEl.textContent = 'n/a';
        detailsEl.style.display = 'none';
        return;
      }}

      const latest = runDelta.latest_run_timestamp || 'latest';
      const previous = runDelta.previous_run_timestamp || 'previous';
      rangeEl.textContent = `${{previous}} → ${{latest}}`;
      subEl.textContent = `Compared symbols: ${{runDelta.symbols_compared ?? 0}}`;

      profileEl.textContent = `${{runDelta.profile_change_count ?? 0}} profile changes · ${{runDelta.source_change_count ?? 0}} source changes`;

      const scoreParts = [`up ${{runDelta.gap_up_count ?? 0}}`, `down ${{runDelta.gap_down_count ?? 0}}`];
      if (runDelta.largest_gap_up_symbol && runDelta.largest_gap_up_delta != null) {{
        scoreParts.push(`best ${{runDelta.largest_gap_up_symbol}} ${{(Number(runDelta.largest_gap_up_delta) * 100).toFixed(2)}}%`);
      }}
      if (runDelta.largest_gap_down_symbol && runDelta.largest_gap_down_delta != null) {{
        scoreParts.push(`worst ${{runDelta.largest_gap_down_symbol}} ${{(Number(runDelta.largest_gap_down_delta) * 100).toFixed(2)}}%`);
      }}
      scoreEl.textContent = scoreParts.join(' · ');

      rankEl.textContent = `improved ${{runDelta.rank_improved_count ?? 0}} · worsened ${{runDelta.rank_worsened_count ?? 0}}`;

      if (runDelta.shadow_match_ratio_latest == null || runDelta.shadow_match_ratio_previous == null) {{
        shadowEl.textContent = 'n/a';
      }} else {{
        const latestPct = (Number(runDelta.shadow_match_ratio_latest) * 100).toFixed(1);
        const prevPct = (Number(runDelta.shadow_match_ratio_previous) * 100).toFixed(1);
        const deltaPct = runDelta.shadow_match_ratio_delta == null ? 'n/a' : `${{(Number(runDelta.shadow_match_ratio_delta) * 100).toFixed(1)}}%`;
        shadowEl.textContent = `match ${{prevPct}}% → ${{latestPct}}% (Δ ${{deltaPct}}) · recovered ${{runDelta.shadow_recovered_matches ?? 0}} · new mismatches ${{runDelta.shadow_new_mismatches ?? 0}}`;
      }}

      const rows = Array.isArray(runDelta.symbol_changes) ? runDelta.symbol_changes : [];
      if (rows.length === 0) {{
        detailsEl.style.display = 'none';
        return;
      }}
      detailsEl.style.display = '';
      rows.forEach((row) => {{
        const tr = document.createElement('tr');
        const profileText = row.profile_changed
          ? `${{row.profile_previous || 'n/a'}} → ${{row.profile_latest || 'n/a'}}`
          : (row.profile_latest || row.profile_previous || 'n/a');
        const sourceText = row.source_changed
          ? `${{row.source_previous || 'n/a'}} → ${{row.source_latest || 'n/a'}}`
          : (row.source_latest || row.source_previous || 'n/a');
        const gapText = row.gap_delta == null ? 'n/a' : `${{(Number(row.gap_delta) * 100).toFixed(2)}}%`;
        const rankText = row.rank_delta == null ? 'n/a' : Number(row.rank_delta).toFixed(2);
        tr.innerHTML = `<td>${{escapeHtml(row.symbol || '')}}</td><td>${{escapeHtml(profileText)}}</td><td>${{escapeHtml(sourceText)}}</td><td>${{gapText}}</td><td>${{rankText}}</td>`;
        tableBody.appendChild(tr);
      }});
    }}

    function renderSystemReports() {{
      const wrap = document.getElementById('system-report-links');
      wrap.innerHTML = '';
      if (!globalReports.length) {{
        wrap.textContent = 'No extra supporting reports were produced for this run.';
        return;
      }}
      globalReports.forEach((item) => {{
        const link = document.createElement('a');
        link.href = item.url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = item.label;
        wrap.appendChild(link);
      }});
    }}

    function renderTicker(symbol) {{
      const data = tickerData[symbol];
      if (!data) return;
      const entry = rankingEntry(symbol);

      document.getElementById('ticker-name').textContent = symbol;
      document.getElementById('ticker-subtitle').textContent = (entry && entry.reason) || data.summary || '';
      document.getElementById('ticker-vs-buy').textContent = data.active_vs_buy_pct_text || 'n/a';
      document.getElementById('ticker-vs-base').textContent = data.active_vs_base_pct_text || 'n/a';
      document.getElementById('ticker-active-model').textContent = data.active_profile || 'n/a';
      document.getElementById('ticker-active-source').textContent = `Source: ${{sourceLabel(data.active_profile_source)}}`;
      document.getElementById('ticker-rank').textContent = entry ? `#${{entry.rank}} of ${{rankingData.length}}` : 'n/a';
      document.getElementById('ticker-action').textContent = data.action || 'n/a';
      document.getElementById('ticker-summary').textContent = data.summary || '';
      document.getElementById('ticker-watch').textContent = data.what_to_watch || 'n/a';
      document.getElementById('ticker-peer-comparison').textContent = (entry && entry.comparison) || 'n/a';
      document.getElementById('ticker-recommendation-subtitle').textContent = data.recommendation_subtitle || '';

      const recommendationBadge = document.getElementById('ticker-recommendation-badge');
      recommendationBadge.className = `badge ${{recommendationClass(data.recommendation)}}`;
      recommendationBadge.textContent = recommendationLabel(data.recommendation);

      const confidenceBadge = document.getElementById('ticker-confidence-badge');
      confidenceBadge.className = `badge ${{confidenceClass(data.confidence)}}`;
      confidenceBadge.textContent = `${{data.confidence || 'n/a'}} confidence`;

      const statusBadge = document.getElementById('ticker-status-badge');
      statusBadge.className = `badge ${{statusClass(data.status)}}`;
      statusBadge.textContent = data.status || 'n/a';

      const outlookBadge = document.getElementById('ticker-outlook-badge');
      outlookBadge.className = 'badge badge-overview';
      outlookBadge.textContent = data.outlook || 'n/a';

      const ops = [
        `Selected profile: ${{data.selected_profile || 'n/a'}}`,
        `Current source: ${{sourceLabel(data.active_profile_source)}}`,
        `Regime: ${{data.regime || 'n/a'}}`,
        `Ensemble confidence: ${{data.ensemble_confidence == null ? 'n/a' : Number(data.ensemble_confidence).toFixed(3)}}`
      ];
      if (shadowGate.enabled) {{
        const row = (shadowGate.symbols || []).find((item) => item.symbol === symbol);
        if (row) {{
          const ratio = row.match_ratio == null ? 'n/a' : `${{(row.match_ratio * 100).toFixed(1)}}%`;
          const required = row.min_match_ratio == null ? 'n/a' : `${{(row.min_match_ratio * 100).toFixed(1)}}%`;
          ops.push(`Shadow agreement: ${{row.gate_passed ? 'pass' : 'fail'}} (${{ratio}} / required ${{required}})`);
        }}
      }}
      if (shadowSuggestion.enabled && shadowSuggestion.accepted) {{
        const ratio = shadowSuggestion.recommended_min_match_ratio == null
          ? 'n/a'
          : `${{(Number(shadowSuggestion.recommended_min_match_ratio) * 100).toFixed(1)}}%`;
        ops.push(`Suggested shadow settings: window_runs=${{shadowSuggestion.recommended_window_runs ?? 'n/a'}}, min_match_ratio=${{ratio}}`);
      }}
      document.getElementById('ops-internals').innerHTML = ops.map((line) => `<div>${{escapeHtml(line)}}</div>`).join('');

      const tbody = document.getElementById('profile-table');
      tbody.innerHTML = '';
      (data.profiles || []).forEach((item) => {{
        const tr = document.createElement('tr');
        const pct = item.gap_pct == null ? 'n/a' : `${{item.gap_pct.toFixed(2)}}%`;
        tr.innerHTML = `<td>${{escapeHtml(item.profile || '')}}</td><td>${{item.rank ?? 'n/a'}}</td><td>${{pct}}</td>`;
        tbody.appendChild(tr);
      }});

      const links = collectTickerLinks(symbol, data);
      const linkWrap = document.getElementById('report-links');
      linkWrap.innerHTML = '';
      if (!links.length) {{
        linkWrap.innerHTML = '<div class="row-item-note">No linked reports for this ticker.</div>';
      }}
      links.forEach((item) => {{
        const a = document.createElement('a');
        a.href = item.url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = item.label;
        linkWrap.appendChild(a);
      }});
      drawTickerChart();
      drawTickerRankChart();
    }}

    function drawTickerChart() {{
      const data = tickerData[currentTicker];
      if (!data) return;
      drawLineChart(
        'ticker-chart',
        [
          {{
            label: 'active edge',
            color: '#111111',
            points: (data.history || []).map((item) => item.active_gap),
          }},
          {{
            label: 'selected edge',
            color: '#94a3b8',
            points: (data.history || []).map((item) => item.selected_gap),
          }},
        ],
        {{ includeZero: true, emptyMessage: 'Need at least two daily points to draw this trend.' }}
      );
    }}

    function drawTickerRankChart() {{
      const data = tickerData[currentTicker];
      if (!data) return;
      drawLineChart(
        'ticker-rank-chart',
        [
          {{
            label: 'active rank',
            color: '#0f172a',
            points: (data.history || []).map((item) => item.active_rank),
          }},
        ],
        {{ invertY: true, emptyMessage: 'Need at least two daily points to draw this rank history.' }}
      );
    }}

    window.addEventListener('resize', () => {{
      if (document.getElementById('overview-panel').classList.contains('active')) renderOverview();
      if (document.getElementById('system-panel').classList.contains('active')) drawSystem();
      if (document.getElementById('ticker-panel').classList.contains('active')) {{
        drawTickerChart();
        drawTickerRankChart();
      }}
    }});

    const initialTicker = rankedTickers()[0] || tickerOrder[0] || null;
    currentTicker = initialTicker;
    showOverview();
  </script>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return output

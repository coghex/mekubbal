from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.reporting.html import render_html_table


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
    if {"run_timestamp_utc", "symbol"}.issubset(set(merged.columns)):
        merged = merged.drop_duplicates(subset=["run_timestamp_utc", "symbol"], keep="last").reset_index(
            drop=True
        )
    elif "run_timestamp_utc" in merged.columns:
        merged = merged.drop_duplicates(subset=["run_timestamp_utc"], keep="last").reset_index(drop=True)
    else:
        merged = merged.drop_duplicates(keep="last").reset_index(drop=True)
    merged.to_csv(history_path, index=False)
    return merged

def _html_table(title: str, note: str, frame: pd.DataFrame) -> str:
    return render_html_table(title, note, frame, escape=False, variant="compact")

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
    prod_symbols = set(production["symbol"].astype(str))
    shadow_symbols = set(shadow["symbol"].astype(str))
    overlap_ratio = len(prod_symbols & shadow_symbols) / max(len(prod_symbols | shadow_symbols), 1)
    if overlap_ratio < 0.8:
        warnings.warn(
            f"Shadow comparison covers only {overlap_ratio:.0%} of symbols "
            f"(production={len(prod_symbols)}, shadow={len(shadow_symbols)}, "
            f"overlap={len(prod_symbols & shadow_symbols)}).",
            stacklevel=2,
        )

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

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

def _load_shadow_suggestion_state(
    *,
    suggestion_state_path: Path,
    fallback_window_runs: int,
    fallback_min_match_ratio: float,
    auto_apply_enabled: bool,
) -> dict[str, Any]:
    effective_window_runs = int(fallback_window_runs)
    effective_min_match_ratio = float(fallback_min_match_ratio)
    loaded_payload: dict[str, Any] | None = None
    if auto_apply_enabled and suggestion_state_path.exists():
        payload = json.loads(suggestion_state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Shadow suggestion state must decode to a JSON object.")
        state_window = payload.get("active_window_runs")
        state_ratio = payload.get("active_min_match_ratio")
        if state_window is not None:
            effective_window_runs = int(state_window)
        if state_ratio is not None:
            effective_min_match_ratio = float(state_ratio)
        loaded_payload = payload
    return {
        "effective_window_runs": int(effective_window_runs),
        "effective_min_match_ratio": float(effective_min_match_ratio),
        "loaded_state": loaded_payload,
    }

def _append_shadow_suggestion_history_and_maybe_apply(
    *,
    run_timestamp_utc: str,
    suggestion_summary: dict[str, Any],
    suggestion_history_path: Path,
    suggestion_state_path: Path,
    stability_runs: int,
    auto_apply_enabled: bool,
    current_effective_window_runs: int,
    current_effective_min_match_ratio: float,
) -> dict[str, Any]:
    history_row = pd.DataFrame(
        [
            {
                "run_timestamp_utc": str(run_timestamp_utc),
                "accepted": bool(suggestion_summary.get("accepted", False)),
                "recommended_window_runs": suggestion_summary.get("recommended_window_runs"),
                "recommended_min_match_ratio": suggestion_summary.get("recommended_min_match_ratio"),
                "reasons": ";".join(str(value) for value in suggestion_summary.get("reasons", [])),
            }
        ]
    )
    history = _append_history_rows(history_row, suggestion_history_path)
    ordered = history.sort_values("run_timestamp_utc").reset_index(drop=True)
    tail = ordered.tail(int(stability_runs))
    tail_accepted = (
        tail["accepted"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    stable_ready = False
    target_window_runs = None
    target_min_match_ratio = None
    if len(tail) >= int(stability_runs) and not tail_accepted.isna().any() and bool(tail_accepted.all()):
        window_values = pd.to_numeric(tail["recommended_window_runs"], errors="coerce")
        ratio_values = pd.to_numeric(tail["recommended_min_match_ratio"], errors="coerce")
        if not window_values.isna().any() and not ratio_values.isna().any():
            stable_ready = bool(
                window_values.nunique(dropna=True) == 1
                and ratio_values.nunique(dropna=True) == 1
            )
            if stable_ready:
                target_window_runs = int(window_values.iloc[-1])
                target_min_match_ratio = float(ratio_values.iloc[-1])

    auto_apply_applied = False
    auto_apply_reason = "auto_apply_disabled"
    next_window_runs = int(current_effective_window_runs)
    next_min_match_ratio = float(current_effective_min_match_ratio)
    if auto_apply_enabled:
        auto_apply_reason = "stability_not_reached"
        if stable_ready and target_window_runs is not None and target_min_match_ratio is not None:
            next_window_runs = int(target_window_runs)
            next_min_match_ratio = float(target_min_match_ratio)
            auto_apply_reason = "already_active"
            if (
                int(current_effective_window_runs) != int(target_window_runs)
                or abs(float(current_effective_min_match_ratio) - float(target_min_match_ratio)) > 1e-12
            ):
                suggestion_state_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "updated_at_utc": str(run_timestamp_utc),
                    "active_window_runs": int(target_window_runs),
                    "active_min_match_ratio": float(target_min_match_ratio),
                    "stability_runs": int(stability_runs),
                    "source": "stable_shadow_suggestion",
                    "source_suggestion_json_path": str(suggestion_summary.get("suggestion_json_path", "")),
                }
                suggestion_state_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                auto_apply_applied = True
                auto_apply_reason = "applied_new_stable_recommendation"

    return {
        "suggestion_history_path": str(suggestion_history_path),
        "suggestion_history_rows": int(len(history)),
        "stability_runs": int(stability_runs),
        "stable_ready": bool(stable_ready),
        "auto_apply_enabled": bool(auto_apply_enabled),
        "auto_apply_applied": bool(auto_apply_applied),
        "auto_apply_reason": auto_apply_reason,
        "active_window_runs_for_next_run": int(next_window_runs),
        "active_min_match_ratio_for_next_run": float(next_min_match_ratio),
        "suggestion_state_path": str(suggestion_state_path),
    }

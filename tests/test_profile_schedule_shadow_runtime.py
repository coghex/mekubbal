from __future__ import annotations

import json

from mekubbal.profile.schedule_shadow_runtime import (
    resolve_shadow_runtime_settings,
    run_schedule_shadow,
)


def test_resolve_shadow_runtime_settings_uses_saved_effective_thresholds(tmp_path):
    suggestion_state_path = tmp_path / "reports" / "profile_shadow_suggestion_state.json"
    suggestion_state_path.parent.mkdir(parents=True, exist_ok=True)
    suggestion_state_path.write_text(
        json.dumps(
            {
                "active_window_runs": 7,
                "active_min_match_ratio": 0.88,
            }
        ),
        encoding="utf-8",
    )

    result = resolve_shadow_runtime_settings(
        shadow_enabled=True,
        shadow_cfg={
            "window_runs": 5,
            "min_match_ratio": 1.0,
            "suggestion_state_path": "reports/profile_shadow_suggestion_state.json",
            "suggestion_auto_apply_enabled": True,
        },
        matrix_output_root=tmp_path,
    )

    assert result["effective_shadow_window_runs"] == 7
    assert result["effective_shadow_min_match_ratio"] == 0.88
    assert result["shadow_suggestion_state_path"] == suggestion_state_path


def test_run_schedule_shadow_applies_promotion_when_gate_passes(monkeypatch, tmp_path):
    import mekubbal.profile.schedule_shadow_runtime as shadow_runtime

    production_state = tmp_path / "prod_selection_state.json"
    production_state.write_text(json.dumps({"active_profiles": {"AAPL": "base"}}), encoding="utf-8")
    shadow_state = tmp_path / "shadow_selection_state.json"
    shadow_state.write_text(
        json.dumps({"active_profiles": {"AAPL": "base"}, "shadow_marker": "candidate-ready"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        shadow_runtime,
        "_build_shadow_comparison",
        lambda **kwargs: {
            "comparison_history_path": str(tmp_path / "comparison_history.csv"),
            "comparison_html_path": str(tmp_path / "comparison.html"),
            "gate_json_path": str(tmp_path / "gate.json"),
            "overall_gate_passed": True,
        },
    )
    monkeypatch.setattr(
        shadow_runtime,
        "_suggest_shadow_thresholds",
        lambda **kwargs: {
            "suggestion_json_path": str(tmp_path / "suggestion.json"),
            "suggestion_html_path": str(tmp_path / "suggestion.html"),
            "suggestion_history_path": str(tmp_path / "suggestion_history.csv"),
            "suggestion_state_path": str(tmp_path / "suggestion_state.json"),
        },
    )
    monkeypatch.setattr(
        shadow_runtime,
        "_append_shadow_suggestion_history_and_maybe_apply",
        lambda **kwargs: {"accepted": False},
    )

    summary = run_schedule_shadow(
        matrix_summary={"symbol_summary_path": str(tmp_path / "profile_symbol_summary.csv")},
        monitor_summary={
            "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
            "health_snapshot_path": str(tmp_path / "prod_snapshot.csv"),
        },
        shadow_cfg={
            "health_snapshot_path": "shadow_snapshot.csv",
            "health_history_path": "shadow_history.csv",
            "drift_alerts_csv_path": "shadow_drift.csv",
            "drift_alerts_html_path": "shadow_drift.html",
            "drift_alerts_history_path": "shadow_drift_history.csv",
            "ensemble_alerts_csv_path": "shadow_ensemble.csv",
            "ensemble_alerts_html_path": "shadow_ensemble.html",
            "ensemble_alerts_history_path": "shadow_ensemble_history.csv",
            "ticker_summary_csv_path": "shadow_ticker.csv",
            "ticker_summary_html_path": "shadow_ticker.html",
            "ensemble_decision_csv_path": "shadow_decisions.csv",
            "ensemble_history_path": "shadow_ensemble_history_runtime.csv",
            "effective_selection_state_path": "shadow_effective_state.json",
            "comparison_csv_path": "comparison.csv",
            "comparison_history_path": "comparison_history.csv",
            "comparison_html_path": "comparison.html",
            "gate_json_path": "gate.json",
            "suggestion_json_path": "suggestion.json",
            "suggestion_html_path": "suggestion.html",
            "suggestion_min_history_runs": 3,
            "suggestion_history_path": "suggestion_history.csv",
            "suggestion_stability_runs": 2,
            "suggestion_auto_apply_enabled": False,
            "apply_promotion_after_shadow": True,
        },
        monitor_cfg={
            "lookback_runs": 1,
            "max_gap_drop": 0.01,
            "max_rank_worsening": 0.5,
            "min_active_minus_base_gap": -0.01,
            "ensemble_low_confidence_threshold": 0.55,
        },
        ensemble_cfg={"enabled": False},
        matrix_output_root=tmp_path,
        selection_state_path=production_state,
        shadow_selection_state_path=shadow_state,
        effective_shadow_window_runs=2,
        effective_shadow_min_match_ratio=1.0,
        shadow_suggestion_state_path=tmp_path / "suggestion_state.json",
        run_profile_monitor_fn=lambda **kwargs: {
            "health_snapshot_path": str(tmp_path / "shadow_snapshot.csv"),
            "run_timestamp_utc": kwargs["run_timestamp_utc"],
        },
    )

    loaded = json.loads(production_state.read_text(encoding="utf-8"))
    assert summary["promotion_applied"] is True
    assert loaded["shadow_marker"] == "candidate-ready"
    assert loaded["shadow_gate"]["window_runs"] == 2

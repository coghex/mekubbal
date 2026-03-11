from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mekubbal.profile_schedule import run_profile_schedule


def test_run_profile_schedule_runs_matrix_and_monitor(monkeypatch, tmp_path):
    import mekubbal.profile_schedule as schedule_module

    matrix_config = tmp_path / "profile-matrix.toml"
    schedule_config = tmp_path / "profile-schedule.toml"
    matrix_config.write_text("symbols = [\"AAPL\"]\n", encoding="utf-8")
    schedule_config.write_text(
        f"""
[schedule]
matrix_config = "{matrix_config}"
symbols = []
health_snapshot_path = "reports/active_profile_health.csv"
health_history_path = "reports/active_profile_health_history.csv"
drift_alerts_csv_path = "reports/profile_drift_alerts.csv"
drift_alerts_html_path = "reports/profile_drift_alerts.html"
drift_alerts_history_path = "reports/profile_drift_alerts_history.csv"
ticker_summary_csv_path = "reports/ticker_health_summary.csv"
ticker_summary_html_path = "reports/ticker_health_summary.html"
summary_json_path = "reports/profile_schedule_summary.json"

[monitor]
lookback_runs = 1
max_gap_drop = 0.01
max_rank_worsening = 0.5
min_active_minus_base_gap = -0.01
""".strip(),
        encoding="utf-8",
    )

    output_root = tmp_path / "matrix_out"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    symbol_summary = reports_root / "profile_symbol_summary.csv"
    selection_state = reports_root / "profile_selection_state.json"
    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 2, "avg_equity_gap": 0.01},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 1, "avg_equity_gap": 0.04},
        ]
    ).to_csv(symbol_summary, index=False)
    selection_state.write_text(
        json.dumps(
            {
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
                "active_profiles": {"AAPL": "candidate"},
            }
        ),
        encoding="utf-8",
    )

    def fake_run_profile_matrix(config_path, *, symbols_override):
        _ = config_path, symbols_override
        return {
            "output_root": str(output_root),
            "symbol_summary_path": str(symbol_summary),
            "profile_selection": {"state_path": str(selection_state)},
        }

    monkeypatch.setattr(schedule_module, "run_profile_matrix", fake_run_profile_matrix)
    summary = run_profile_schedule(schedule_config)
    assert Path(summary["summary_json_path"]).exists()
    assert Path(summary["monitor_summary"]["health_snapshot_path"]).exists()
    assert Path(summary["monitor_summary"]["drift_alerts_csv_path"]).exists()
    assert Path(summary["monitor_summary"]["drift_alerts_history_path"]).exists()
    assert Path(summary["monitor_summary"]["ticker_summary_csv_path"]).exists()
    assert Path(summary["monitor_summary"]["ticker_summary_html_path"]).exists()
    assert Path(summary["product_dashboard_path"]).exists()
    loaded = json.loads(Path(summary["summary_json_path"]).read_text(encoding="utf-8"))
    assert "matrix_summary" in loaded
    assert "monitor_summary" in loaded
    assert "product_dashboard_path" in loaded


def test_run_profile_schedule_runs_rollback_when_enabled(monkeypatch, tmp_path):
    import mekubbal.profile_schedule as schedule_module

    matrix_config = tmp_path / "profile-matrix.toml"
    schedule_config = tmp_path / "profile-schedule.toml"
    matrix_config.write_text("symbols = [\"AAPL\"]\n", encoding="utf-8")
    schedule_config.write_text(
        f"""
[schedule]
matrix_config = "{matrix_config}"
symbols = []
health_snapshot_path = "reports/active_profile_health.csv"
health_history_path = "reports/active_profile_health_history.csv"
drift_alerts_csv_path = "reports/profile_drift_alerts.csv"
drift_alerts_html_path = "reports/profile_drift_alerts.html"
drift_alerts_history_path = "reports/profile_drift_alerts_history.csv"
ticker_summary_csv_path = "reports/ticker_health_summary.csv"
ticker_summary_html_path = "reports/ticker_health_summary.html"
summary_json_path = "reports/profile_schedule_summary.json"

[monitor]
lookback_runs = 1
max_gap_drop = 0.01
max_rank_worsening = 0.5
min_active_minus_base_gap = -0.01
ensemble_low_confidence_threshold = 0.55

[rollback]
enabled = true
rollback_state_path = "reports/profile_rollback_state.json"
min_consecutive_alert_runs = 2
rollback_profile = "base"
apply_rollback = false
""".strip(),
        encoding="utf-8",
    )

    output_root = tmp_path / "matrix_out"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    symbol_summary = reports_root / "profile_symbol_summary.csv"
    selection_state = reports_root / "profile_selection_state.json"
    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.03},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": 0.01},
        ]
    ).to_csv(symbol_summary, index=False)
    selection_state.write_text(
        json.dumps(
            {
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
                "active_profiles": {"AAPL": "candidate"},
            }
        ),
        encoding="utf-8",
    )

    def fake_run_profile_matrix(config_path, *, symbols_override):
        _ = config_path, symbols_override
        return {
            "output_root": str(output_root),
            "symbol_summary_path": str(symbol_summary),
            "profile_selection": {"state_path": str(selection_state)},
        }

    rollback_calls: list[dict[str, object]] = []

    def fake_run_profile_rollback(**kwargs):
        rollback_calls.append(kwargs)
        rollback_path = output_root / "reports" / "profile_rollback_state.json"
        rollback_path.write_text("{}", encoding="utf-8")
        return {
            "rollback_state_path": str(rollback_path),
            "symbols_evaluated": 1,
            "rollback_recommended_count": 0,
            "rollback_applied_count": 0,
            "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(schedule_module, "run_profile_matrix", fake_run_profile_matrix)
    monkeypatch.setattr(schedule_module, "run_profile_rollback", fake_run_profile_rollback)
    summary = run_profile_schedule(schedule_config)
    assert summary["rollback_summary"] is not None
    assert len(rollback_calls) == 1


def test_run_profile_schedule_uses_ensemble_state_for_non_mutating_rollback(monkeypatch, tmp_path):
    import mekubbal.profile_schedule as schedule_module

    matrix_config = tmp_path / "profile-matrix.toml"
    schedule_config = tmp_path / "profile-schedule.toml"
    matrix_config.write_text("symbols = [\"AAPL\"]\n", encoding="utf-8")
    schedule_config.write_text(
        f"""
[schedule]
matrix_config = "{matrix_config}"
symbols = []
health_snapshot_path = "reports/active_profile_health.csv"
health_history_path = "reports/active_profile_health_history.csv"
drift_alerts_csv_path = "reports/profile_drift_alerts.csv"
drift_alerts_html_path = "reports/profile_drift_alerts.html"
drift_alerts_history_path = "reports/profile_drift_alerts_history.csv"
ticker_summary_csv_path = "reports/ticker_health_summary.csv"
ticker_summary_html_path = "reports/ticker_health_summary.html"
summary_json_path = "reports/profile_schedule_summary.json"

[monitor]
lookback_runs = 1
max_gap_drop = 0.01
max_rank_worsening = 0.5
min_active_minus_base_gap = -0.01

[rollback]
enabled = true
rollback_state_path = "reports/profile_rollback_state.json"
min_consecutive_alert_runs = 2
rollback_on_drift_alerts = true
rollback_on_ensemble_events = true
min_consecutive_ensemble_event_runs = 2
rollback_profile = "base"
apply_rollback = false

[ensemble_v3]
enabled = true
lookback_runs = 1
min_regime_confidence = 0.5
rank_weight = 0.5
gap_weight = 0.5
significance_bonus = 0.0
fallback_profile = "base"
high_vol_gap_std_threshold = 0.03
high_vol_rank_std_threshold = 0.75
trending_min_gap_improvement = 0.01
trending_min_rank_improvement = 0.25
decision_csv_path = "reports/profile_ensemble_decisions.csv"
decision_history_path = "reports/profile_ensemble_history.csv"
effective_selection_state_path = "reports/profile_selection_state_ensemble.json"
""".strip(),
        encoding="utf-8",
    )

    output_root = tmp_path / "matrix_out"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    symbol_summary = reports_root / "profile_symbol_summary.csv"
    selection_state = reports_root / "profile_selection_state.json"
    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.03},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": 0.01},
        ]
    ).to_csv(symbol_summary, index=False)
    selection_state.write_text(
        json.dumps(
            {
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
                "active_profiles": {"AAPL": "base"},
            }
        ),
        encoding="utf-8",
    )

    def fake_run_profile_matrix(config_path, *, symbols_override):
        _ = config_path, symbols_override
        return {
            "output_root": str(output_root),
            "symbol_summary_path": str(symbol_summary),
            "profile_selection": {"state_path": str(selection_state)},
        }

    def fake_run_profile_monitor(**kwargs):
        _ = kwargs
        history = reports_root / "active_profile_health_history.csv"
        pd.DataFrame(
            [
                {
                    "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                    "symbol": "AAPL",
                    "active_profile": "base",
                    "selected_profile": "base",
                    "active_rank": 1,
                    "active_gap": 0.03,
                    "selected_gap": 0.03,
                }
            ]
        ).to_csv(history, index=False)
        ticker_summary = reports_root / "ticker_health_summary.csv"
        pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "status": "Healthy",
                    "selected_profile": "base",
                    "active_profile": "base",
                    "active_profile_source": "selection_state",
                    "active_rank": 1,
                    "active_vs_buy_and_hold": "+3.00%",
                    "active_vs_base": "+0.00%",
                    "recommended_action": "Keep current active profile.",
                    "summary": "base vs buy-and-hold +3.00%; vs base +0.00%.",
                }
            ]
        ).to_csv(ticker_summary, index=False)
        ticker_summary_html = reports_root / "ticker_health_summary.html"
        ticker_summary_html.write_text("<html>ticker summary</html>", encoding="utf-8")
        drift_html = reports_root / "profile_drift_alerts.html"
        drift_html.write_text("<html>drift</html>", encoding="utf-8")
        ensemble_html = reports_root / "profile_ensemble_alerts.html"
        ensemble_html.write_text("<html>ensemble</html>", encoding="utf-8")
        ensemble_state = reports_root / "profile_selection_state_ensemble.json"
        ensemble_state.write_text("{}", encoding="utf-8")
        return {
            "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
            "health_history_path": str(history),
            "ticker_summary_csv_path": str(ticker_summary),
            "ticker_summary_html_path": str(ticker_summary_html),
            "drift_alerts_html_path": str(drift_html),
            "ensemble_alerts_html_path": str(ensemble_html),
            "ensemble_alerts_history_path": str(reports_root / "profile_ensemble_alerts_history.csv"),
            "ensemble_effective_selection_state_path": str(ensemble_state),
        }

    rollback_calls: list[dict[str, object]] = []

    def fake_run_profile_rollback(**kwargs):
        rollback_calls.append(kwargs)
        rollback_path = reports_root / "profile_rollback_state.json"
        rollback_path.write_text("{}", encoding="utf-8")
        return {
            "rollback_state_path": str(rollback_path),
            "symbols_evaluated": 1,
            "rollback_recommended_count": 0,
            "rollback_applied_count": 0,
            "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(schedule_module, "run_profile_matrix", fake_run_profile_matrix)
    monkeypatch.setattr(schedule_module, "run_profile_monitor", fake_run_profile_monitor)
    monkeypatch.setattr(schedule_module, "run_profile_rollback", fake_run_profile_rollback)

    run_profile_schedule(schedule_config)
    assert len(rollback_calls) == 1
    assert str(rollback_calls[0]["selection_state_path"]).endswith(
        "profile_selection_state_ensemble.json"
    )


def test_run_profile_schedule_shadow_gate_applies_promotions(monkeypatch, tmp_path):
    import mekubbal.profile_schedule as schedule_module

    matrix_config = tmp_path / "profile-matrix.toml"
    schedule_config = tmp_path / "profile-schedule.toml"
    matrix_config.write_text("symbols = [\"AAPL\"]\n", encoding="utf-8")
    schedule_config.write_text(
        f"""
[schedule]
matrix_config = "{matrix_config}"
symbols = []
health_snapshot_path = "reports/active_profile_health.csv"
health_history_path = "reports/active_profile_health_history.csv"
drift_alerts_csv_path = "reports/profile_drift_alerts.csv"
drift_alerts_html_path = "reports/profile_drift_alerts.html"
drift_alerts_history_path = "reports/profile_drift_alerts_history.csv"
ensemble_alerts_csv_path = "reports/profile_ensemble_alerts.csv"
ensemble_alerts_html_path = "reports/profile_ensemble_alerts.html"
ensemble_alerts_history_path = "reports/profile_ensemble_alerts_history.csv"
ticker_summary_csv_path = "reports/ticker_health_summary.csv"
ticker_summary_html_path = "reports/ticker_health_summary.html"
summary_json_path = "reports/profile_schedule_summary.json"

[monitor]
lookback_runs = 1
max_gap_drop = 0.01
max_rank_worsening = 0.5
min_active_minus_base_gap = -0.01
ensemble_low_confidence_threshold = 0.55

[shadow]
enabled = true
production_state_path = "reports/prod_selection_state.json"
shadow_state_path = "reports/shadow_selection_state.json"
window_runs = 2
min_match_ratio = 1.0
apply_promotion_after_shadow = true
comparison_csv_path = "reports/profile_shadow_comparison.csv"
comparison_html_path = "reports/profile_shadow_comparison.html"
comparison_history_path = "reports/profile_shadow_comparison_history.csv"
gate_json_path = "reports/profile_shadow_gate.json"
health_snapshot_path = "reports/shadow_active_profile_health.csv"
health_history_path = "reports/shadow_active_profile_health_history.csv"
drift_alerts_csv_path = "reports/shadow_profile_drift_alerts.csv"
drift_alerts_html_path = "reports/shadow_profile_drift_alerts.html"
drift_alerts_history_path = "reports/shadow_profile_drift_alerts_history.csv"
ensemble_alerts_csv_path = "reports/shadow_profile_ensemble_alerts.csv"
ensemble_alerts_html_path = "reports/shadow_profile_ensemble_alerts.html"
ensemble_alerts_history_path = "reports/shadow_profile_ensemble_alerts_history.csv"
ticker_summary_csv_path = "reports/shadow_ticker_health_summary.csv"
ticker_summary_html_path = "reports/shadow_ticker_health_summary.html"
ensemble_decision_csv_path = "reports/shadow_profile_ensemble_decisions.csv"
ensemble_history_path = "reports/shadow_profile_ensemble_history.csv"
effective_selection_state_path = "reports/profile_selection_state_shadow_ensemble.json"
""".strip(),
        encoding="utf-8",
    )

    output_root = tmp_path / "matrix_out"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    symbol_summary = reports_root / "profile_symbol_summary.csv"
    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.03},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": 0.02},
        ]
    ).to_csv(symbol_summary, index=False)

    production_state = reports_root / "prod_selection_state.json"
    production_state.write_text(
        json.dumps(
            {
                "active_profiles": {"AAPL": "base"},
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
            }
        ),
        encoding="utf-8",
    )
    shadow_state = reports_root / "shadow_selection_state.json"
    shadow_state.write_text(
        json.dumps(
            {
                "active_profiles": {"AAPL": "base"},
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
                "shadow_marker": "candidate-ready",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "production_selected_profile": "base",
                "production_active_profile": "base",
                "production_active_profile_source": "selection_state",
                "production_active_rank": 1,
                "production_active_gap": 0.03,
                "shadow_selected_profile": "base",
                "shadow_active_profile": "base",
                "shadow_active_profile_source": "selection_state",
                "shadow_active_rank": 1,
                "shadow_active_gap": 0.03,
                "active_profile_match": True,
                "active_rank_delta": 0.0,
                "active_gap_delta": 0.0,
            }
        ]
    ).to_csv(reports_root / "profile_shadow_comparison_history.csv", index=False)

    monkeypatch.setattr(
        schedule_module,
        "load_profile_matrix_config",
        lambda _path: {
            "matrix": {"output_root": str(output_root)},
            "promotion": {"state_path": "reports/profile_selection_state.json"},
        },
    )

    matrix_calls: list[dict[str, object]] = []

    def fake_run_profile_matrix(config_path, *, symbols_override, promotion_override=None):
        _ = config_path, symbols_override
        matrix_calls.append({"promotion_override": promotion_override})
        return {
            "output_root": str(output_root),
            "symbol_summary_path": str(symbol_summary),
            "profile_selection": {"state_path": str(shadow_state)},
            "dashboard_path": str(reports_root / "profile_matrix_workspace.html"),
            "profile_aggregate_html_path": str(reports_root / "profile_aggregate_leaderboard.html"),
            "profile_pairwise_html_path": str(reports_root / "profile_pairwise_across_symbols.html"),
        }

    monitor_calls: list[str] = []

    def fake_run_profile_monitor(**kwargs):
        state_path = str(kwargs["selection_state_path"])
        monitor_calls.append(state_path)
        snapshot_path = Path(str(kwargs["health_snapshot_path"]))
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        summary_profile = "base"
        pd.DataFrame(
            [
                {
                    "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                    "symbol": "AAPL",
                    "selected_profile": summary_profile,
                    "active_profile": summary_profile,
                    "active_profile_source": "selection_state",
                    "active_rank": 1,
                    "active_gap": 0.03,
                }
            ]
        ).to_csv(snapshot_path, index=False)
        history_path = Path(str(kwargs["health_history_path"]))
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                    "symbol": "AAPL",
                    "active_profile": summary_profile,
                    "active_rank": 1,
                    "active_gap": 0.03,
                    "selected_profile": summary_profile,
                    "selected_gap": 0.03,
                    "active_minus_base_gap": 0.0,
                }
            ]
        ).to_csv(history_path, index=False)
        ticker_summary_csv = Path(str(kwargs["ticker_summary_csv_path"]))
        ticker_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "status": "Healthy",
                    "selected_profile": summary_profile,
                    "active_profile": summary_profile,
                    "active_profile_source": "selection_state",
                    "recommended_action": "Keep current active profile.",
                    "summary": "all clear",
                }
            ]
        ).to_csv(ticker_summary_csv, index=False)
        ticker_summary_html = Path(str(kwargs["ticker_summary_html_path"]))
        ticker_summary_html.write_text("<html>ticker</html>", encoding="utf-8")
        drift_html = Path(str(kwargs["drift_alerts_html_path"]))
        drift_html.write_text("<html>drift</html>", encoding="utf-8")
        drift_csv = Path(str(kwargs["drift_alerts_csv_path"]))
        pd.DataFrame(columns=["symbol", "reasons"]).to_csv(drift_csv, index=False)
        drift_history = kwargs.get("drift_alerts_history_path")
        if drift_history:
            Path(str(drift_history)).write_text("symbol,reasons\n", encoding="utf-8")
        ensemble_html = kwargs.get("ensemble_alerts_html_path")
        if ensemble_html:
            Path(str(ensemble_html)).write_text("<html>ensemble</html>", encoding="utf-8")
        ensemble_csv = kwargs.get("ensemble_alerts_csv_path")
        if ensemble_csv:
            pd.DataFrame(columns=["symbol", "reasons"]).to_csv(Path(str(ensemble_csv)), index=False)
        ensemble_hist = kwargs.get("ensemble_alerts_history_path")
        if ensemble_hist:
            Path(str(ensemble_hist)).write_text("symbol,reasons\n", encoding="utf-8")
        return {
            "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
            "health_snapshot_path": str(snapshot_path),
            "health_history_path": str(history_path),
            "drift_alerts_csv_path": str(drift_csv),
            "drift_alerts_html_path": str(drift_html),
            "drift_alerts_history_path": str(drift_history) if drift_history else None,
            "ticker_summary_csv_path": str(ticker_summary_csv),
            "ticker_summary_html_path": str(ticker_summary_html),
            "ensemble_alerts_csv_path": str(ensemble_csv) if ensemble_csv else None,
            "ensemble_alerts_html_path": str(ensemble_html) if ensemble_html else None,
            "ensemble_alerts_history_path": str(ensemble_hist) if ensemble_hist else None,
            "ensemble_alerts_count": 0,
            "ensemble_alerts_history_count": 0,
            "history_rows": 1,
            "alerts_count": 0,
            "alerts_history_count": 0,
            "ticker_status_counts": {"Healthy": 1},
            "ensemble_v3_summary": None,
            "ensemble_effective_selection_state_path": None,
        }

    monkeypatch.setattr(schedule_module, "run_profile_matrix", fake_run_profile_matrix)
    monkeypatch.setattr(schedule_module, "run_profile_monitor", fake_run_profile_monitor)

    summary = run_profile_schedule(schedule_config)
    assert len(matrix_calls) == 1
    assert isinstance(matrix_calls[0]["promotion_override"], dict)
    assert str(matrix_calls[0]["promotion_override"]["state_path"]).endswith(
        "reports/shadow_selection_state.json"
    )
    assert len(monitor_calls) == 2
    assert monitor_calls[0].endswith("reports/prod_selection_state.json")
    assert monitor_calls[1].endswith("reports/shadow_selection_state.json")
    assert summary["shadow_summary"] is not None
    assert summary["shadow_summary"]["comparison_summary"]["overall_gate_passed"] is True
    assert summary["shadow_summary"]["promotion_applied"] is True
    production_loaded = json.loads(production_state.read_text(encoding="utf-8"))
    assert production_loaded["shadow_marker"] == "candidate-ready"
    assert "shadow_gate" in production_loaded

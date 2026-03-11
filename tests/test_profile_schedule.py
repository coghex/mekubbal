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

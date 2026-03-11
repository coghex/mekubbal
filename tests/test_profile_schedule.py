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
    loaded = json.loads(Path(summary["summary_json_path"]).read_text(encoding="utf-8"))
    assert "matrix_summary" in loaded
    assert "monitor_summary" in loaded


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

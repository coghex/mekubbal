from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mekubbal.profile_backfill import run_profile_backfill


def _write_ohlcv(path: Path, dates: list[str]) -> None:
    rows = []
    for index, date in enumerate(dates, start=1):
        price = 100 + index
        rows.append(
            {
                "date": date,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1000 + index,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_control_config(path: Path, *, train_window: int, test_window: int) -> None:
    path.write_text(
        f"""
[data]
path = "unused.csv"
refresh = false
symbol = "AAPL"
start = ""
end = ""

[policy]
timesteps = 1
trade_cost = 0.001
risk_penalty = 0.0
switch_penalty = 0.0
position_levels = [-1.0, 0.0, 1.0]
seed = 7
downside_window = 2

[walkforward]
enabled = true
models_dir = "models/walkforward"
report_path = "logs/walkforward.csv"
train_window = {train_window}
test_window = {test_window}
step_window = {test_window}
expanding = false

[ablation]
enabled = false
models_dir = "models/ablation"
report_path = "logs/ablation.csv"
summary_path = "logs/ablation_summary.csv"
v2_downside_penalty = 0.01
v2_drawdown_penalty = 0.05

[sweep]
enabled = false
output_dir = "logs/sweeps"
report_path = "logs/sweeps/ranking.csv"
downside_grid = [0.0]
drawdown_grid = [0.0]
regime_tie_break_tolerance = 0.0

[selection]
enabled = false
report_path = "logs/walkforward.csv"
state_path = "models/current_model.json"
lookback = 1
min_gap = 0.0
allow_average_rule = false

[visualization]
enabled = false
output_path = "logs/report.html"
title = "Report"

[logging]
enabled = false
db_path = "logs/experiments.db"
symbol = "AAPL"
""".strip(),
        encoding="utf-8",
    )


def _write_runner_config(path: Path, *, base_name: str = "base", candidate_name: str = "candidate") -> None:
    path.write_text(
        f"""
[runner]
output_root = "logs/profile_runner"
profile_summary_path = "reports/profile_summary.csv"
pairwise_csv_path = "reports/pairwise.csv"
pairwise_html_path = "reports/pairwise.html"
dashboard_path = "reports/dashboard.html"
dashboard_title = "Runner"
build_dashboard = false

[data]
path = "data/aapl.csv"
refresh = false
symbol = "AAPL"
start = ""
end = ""

[comparison]
confidence_level = 0.95
bootstrap_samples = 100
permutation_samples = 100
seed = 7
title = "Pairwise"

[[profiles]]
name = "{base_name}"
config = "control-base.toml"

[[profiles]]
name = "{candidate_name}"
config = "control-candidate.toml"
""".strip(),
        encoding="utf-8",
    )


def test_run_profile_backfill_resets_output_and_replays_recent_common_dates(monkeypatch, tmp_path):
    import mekubbal.profile_backfill as backfill_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ohlcv(data_dir / "aapl.csv", [f"2026-01-0{day}" for day in range(1, 7)])
    _write_ohlcv(data_dir / "msft.csv", [f"2026-01-0{day}" for day in range(2, 8)])

    _write_control_config(tmp_path / "control-base.toml", train_window=2, test_window=1)
    _write_control_config(tmp_path / "control-candidate.toml", train_window=2, test_window=1)
    _write_runner_config(tmp_path / "profile-runner.toml")

    output_root = tmp_path / "logs" / "profile_matrix_daily"
    output_root.mkdir(parents=True)
    (output_root / "stale.txt").write_text("old", encoding="utf-8")

    matrix_config = tmp_path / "profile-matrix.toml"
    matrix_config.write_text(
        f"""
symbols = ["AAPL", "MSFT"]

[matrix]
output_root = "{output_root}"
build_dashboard = false

[base_runner]
config = "{tmp_path / 'profile-runner.toml'}"
output_root_template = "symbols/{{symbol_lower}}"
data_path_template = "{data_dir / '{symbol_lower}.csv'}"
refresh = false
start = ""
end = ""
build_symbol_dashboards = false

[promotion]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    schedule_config = tmp_path / "profile-schedule.toml"
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
product_dashboard_path = "reports/product_dashboard.html"
summary_json_path = "reports/profile_schedule_summary.json"

[monitor]
lookback_runs = 1
max_gap_drop = 0.01
max_rank_worsening = 0.5
min_active_minus_base_gap = -0.01
ensemble_low_confidence_threshold = 0.55
""".strip(),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_run_profile_schedule_config(
        config,
        *,
        config_dir,
        config_label,
        run_timestamp_utc,
        matrix_config_override,
        matrix_config_dir,
        matrix_config_label,
    ):
        _ = config, config_dir, config_label, matrix_config_dir, matrix_config_label
        calls.append(
            {
                "run_timestamp_utc": run_timestamp_utc,
                "data_template": matrix_config_override["base_runner"]["data_path_template"],
            }
        )
        return {
            "monitor_summary": {"run_timestamp_utc": run_timestamp_utc},
            "summary_json_path": str(output_root / "reports" / "profile_schedule_summary.json"),
            "product_dashboard_path": str(output_root / "reports" / "product_dashboard.html"),
        }

    monkeypatch.setattr(backfill_module, "run_profile_schedule_config", fake_run_profile_schedule_config)
    summary = run_profile_backfill(schedule_config, reset_output=True, max_runs=2)

    assert not (output_root / "stale.txt").exists()
    assert [call["run_timestamp_utc"] for call in calls] == [
        "2026-01-05T00:00:00+00:00",
        "2026-01-06T00:00:00+00:00",
    ]
    assert calls[0]["data_template"].endswith("{symbol_lower}.csv")
    assert summary["runs_replayed"] == 2
    assert summary["first_replay_date"] == "2026-01-05"
    assert summary["last_replay_date"] == "2026-01-06"
    loaded = json.loads(Path(summary["summary_json_path"]).read_text(encoding="utf-8"))
    assert loaded["symbols"] == ["AAPL", "MSFT"]


def test_run_profile_backfill_respects_schedule_symbol_override_and_stride(monkeypatch, tmp_path):
    import mekubbal.profile_backfill as backfill_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ohlcv(data_dir / "aapl.csv", [f"2026-01-0{day}" for day in range(1, 7)])
    _write_ohlcv(data_dir / "msft.csv", [f"2026-01-0{day}" for day in range(3, 8)])

    _write_control_config(tmp_path / "control-base.toml", train_window=2, test_window=1)
    _write_control_config(tmp_path / "control-candidate.toml", train_window=2, test_window=1)
    _write_runner_config(tmp_path / "profile-runner.toml")

    output_root = tmp_path / "logs" / "profile_matrix_daily"
    matrix_config = tmp_path / "profile-matrix.toml"
    matrix_config.write_text(
        f"""
symbols = ["AAPL", "MSFT"]

[matrix]
output_root = "{output_root}"
build_dashboard = false

[base_runner]
config = "{tmp_path / 'profile-runner.toml'}"
output_root_template = "symbols/{{symbol_lower}}"
data_path_template = "{data_dir / '{symbol_lower}.csv'}"
refresh = false
start = ""
end = ""
build_symbol_dashboards = false

[promotion]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    schedule_config = tmp_path / "profile-schedule.toml"
    schedule_config.write_text(
        f"""
[schedule]
matrix_config = "{matrix_config}"
symbols = ["AAPL"]
health_snapshot_path = "reports/active_profile_health.csv"
health_history_path = "reports/active_profile_health_history.csv"
drift_alerts_csv_path = "reports/profile_drift_alerts.csv"
drift_alerts_html_path = "reports/profile_drift_alerts.html"
drift_alerts_history_path = "reports/profile_drift_alerts_history.csv"
ticker_summary_csv_path = "reports/ticker_health_summary.csv"
ticker_summary_html_path = "reports/ticker_health_summary.html"
product_dashboard_path = "reports/product_dashboard.html"
summary_json_path = "reports/profile_schedule_summary.json"

[monitor]
lookback_runs = 1
max_gap_drop = 0.01
max_rank_worsening = 0.5
min_active_minus_base_gap = -0.01
ensemble_low_confidence_threshold = 0.55
""".strip(),
        encoding="utf-8",
    )

    replayed: list[str] = []

    def fake_run_profile_schedule_config(*args, **kwargs):
        replayed.append(kwargs["run_timestamp_utc"])
        return {
            "monitor_summary": {"run_timestamp_utc": kwargs["run_timestamp_utc"]},
            "summary_json_path": str(output_root / "reports" / "profile_schedule_summary.json"),
            "product_dashboard_path": str(output_root / "reports" / "product_dashboard.html"),
        }

    monkeypatch.setattr(backfill_module, "run_profile_schedule_config", fake_run_profile_schedule_config)
    summary = run_profile_backfill(schedule_config, every=2)

    assert replayed == [
        "2026-01-03T00:00:00+00:00",
        "2026-01-05T00:00:00+00:00",
    ]
    assert summary["symbols"] == ["AAPL"]


def test_run_profile_backfill_skips_symbols_without_enough_total_history(monkeypatch, tmp_path):
    import mekubbal.profile_backfill as backfill_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_ohlcv(data_dir / "aapl.csv", [f"2026-01-0{day}" for day in range(1, 7)])
    _write_ohlcv(data_dir / "rddt.csv", [f"2026-01-0{day}" for day in range(1, 3)])

    _write_control_config(tmp_path / "control-base.toml", train_window=2, test_window=1)
    _write_control_config(tmp_path / "control-candidate.toml", train_window=2, test_window=1)
    _write_runner_config(tmp_path / "profile-runner.toml")

    output_root = tmp_path / "logs" / "profile_matrix_daily"
    matrix_config = tmp_path / "profile-matrix.toml"
    matrix_config.write_text(
        f"""
symbols = ["AAPL", "RDDT"]

[matrix]
output_root = "{output_root}"
build_dashboard = false

[base_runner]
config = "{tmp_path / 'profile-runner.toml'}"
output_root_template = "symbols/{{symbol_lower}}"
data_path_template = "{data_dir / '{symbol_lower}.csv'}"
refresh = false
start = ""
end = ""
build_symbol_dashboards = false

[promotion]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    schedule_config = tmp_path / "profile-schedule.toml"
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
product_dashboard_path = "reports/product_dashboard.html"
summary_json_path = "reports/profile_schedule_summary.json"

[monitor]
lookback_runs = 1
max_gap_drop = 0.01
max_rank_worsening = 0.5
min_active_minus_base_gap = -0.01
ensemble_low_confidence_threshold = 0.55
""".strip(),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_run_profile_schedule_config(
        config,
        *,
        config_dir,
        config_label,
        run_timestamp_utc,
        matrix_config_override,
        matrix_config_dir,
        matrix_config_label,
    ):
        _ = config, config_dir, config_label, matrix_config_dir, matrix_config_label
        calls.append(
            {
                "run_timestamp_utc": run_timestamp_utc,
                "symbols": list(matrix_config_override["symbols"]),
            }
        )
        return {
            "monitor_summary": {"run_timestamp_utc": run_timestamp_utc},
            "summary_json_path": str(output_root / "reports" / "profile_schedule_summary.json"),
            "product_dashboard_path": str(output_root / "reports" / "product_dashboard.html"),
        }

    monkeypatch.setattr(backfill_module, "run_profile_schedule_config", fake_run_profile_schedule_config)
    summary = run_profile_backfill(schedule_config, max_runs=1)

    assert calls == [
        {
            "run_timestamp_utc": "2026-01-06T00:00:00+00:00",
            "symbols": ["AAPL"],
        }
    ]
    assert summary["requested_symbols"] == ["AAPL", "RDDT"]
    assert summary["symbols"] == ["AAPL"]
    assert summary["skipped_symbols"] == [
        {
            "symbol": "RDDT",
            "available_rows": 2,
            "required_rows": 3,
            "reason": "RDDT only has 2 rows, but backfill requires at least 3 rows per ticker.",
        }
    ]

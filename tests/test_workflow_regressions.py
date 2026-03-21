from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mekubbal.control import run_research_control
from mekubbal.multi_symbol import run_multi_symbol_control
from mekubbal.profile_runner import run_profile_runner
from mekubbal.profile_schedule import run_profile_schedule


def test_control_workflow_refreshes_data_and_skips_disabled_sections(monkeypatch, tmp_path):
    import mekubbal.control as control_module

    config_path = tmp_path / "control.toml"
    config_path.write_text(
        f"""
[data]
path = "{tmp_path / "data.csv"}"
refresh = true
symbol = "AAPL"
start = "2024-01-01"
end = "2024-12-31"

[policy]
timesteps = 100
trade_cost = 0.001
risk_penalty = 0.0002
switch_penalty = 0.0001
position_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
seed = 7
downside_window = 20

[walkforward]
enabled = true
models_dir = "{tmp_path / "models" / "wf"}"
report_path = "{tmp_path / "logs" / "walkforward.csv"}"
train_window = 120
test_window = 40
step_window = 40
expanding = false

[ablation]
enabled = false

[sweep]
enabled = false

[selection]
enabled = false
state_path = "{tmp_path / "models" / "current_model.json"}"

[visualization]
enabled = false
output_path = "{tmp_path / "logs" / "report.html"}"

[logging]
enabled = false
db_path = "{tmp_path / "logs" / "experiments.db"}"
symbol = "AAPL"
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_download_ohlcv(*, symbol, start, end):
        captured["download"] = {"symbol": symbol, "start": start, "end": end}
        return pd.DataFrame(
            [
                {"date": "2024-01-01", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
                {"date": "2024-01-02", "open": 1.05, "high": 1.15, "low": 1.0, "close": 1.1, "volume": 1200},
            ]
        )

    def fake_save_ohlcv_csv(frame, path):
        captured["save"] = {"rows": len(frame), "path": str(path)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def fake_set_global_seed(seed):
        captured["seed"] = seed

    def fake_walkforward(**kwargs):
        captured["walkforward"] = kwargs
        return {"report_path": kwargs["report_path"], "folds": 2}

    monkeypatch.setattr(control_module, "download_ohlcv", fake_download_ohlcv)
    monkeypatch.setattr(control_module, "save_ohlcv_csv", fake_save_ohlcv_csv)
    monkeypatch.setattr(control_module, "set_global_seed", fake_set_global_seed)
    monkeypatch.setattr(control_module, "run_walk_forward_validation", fake_walkforward)

    summary = run_research_control(config_path)

    assert captured["download"] == {
        "symbol": "AAPL",
        "start": "2024-01-01",
        "end": "2024-12-31",
    }
    assert captured["save"]["rows"] == 2
    assert captured["seed"] == 7
    assert captured["walkforward"]["log_db_path"] is None
    assert summary["walkforward"]["folds"] == 2
    assert "ablation" not in summary
    assert "sweep" not in summary
    assert "selection" not in summary
    assert "experiment_run_id" not in summary
    assert "visual_report_path" not in summary


def test_multi_symbol_workflow_combines_refresh_hardening_and_dashboard(monkeypatch, tmp_path):
    import mekubbal.multi_symbol as ms_module

    base_config = tmp_path / "control.toml"
    base_config.write_text(
        f"""
[data]
path = "{tmp_path / "data.csv"}"
refresh = false

[policy]
timesteps = 100
trade_cost = 0.001
risk_penalty = 0.0002
switch_penalty = 0.0001
position_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
seed = 7
downside_window = 20

[walkforward]
enabled = true
models_dir = "{tmp_path / "models" / "wf"}"
report_path = "{tmp_path / "logs" / "walkforward.csv"}"
train_window = 120
test_window = 40
step_window = 40
expanding = false

[ablation]
enabled = true
models_dir = "{tmp_path / "models" / "ablation"}"
report_path = "{tmp_path / "logs" / "ablation_folds.csv"}"
summary_path = "{tmp_path / "logs" / "ablation_summary.csv"}"
v2_downside_penalty = 0.01
v2_drawdown_penalty = 0.05

[sweep]
enabled = true
output_dir = "{tmp_path / "logs" / "sweep"}"
report_path = "{tmp_path / "logs" / "sweep" / "ranking.csv"}"
downside_grid = [0.0, 0.01]
drawdown_grid = [0.0, 0.05]

[selection]
enabled = true
report_path = "{tmp_path / "logs" / "walkforward.csv"}"
state_path = "{tmp_path / "models" / "current_model.json"}"
lookback = 3
min_gap = 0.0
allow_average_rule = false
min_turbulent_steps = 10
min_turbulent_win_rate = 0.2
min_turbulent_equity_factor = 0.5
max_turbulent_drawdown = 1.0

[visualization]
enabled = true
output_path = "{tmp_path / "logs" / "report.html"}"
title = "Control report"

[logging]
enabled = false
db_path = "{tmp_path / "logs" / "experiments.db"}"
symbol = "AAPL"
""".strip(),
        encoding="utf-8",
    )

    control_calls: list[dict[str, object]] = []
    harden_calls: list[dict[str, object]] = []

    def fake_run_research_control_config(config: dict, *, config_label: str):
        control_calls.append(
            {
                "label": config_label,
                "symbol": config["logging"]["symbol"],
                "data": dict(config["data"]),
            }
        )
        report_path = Path(config["visualization"]["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("<html></html>", encoding="utf-8")
        return {
            "data_path": config["data"]["path"],
            "visual_report_path": str(report_path),
            "walkforward": {
                "avg_policy_final_equity": 1.05,
                "avg_buy_and_hold_equity": 1.02,
            },
            "ablation": {"v2_minus_v1_like_avg_equity_gap": 0.03},
            "sweep": {
                "sweep_report_path": str(Path(config["sweep"]["report_path"])),
                "best_v2_minus_v1_like_avg_equity_gap": 0.04,
            },
            "selection": {"promoted": True, "active_model_path": f"models/{config['logging']['symbol']}.zip"},
        }

    def fake_harden_control_config(**kwargs):
        harden_calls.append(kwargs)
        output = Path(kwargs["output_config_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("generated", encoding="utf-8")
        return {
            "output_config_path": str(output),
            "selected_rank": kwargs["rank"],
            "selected_delta": 0.04,
        }

    def fake_render_tabs(*, output_path, ticker_reports, title, ticker_categories=None):
        _ = title, ticker_categories
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({key: str(value) for key, value in ticker_reports.items()}), encoding="utf-8")
        return out

    monkeypatch.setattr(ms_module, "run_research_control_config", fake_run_research_control_config)
    monkeypatch.setattr(ms_module, "harden_control_config", fake_harden_control_config)
    monkeypatch.setattr(ms_module, "render_ticker_tabs_report", fake_render_tabs)

    summary = run_multi_symbol_control(
        base_config_path=base_config,
        symbols=["AAPL", "MSFT"],
        output_root=tmp_path / "multi",
        refresh=True,
        start="2024-01-01",
        end="2024-12-31",
        harden_configs=True,
        hardened_rank=1,
        build_dashboard=True,
    )

    assert summary["symbols_run"] == 2
    assert Path(summary["summary_report_path"]).exists()
    assert summary["dashboard_path"] is not None
    assert Path(summary["dashboard_path"]).exists()
    assert len(control_calls) == 2
    assert len(harden_calls) == 2
    for call in control_calls:
        assert call["data"]["refresh"] is True
        assert call["data"]["start"] == "2024-01-01"
        assert call["data"]["end"] == "2024-12-31"
        assert call["data"]["path"].endswith(f"{str(call['symbol']).lower()}.csv")
    assert set(summary["hardened_config_paths"]) == {"AAPL", "MSFT"}


def test_profile_schedule_workflow_runs_shadow_rollback_and_dashboard(monkeypatch, tmp_path):
    import mekubbal.profile_schedule as schedule_module

    matrix_config = tmp_path / "profile-matrix.toml"
    schedule_config = tmp_path / "profile-schedule.toml"
    matrix_config.write_text('symbols = ["AAPL"]\n', encoding="utf-8")
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
max_gap_drop = 0.05
max_rank_worsening = 1.0
min_active_minus_base_gap = -0.05

[rollback]
enabled = true
rollback_state_path = "reports/profile_rollback_state.json"
min_consecutive_alert_runs = 2
rollback_profile = "base"
apply_rollback = false

[shadow]
enabled = true
production_state_path = "reports/prod_selection_state.json"
shadow_state_path = "reports/shadow_selection_state.json"
window_runs = 1
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
ticker_summary_csv_path = "reports/shadow_ticker_health_summary.csv"
ticker_summary_html_path = "reports/shadow_ticker_health_summary.html"
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
                "shadow_marker": "validated",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        schedule_module,
        "load_profile_matrix_config",
        lambda _path: {
            "matrix": {"output_root": str(output_root)},
            "promotion": {"state_path": "reports/profile_selection_state.json"},
        },
    )

    def fake_run_profile_matrix(config_path, *, symbols_override, promotion_override=None):
        _ = config_path, symbols_override, promotion_override
        return {
            "output_root": str(output_root),
            "symbol_summary_path": str(symbol_summary),
            "profile_selection": {"state_path": str(shadow_state)},
            "dashboard_path": str(reports_root / "profile_matrix_workspace.html"),
            "profile_aggregate_html_path": str(reports_root / "profile_aggregate_leaderboard.html"),
            "profile_pairwise_html_path": str(reports_root / "profile_pairwise_across_symbols.html"),
        }

    def fake_run_profile_monitor(**kwargs):
        snapshot_path = Path(str(kwargs["health_snapshot_path"]))
        history_path = Path(str(kwargs["health_history_path"]))
        ticker_summary_csv = Path(str(kwargs["ticker_summary_csv_path"]))
        ticker_summary_html = Path(str(kwargs["ticker_summary_html_path"]))
        drift_alerts_html = Path(str(kwargs["drift_alerts_html_path"]))

        for path in [snapshot_path, history_path, ticker_summary_csv, ticker_summary_html, drift_alerts_html]:
            path.parent.mkdir(parents=True, exist_ok=True)

        state_path = Path(str(kwargs["selection_state_path"]))
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        active_profile = str(state_payload["active_profiles"]["AAPL"])

        pd.DataFrame(
            [
                {
                    "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                    "symbol": "AAPL",
                    "selected_profile": active_profile,
                    "active_profile": active_profile,
                    "active_profile_source": "selection_state",
                    "active_rank": 1,
                    "active_gap": 0.03,
                }
            ]
        ).to_csv(snapshot_path, index=False)
        pd.DataFrame(
            [
                {
                    "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                    "symbol": "AAPL",
                    "active_profile": active_profile,
                    "selected_profile": active_profile,
                    "active_rank": 1,
                    "active_gap": 0.03,
                    "selected_gap": 0.03,
                    "active_minus_base_gap": 0.0,
                }
            ]
        ).to_csv(history_path, index=False)
        pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "status": "Healthy",
                    "selected_profile": active_profile,
                    "active_profile": active_profile,
                    "active_profile_source": "selection_state",
                    "active_rank": 1,
                    "active_vs_buy_and_hold": "+3.00%",
                    "active_vs_base": "+0.00%",
                    "recommended_action": "Keep current active profile.",
                    "summary": "base vs buy-and-hold +3.00%; vs base +0.00%.",
                }
            ]
        ).to_csv(ticker_summary_csv, index=False)
        ticker_summary_html.write_text("<html>ticker summary</html>", encoding="utf-8")
        drift_alerts_html.write_text("<html>drift</html>", encoding="utf-8")
        return {
            "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
            "health_snapshot_path": str(snapshot_path),
            "health_history_path": str(history_path),
            "ticker_summary_csv_path": str(ticker_summary_csv),
            "ticker_summary_html_path": str(ticker_summary_html),
            "drift_alerts_html_path": str(drift_alerts_html),
            "symbols_in_snapshot": 1,
            "history_rows": 1,
            "alerts_count": 0,
            "ensemble_alerts_count": 0,
        }

    monkeypatch.setattr(schedule_module, "run_profile_matrix", fake_run_profile_matrix)
    monkeypatch.setattr(schedule_module, "run_profile_monitor", fake_run_profile_monitor)

    summary = run_profile_schedule(schedule_config)

    assert summary["shadow_summary"]["comparison_summary"]["overall_gate_passed"] is True
    assert summary["shadow_summary"]["promotion_applied"] is True
    assert summary["rollback_summary"] is not None
    assert summary["rollback_summary"]["rollback_recommended_count"] == 0
    assert Path(summary["rollback_summary"]["rollback_state_path"]).exists()
    assert Path(summary["ops_summary"]["journal_csv_path"]).exists()
    assert Path(summary["product_dashboard_path"]).exists()

    production_loaded = json.loads(production_state.read_text(encoding="utf-8"))
    assert production_loaded["shadow_marker"] == "validated"
    assert production_loaded["shadow_gate"]["window_runs"] == 1


def test_profile_runner_workflow_propagates_shared_data_overrides(monkeypatch, tmp_path):
    import mekubbal.profile_runner as runner_module

    runner_config = tmp_path / "profile-runner.toml"
    base_control = tmp_path / "base.toml"
    hardened_control = tmp_path / "hardened.toml"
    data_path = tmp_path / "shared.csv"
    base_control.write_text("[data]\npath='unused.csv'\n", encoding="utf-8")
    hardened_control.write_text("[data]\npath='unused.csv'\n", encoding="utf-8")
    runner_config.write_text(
        f"""
[runner]
output_root = "{tmp_path / "out"}"
profile_summary_path = "reports/profile_summary.csv"
pairwise_csv_path = "reports/pairwise.csv"
pairwise_html_path = "reports/pairwise.html"
dashboard_path = "reports/dashboard.html"
dashboard_title = "Profile Dashboard"
build_dashboard = true

[data]
path = "{data_path}"
refresh = true
symbol = "AAPL"
start = "2024-01-01"
end = "2024-12-31"

[comparison]
confidence_level = 0.9
bootstrap_samples = 300
permutation_samples = 1000
seed = 5
title = "Profile Pairwise"

[[profiles]]
name = "base"
config = "{base_control}"

[[profiles]]
name = "hardened"
config = "{hardened_control}"
""".strip(),
        encoding="utf-8",
    )

    control_calls: list[dict[str, object]] = []

    def fake_load_control_config(config_path):
        _ = config_path
        return {
            "data": {"path": "unused.csv", "refresh": False, "symbol": None, "start": None, "end": None},
            "walkforward": {"models_dir": "models/wf", "report_path": "logs/wf.csv"},
            "ablation": {"models_dir": "models/ab", "report_path": "logs/ab.csv", "summary_path": "logs/abs.csv"},
            "sweep": {"output_dir": "logs/sw", "report_path": "logs/sw.csv"},
            "selection": {"report_path": "logs/wf.csv", "state_path": "models/current_model.json"},
            "visualization": {"output_path": "logs/report.html", "title": "Report"},
            "logging": {"symbol": None},
        }

    def fake_run_research_control_config(control_cfg, *, config_label):
        control_calls.append({"label": config_label, "data": dict(control_cfg["data"]), "logging": dict(control_cfg["logging"])})
        profile = config_label.split(":")[-1]
        walk_path = Path(control_cfg["walkforward"]["report_path"])
        walk_path.parent.mkdir(parents=True, exist_ok=True)
        policy = [1.02, 1.03] if profile == "base" else [1.05, 1.06]
        pd.DataFrame(
            {
                "fold_index": [1, 2],
                "test_start_date": ["2024-01-01", "2024-04-01"],
                "test_end_date": ["2024-03-31", "2024-06-30"],
                "policy_final_equity": policy,
                "buy_and_hold_equity": [1.00, 1.01],
            }
        ).to_csv(walk_path, index=False)

        visual = Path(control_cfg["visualization"]["output_path"])
        visual.parent.mkdir(parents=True, exist_ok=True)
        visual.write_text("<html></html>", encoding="utf-8")
        return {
            "visual_report_path": str(visual),
            "walkforward": {
                "avg_policy_final_equity": sum(policy) / len(policy),
                "avg_buy_and_hold_equity": 1.005,
            },
        }

    monkeypatch.setattr(runner_module, "load_control_config", fake_load_control_config)
    monkeypatch.setattr(runner_module, "run_research_control_config", fake_run_research_control_config)

    summary = run_profile_runner(runner_config)

    assert summary["profile_count"] == 2
    assert Path(summary["profile_summary_path"]).exists()
    assert Path(summary["pairwise_summary"]["output_csv_path"]).exists()
    assert Path(summary["dashboard_path"]).exists()
    assert len(control_calls) == 2
    for call in control_calls:
        assert call["data"]["refresh"] is True
        assert call["data"]["path"] == str(data_path.resolve())
        assert call["data"]["symbol"] == "AAPL"
        assert call["data"]["start"] == "2024-01-01"
        assert call["data"]["end"] == "2024-12-31"
        assert call["logging"]["symbol"] == "AAPL"

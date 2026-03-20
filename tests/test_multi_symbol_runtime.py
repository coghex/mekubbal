from __future__ import annotations

from pathlib import Path

import pytest

from mekubbal.multi_symbol_runtime import (
    build_multi_symbol_row,
    build_symbol_control_config,
    maybe_harden_symbol_config,
)


def test_build_symbol_control_config_scopes_paths_per_symbol(tmp_path):
    base_config = {
        "data": {"path": "data/base.csv", "refresh": False, "symbol": None, "start": None, "end": None},
        "walkforward": {"models_dir": "models/wf", "report_path": "logs/wf.csv"},
        "ablation": {"models_dir": "models/ab", "report_path": "logs/ab.csv", "summary_path": "logs/abs.csv"},
        "sweep": {"output_dir": "logs/sw", "report_path": "logs/sw.csv"},
        "selection": {"report_path": "logs/wf.csv", "state_path": "models/current_model.json"},
        "visualization": {"output_path": "logs/report.html", "title": "Report"},
        "logging": {"symbol": None},
    }

    config = build_symbol_control_config(
        base_config=base_config,
        symbol="AAPL",
        root=tmp_path / "multi",
        reports_root=tmp_path / "multi" / "reports",
        data_path_template="data/{symbol_lower}.csv",
        refresh=True,
        start="2024-01-01",
        end="2024-12-31",
    )

    assert config["data"]["path"] == "data/aapl.csv"
    assert config["walkforward"]["report_path"].endswith("walkforward_aapl.csv")
    assert config["selection"]["state_path"].endswith("aapl/models/current_model.json")
    assert config["visualization"]["title"] == "AAPL Research Report"
    assert config["logging"]["symbol"] == "AAPL"


def test_maybe_harden_symbol_config_uses_sweep_fallback(tmp_path):
    calls: list[dict[str, object]] = []

    def fake_harden_control_config(**kwargs):
        calls.append(kwargs)
        return {"output_config_path": str(kwargs["output_config_path"]), "selected_rank": kwargs["rank"]}

    result = maybe_harden_symbol_config(
        enabled=True,
        summary={"sweep": {}},
        config={"sweep": {"report_path": str(tmp_path / "reports" / "fallback.csv")}},
        base_config_path=tmp_path / "control.toml",
        configs_root=tmp_path / "configs",
        hardened_rank=2,
        hardened_profile_template="profile-{symbol_lower}",
        symbol="MSFT",
        harden_control_config_fn=fake_harden_control_config,
    )

    assert result is not None
    assert calls[0]["profile"] == "profile-msft"
    assert Path(calls[0]["sweep_report_path"]).name == "fallback.csv"


def test_build_multi_symbol_row_computes_walk_gap_and_hardening_fields():
    row = build_multi_symbol_row(
        symbol="NVDA",
        summary={
            "data_path": "data/nvda.csv",
            "visual_report_path": "logs/reports/nvda.html",
            "walkforward": {
                "avg_policy_final_equity": 1.08,
                "avg_buy_and_hold_equity": 1.03,
            },
            "ablation": {"v2_minus_v1_like_avg_equity_gap": 0.02},
            "sweep": {"best_v2_minus_v1_like_avg_equity_gap": 0.03},
            "selection": {"promoted": True, "active_model_path": "models/nvda.zip"},
            "lineage": {"config_profile": "base", "config_version": 2, "git_commit": "abc123"},
        },
        hardening={"output_config_path": "configs/nvda.hardened.toml", "selected_delta": 0.03, "selected_rank": 1},
    )

    assert row["walkforward_avg_equity_gap"] == pytest.approx(0.05)
    assert row["hardened_config_path"] == "configs/nvda.hardened.toml"
    assert row["hardened_selected_rank"] == 1

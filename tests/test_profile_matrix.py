from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mekubbal.profile_matrix import load_profile_matrix_config, run_profile_matrix


def test_load_profile_matrix_config_requires_symbols(tmp_path):
    profile_runner = tmp_path / "profile-runner.toml"
    control = tmp_path / "control.toml"
    control.write_text("[data]\npath='x'\n", encoding="utf-8")
    profile_runner.write_text(
        f"""
[[profiles]]
name = "base"
config = "{control}"

[[profiles]]
name = "candidate"
config = "{control}"
""".strip(),
        encoding="utf-8",
    )
    matrix = tmp_path / "profile-matrix.toml"
    matrix.write_text(
        f"""
symbols = []

[base_runner]
config = "{profile_runner}"
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one ticker"):
        load_profile_matrix_config(matrix)


def test_run_profile_matrix_generates_aggregate_outputs(monkeypatch, tmp_path):
    import mekubbal.profile_matrix as matrix_module

    control = tmp_path / "control.toml"
    profile_runner = tmp_path / "profile-runner.toml"
    matrix_config = tmp_path / "profile-matrix.toml"
    control.write_text("[data]\npath='unused.csv'\n", encoding="utf-8")
    profile_runner.write_text(
        f"""
[[profiles]]
name = "base"
config = "{control}"

[[profiles]]
name = "candidate"
config = "{control}"
""".strip(),
        encoding="utf-8",
    )
    matrix_config.write_text(
        f"""
symbols = ["AAPL", "MSFT"]

[matrix]
output_root = "{tmp_path / "out"}"
symbol_summary_path = "reports/profile_symbol_summary.csv"
profile_aggregate_csv_path = "reports/profile_aggregate.csv"
profile_aggregate_html_path = "reports/profile_aggregate.html"
profile_pairwise_csv_path = "reports/profile_pairwise.csv"
profile_pairwise_html_path = "reports/profile_pairwise.html"
dashboard_path = "reports/profile_workspace.html"
dashboard_title = "Matrix Workspace"
build_dashboard = true
include_symbol_pairwise_leaderboards = true

[base_runner]
config = "{profile_runner}"
output_root_template = "symbols/{{symbol_lower}}"
data_path_template = "data/{{symbol_lower}}.csv"
refresh = false
start = ""
end = ""
build_symbol_dashboards = true

[comparison]
confidence_level = 0.9
bootstrap_samples = 300
permutation_samples = 1000
seed = 11
aggregate_title = "Aggregate"
pairwise_title = "Pairwise"
""".strip(),
        encoding="utf-8",
    )

    def fake_run_profile_runner_config(config, *, config_dir, config_label):
        _ = config_dir, config_label
        symbol = str(config["data"]["symbol"]).upper()
        out_root = Path(config["runner"]["output_root"])
        out_root.mkdir(parents=True, exist_ok=True)
        pairwise_csv = out_root / "pairwise.csv"
        pairwise_html = out_root / "pairwise.html"
        pd.DataFrame(
            {
                "profile_a": ["base"],
                "profile_b": ["candidate"],
                "p_value_two_sided": [0.5],
            }
        ).to_csv(pairwise_csv, index=False)
        pairwise_html.write_text("<html>pairwise</html>", encoding="utf-8")

        dashboard = out_root / "dashboard.html"
        dashboard.write_text("<html>dashboard</html>", encoding="utf-8")

        baseline_gap = 0.02 if symbol == "AAPL" else -0.01
        candidate_gap = 0.03 if symbol == "AAPL" else 0.04
        return {
            "pairwise_summary": {
                "output_csv_path": str(pairwise_csv),
                "output_html_path": str(pairwise_html),
            },
            "dashboard_path": str(dashboard),
            "profiles": [
                {
                    "profile": "base",
                    "profile_slug": "base",
                    "walkforward_report_path": str(out_root / "walk_base.csv"),
                    "visual_report_path": str(out_root / "base.html"),
                    "walkforward_avg_policy_final_equity": 1.0 + baseline_gap,
                    "walkforward_avg_buy_and_hold_equity": 1.0,
                },
                {
                    "profile": "candidate",
                    "profile_slug": "candidate",
                    "walkforward_report_path": str(out_root / "walk_candidate.csv"),
                    "visual_report_path": str(out_root / "candidate.html"),
                    "walkforward_avg_policy_final_equity": 1.0 + candidate_gap,
                    "walkforward_avg_buy_and_hold_equity": 1.0,
                },
            ],
        }

    def fake_render_tabs(*, output_path, ticker_reports, leaderboard_reports, title):
        _ = ticker_reports, leaderboard_reports, title
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<html>workspace</html>", encoding="utf-8")
        return output

    monkeypatch.setattr(matrix_module, "run_profile_runner_config", fake_run_profile_runner_config)
    monkeypatch.setattr(matrix_module, "render_ticker_tabs_report", fake_render_tabs)

    summary = run_profile_matrix(matrix_config)
    assert summary["symbols_run"] == 2
    assert summary["profile_count"] == 2
    assert Path(summary["symbol_summary_path"]).exists()
    assert Path(summary["profile_aggregate_csv_path"]).exists()
    assert Path(summary["profile_aggregate_html_path"]).exists()
    assert Path(summary["profile_pairwise_csv_path"]).exists()
    assert Path(summary["profile_pairwise_html_path"]).exists()
    assert summary["dashboard_path"] is not None
    assert Path(summary["dashboard_path"]).exists()

    aggregate = pd.read_csv(summary["profile_aggregate_csv_path"])
    assert set(aggregate["profile"]) == {"base", "candidate"}
    assert "win_rate" in aggregate.columns
    assert "net_significant_wins" in aggregate.columns

    per_symbol = pd.read_csv(summary["symbol_summary_path"])
    assert len(per_symbol) == 4
    assert set(per_symbol["symbol"]) == {"AAPL", "MSFT"}
    assert set(per_symbol["symbol_rank"]) == {1, 2}


def test_run_profile_matrix_invokes_profile_selection_when_enabled(monkeypatch, tmp_path):
    import mekubbal.profile_matrix as matrix_module

    control = tmp_path / "control.toml"
    profile_runner = tmp_path / "profile-runner.toml"
    matrix_config = tmp_path / "profile-matrix.toml"
    control.write_text("[data]\npath='unused.csv'\n", encoding="utf-8")
    profile_runner.write_text(
        f"""
[[profiles]]
name = "base"
config = "{control}"

[[profiles]]
name = "candidate"
config = "{control}"
""".strip(),
        encoding="utf-8",
    )
    matrix_config.write_text(
        f"""
symbols = ["AAPL"]

[matrix]
output_root = "{tmp_path / "out"}"
build_dashboard = false

[base_runner]
config = "{profile_runner}"
output_root_template = "symbols/{{symbol_lower}}"
data_path_template = "data/{{symbol_lower}}.csv"
refresh = false
start = ""
end = ""
build_symbol_dashboards = false

[comparison]
confidence_level = 0.9
bootstrap_samples = 300
permutation_samples = 1000
seed = 11
aggregate_title = "Aggregate"
pairwise_title = "Pairwise"

[promotion]
enabled = true
state_path = "reports/selection_state.json"
base_profile = "base"
candidate_profile = "candidate"
min_candidate_gap_vs_base = 0.0
max_candidate_rank = 1
require_candidate_significant = false
forbid_base_significant_better = true
prefer_previous_active = true
fallback_profile = "base"
""".strip(),
        encoding="utf-8",
    )

    def fake_run_profile_runner_config(config, *, config_dir, config_label):
        _ = config, config_dir, config_label
        out_root = tmp_path / "out" / "symbols" / "aapl"
        out_root.mkdir(parents=True, exist_ok=True)
        pairwise_csv = out_root / "pairwise.csv"
        pairwise_html = out_root / "pairwise.html"
        pd.DataFrame(
            {
                "profile_a": ["candidate"],
                "profile_b": ["base"],
                "profile_a_better_significant": [True],
                "profile_b_better_significant": [False],
            }
        ).to_csv(pairwise_csv, index=False)
        pairwise_html.write_text("<html>pairwise</html>", encoding="utf-8")
        return {
            "pairwise_summary": {
                "output_csv_path": str(pairwise_csv),
                "output_html_path": str(pairwise_html),
            },
            "dashboard_path": None,
            "profiles": [
                {
                    "profile": "base",
                    "profile_slug": "base",
                    "walkforward_report_path": str(out_root / "walk_base.csv"),
                    "visual_report_path": str(out_root / "base.html"),
                    "walkforward_avg_policy_final_equity": 1.01,
                    "walkforward_avg_buy_and_hold_equity": 1.0,
                },
                {
                    "profile": "candidate",
                    "profile_slug": "candidate",
                    "walkforward_report_path": str(out_root / "walk_candidate.csv"),
                    "visual_report_path": str(out_root / "candidate.html"),
                    "walkforward_avg_policy_final_equity": 1.03,
                    "walkforward_avg_buy_and_hold_equity": 1.0,
                },
            ],
        }

    captured: dict[str, object] = {}

    def fake_run_profile_promotion(**kwargs):
        captured.update(kwargs)
        return {"state_path": str(tmp_path / "out" / "reports" / "selection_state.json")}

    monkeypatch.setattr(matrix_module, "run_profile_runner_config", fake_run_profile_runner_config)
    monkeypatch.setattr(matrix_module, "run_profile_promotion", fake_run_profile_promotion)

    summary = run_profile_matrix(matrix_config)
    assert summary["profile_selection"] is not None
    assert str(captured["base_profile"]) == "base"
    assert str(captured["candidate_profile"]) == "candidate"
    assert str(captured["state_path"]).endswith("reports/selection_state.json")

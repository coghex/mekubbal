from __future__ import annotations

import json
from pathlib import Path

from mekubbal.profile.schedule_output import write_schedule_outputs


def test_write_schedule_outputs_writes_summary_and_dashboard(tmp_path):
    captured: dict[str, object] = {}

    def fake_render_product_dashboard(output_path, **kwargs):
        captured["output_path"] = output_path
        captured["global_report_paths"] = kwargs["global_report_paths"]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("<html></html>", encoding="utf-8")
        return out

    summary = write_schedule_outputs(
        config_label="schedule.toml",
        schedule_cfg={
            "summary_json_path": "reports/summary.json",
            "product_dashboard_path": "reports/dashboard.html",
            "product_dashboard_title": "Dashboard",
        },
        matrix_output_root=tmp_path,
        matrix_summary={
            "symbol_summary_path": str(tmp_path / "reports" / "profile_symbol_summary.csv"),
            "dashboard_path": str(tmp_path / "reports" / "matrix_dashboard.html"),
        },
        monitor_summary={
            "ticker_summary_csv_path": str(tmp_path / "reports" / "ticker_summary.csv"),
            "ticker_summary_html_path": str(tmp_path / "reports" / "ticker_summary.html"),
            "health_history_path": str(tmp_path / "reports" / "health_history.csv"),
            "drift_alerts_html_path": str(tmp_path / "reports" / "drift_alerts.html"),
            "ensemble_alerts_html_path": str(tmp_path / "reports" / "ensemble_alerts.html"),
        },
        shadow_summary={
            "comparison_summary": {
                "comparison_html_path": str(tmp_path / "reports" / "shadow_comparison.html"),
                "gate_json_path": str(tmp_path / "reports" / "shadow_gate.json"),
                "comparison_history_path": str(tmp_path / "reports" / "shadow_history.csv"),
            },
            "suggestion_summary": {
                "suggestion_html_path": str(tmp_path / "reports" / "shadow_suggestion.html"),
                "suggestion_json_path": str(tmp_path / "reports" / "shadow_suggestion.json"),
                "suggestion_history_path": str(tmp_path / "reports" / "shadow_suggestion_history.csv"),
                "suggestion_state_path": str(tmp_path / "reports" / "shadow_suggestion_state.json"),
            },
        },
        rollback_summary={
            "rollback_state_path": str(tmp_path / "reports" / "rollback_state.json"),
        },
        ops_summary={
            "digest_html_path": str(tmp_path / "reports" / "ops_digest.html"),
            "journal_csv_path": str(tmp_path / "reports" / "ops_journal.csv"),
        },
        render_product_dashboard_fn=fake_render_product_dashboard,
    )

    assert Path(summary["summary_json_path"]).exists()
    assert Path(summary["product_dashboard_path"]).exists()
    assert captured["global_report_paths"]["Shadow comparison"].endswith("shadow_comparison.html")
    assert captured["global_report_paths"]["Rollback state JSON"].endswith("rollback_state.json")

    loaded = json.loads(Path(summary["summary_json_path"]).read_text(encoding="utf-8"))
    assert loaded["product_dashboard_path"].endswith("dashboard.html")
    assert loaded["summary_json_path"].endswith("summary.json")

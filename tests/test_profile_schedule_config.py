from __future__ import annotations

import pytest

from mekubbal.profile.schedule_config import load_profile_schedule_config


def test_load_profile_schedule_config_merges_defaults_and_normalizes_symbols(tmp_path):
    matrix_config = tmp_path / "profile-matrix.toml"
    matrix_config.write_text("symbols = [\"AAPL\"]\n", encoding="utf-8")
    schedule_config = tmp_path / "profile-schedule.toml"
    schedule_config.write_text(
        f"""
[schedule]
matrix_config = "{matrix_config.name}"
symbols = ["aapl", "msft"]
""".strip(),
        encoding="utf-8",
    )

    loaded = load_profile_schedule_config(schedule_config)

    assert loaded["schedule"]["symbols"] == ["AAPL", "MSFT"]
    assert loaded["schedule"]["ops_digest_lookback_runs"] == 14
    assert loaded["monitor"]["ensemble_low_confidence_threshold"] == 0.55
    assert loaded["ensemble_v3"]["decision_csv_path"] == "reports/profile_ensemble_decisions.csv"


def test_load_profile_schedule_config_uses_sibling_matrix_config_by_default(tmp_path):
    matrix_config = tmp_path / "profile-matrix.toml"
    matrix_config.write_text('symbols = ["AAPL"]\n', encoding="utf-8")
    schedule_config = tmp_path / "profile-schedule.toml"
    schedule_config.write_text("[schedule]\nsymbols = []\n", encoding="utf-8")

    loaded = load_profile_schedule_config(schedule_config)

    assert loaded["schedule"]["matrix_config"] == "profile-matrix.toml"


def test_load_profile_schedule_config_rejects_invalid_shadow_min_match_ratio(tmp_path):
    matrix_config = tmp_path / "profile-matrix.toml"
    matrix_config.write_text("symbols = [\"AAPL\"]\n", encoding="utf-8")
    schedule_config = tmp_path / "profile-schedule.toml"
    schedule_config.write_text(
        f"""
[schedule]
matrix_config = "{matrix_config.name}"

[shadow]
min_match_ratio = 1.5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="shadow.min_match_ratio must be in \\[0, 1\\]"):
        load_profile_schedule_config(schedule_config)

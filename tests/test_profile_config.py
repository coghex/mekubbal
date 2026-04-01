from __future__ import annotations

from copy import deepcopy
import pytest

from mekubbal.profile.config import (
    deep_merge,
    normalize_symbol_overrides,
    parse_symbols,
    resolve_existing_path,
    resolve_path,
    templated_path,
)


def test_deep_merge_merges_nested_dicts():
    base = {"runner": {"enabled": True, "count": 1}, "name": "base"}
    override = {"runner": {"count": 3, "extra": "x"}, "name": "candidate"}

    merged = deep_merge(deepcopy(base), override)

    assert merged == {
        "runner": {"enabled": True, "count": 3, "extra": "x"},
        "name": "candidate",
    }


def test_resolve_path_anchors_relative_paths_to_base_dir(tmp_path):
    base_dir = tmp_path / "configs"
    base_dir.mkdir()

    resolved = resolve_path(base_dir, "reports/output.csv")

    assert resolved == (base_dir / "reports" / "output.csv").resolve()


def test_resolve_existing_path_prefers_config_dir_when_file_exists(tmp_path, monkeypatch):
    base_dir = tmp_path / "configs"
    base_dir.mkdir()
    local = base_dir / "profile-runner.toml"
    local.write_text("", encoding="utf-8")

    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = resolve_existing_path(base_dir, "profile-runner.toml")

    assert resolved == local.resolve()


def test_parse_symbols_deduplicates_and_uppercases():
    assert parse_symbols(["aapl", " AAPL ", "msft"], field_name="symbols", require_non_empty=True) == [
        "AAPL",
        "MSFT",
    ]


def test_parse_symbols_allows_empty_when_requested():
    assert parse_symbols([], field_name="schedule.symbols", require_non_empty=False) == []


def test_parse_symbols_rejects_missing_required_symbols():
    with pytest.raises(ValueError, match="must contain at least one ticker"):
        parse_symbols([], field_name="symbols", require_non_empty=True)


def test_normalize_symbol_overrides_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown keys"):
        normalize_symbol_overrides({"aapl": {"unsupported": True}})


def test_normalize_symbol_overrides_uppercases_symbols():
    normalized = normalize_symbol_overrides({"aapl": {"refresh": True}})

    assert normalized == {"AAPL": {"refresh": True}}


def test_templated_path_renders_symbol_variants():
    rendered = templated_path("symbols/{symbol_lower}/{symbol_upper}/{symbol}", "AAPL")

    assert rendered == "symbols/aapl/AAPL/AAPL"


def test_templated_path_preserves_prefixed_crypto_symbols():
    rendered = templated_path("data/{symbol_lower}.csv", "$BTC")

    assert rendered == "data/$btc.csv"


def test_templated_path_preserves_non_uniform_foreign_tickers():
    rendered = templated_path("data/{symbol_lower}.csv", "005930.KS")

    assert rendered == "data/005930.ks.csv"

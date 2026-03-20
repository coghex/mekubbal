from __future__ import annotations

from pathlib import Path
from typing import Any

from mekubbal.config import deep_merge

__all__ = [
    "deep_merge",
    "resolve_path",
    "resolve_existing_path",
    "parse_symbols",
    "templated_path",
    "normalize_symbol_overrides",
]


def resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def resolve_existing_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_config_dir = (base_dir / path).resolve()
    if from_config_dir.exists():
        return from_config_dir
    return path.resolve()


def parse_symbols(
    values: Any,
    *,
    field_name: str,
    require_non_empty: bool,
) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list.")
    symbols: list[str] = []
    seen: set[str] = set()
    for raw in values:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    if require_non_empty and not symbols:
        raise ValueError(f"{field_name} must contain at least one ticker.")
    return symbols


def templated_path(template: str, symbol: str) -> str:
    return template.format(symbol=symbol, symbol_lower=symbol.lower(), symbol_upper=symbol.upper())


def normalize_symbol_overrides(raw_overrides: Any) -> dict[str, dict[str, Any]]:
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, dict):
        raise ValueError("symbol_overrides must be a TOML table when provided.")

    allowed_keys = {
        "config",
        "output_root_template",
        "data_path_template",
        "refresh",
        "start",
        "end",
        "build_symbol_dashboards",
    }
    normalized: dict[str, dict[str, Any]] = {}
    for raw_symbol, raw_override in raw_overrides.items():
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol_overrides keys must be non-empty ticker symbols.")
        if not isinstance(raw_override, dict):
            raise ValueError(f"symbol_overrides.{symbol} must be a TOML table.")
        unknown_keys = sorted(set(raw_override) - allowed_keys)
        if unknown_keys:
            raise ValueError(f"symbol_overrides.{symbol} contains unknown keys: {unknown_keys}")
        normalized[symbol] = dict(raw_override)
    return normalized

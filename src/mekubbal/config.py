from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_toml_table(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {file_path}")
    with file_path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must decode to a TOML table.")
    return loaded


def validate_config_types(
    config: dict[str, Any], schema: dict[str, type]
) -> list[str]:
    """Check that config values match expected types. Returns list of error strings."""
    errors: list[str] = []
    for key, expected_type in schema.items():
        if key in config and not isinstance(config[key], expected_type):
            errors.append(
                f"{key}: expected {expected_type.__name__}, "
                f"got {type(config[key]).__name__}"
            )
    return errors

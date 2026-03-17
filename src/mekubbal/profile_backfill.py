from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from mekubbal.control import load_control_config
from mekubbal.data import load_ohlcv_csv
from mekubbal.profile_matrix import (
    _base_runner_settings,
    _resolve_existing_path,
    _resolve_path,
    _templated_path,
    load_profile_matrix_config,
)
from mekubbal.profile_runner import load_profile_runner_config
from mekubbal.profile_schedule import load_profile_schedule_config, run_profile_schedule_config


def _active_symbols(schedule_config: dict[str, Any], matrix_config: dict[str, Any]) -> list[str]:
    schedule_symbols = [str(value).strip().upper() for value in schedule_config["schedule"]["symbols"]]
    return schedule_symbols or [str(value).strip().upper() for value in matrix_config["symbols"]]


def _required_rows_for_control(control_config: dict[str, Any]) -> int:
    walkforward = control_config["walkforward"]
    ablation = control_config["ablation"]
    sweep = control_config["sweep"]

    required_rows = 1
    walk_train = int(walkforward["train_window"])
    walk_test = int(walkforward["test_window"])
    if bool(walkforward["enabled"]):
        required_rows = max(required_rows, walk_train + walk_test)

    if bool(ablation["enabled"]):
        ablation_train = walk_train if ablation.get("train_window") is None else int(ablation["train_window"])
        ablation_test = walk_test if ablation.get("test_window") is None else int(ablation["test_window"])
        required_rows = max(required_rows, ablation_train + ablation_test)

    if bool(sweep["enabled"]):
        sweep_train = walk_train if sweep.get("train_window") is None else int(sweep["train_window"])
        sweep_test = walk_test if sweep.get("test_window") is None else int(sweep["test_window"])
        required_rows = max(required_rows, sweep_train + sweep_test)

    return required_rows


def _required_rows_for_runner_config(runner_config_path: Path) -> int:
    runner_config = load_profile_runner_config(runner_config_path)
    runner_config_dir = runner_config_path.parent

    required_rows = 1
    for profile in runner_config["profiles"]:
        profile_config_path = _resolve_existing_path(runner_config_dir, str(profile["config"]))
        control_config = load_control_config(profile_config_path)
        required_rows = max(required_rows, _required_rows_for_control(control_config))
    return required_rows


def _symbol_backfill_settings(
    matrix_config: dict[str, Any],
    *,
    matrix_config_dir: Path,
    symbols: list[str],
) -> dict[str, dict[str, Any]]:
    required_rows_cache: dict[Path, int] = {}
    settings: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        base_runner = _base_runner_settings(matrix_config, symbol)
        runner_config_path = _resolve_existing_path(matrix_config_dir, str(base_runner["config"]))
        if runner_config_path not in required_rows_cache:
            required_rows_cache[runner_config_path] = _required_rows_for_runner_config(runner_config_path)
        base_runner_dir = runner_config_path.parent
        data_path = _resolve_existing_path(
            base_runner_dir,
            _templated_path(str(base_runner["data_path_template"]), symbol),
        )
        settings[symbol] = {
            "base_runner": base_runner,
            "runner_config_path": runner_config_path,
            "data_path": data_path,
            "required_rows": required_rows_cache[runner_config_path],
        }
    return settings


def _candidate_replay_dates(
    frames: dict[str, pd.DataFrame],
    *,
    required_rows_by_symbol: dict[str, int],
    start_date: str | None,
    end_date: str | None,
    every: int,
    max_runs: int | None,
) -> list[pd.Timestamp]:
    if every < 1:
        raise ValueError("every must be >= 1.")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be >= 1 when provided.")

    normalized_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        ordered = frame.sort_values("date").reset_index(drop=True).copy()
        ordered["date"] = pd.to_datetime(ordered["date"]).dt.normalize()
        normalized_frames[symbol] = ordered

    all_dates: set[pd.Timestamp] = set()
    first_eligible_dates: dict[str, pd.Timestamp] = {}
    for frame in normalized_frames.values():
        all_dates.update(frame["date"].tolist())
    if not all_dates:
        raise ValueError("No replay dates exist across the selected symbols.")

    for symbol, frame in normalized_frames.items():
        required_rows = int(required_rows_by_symbol[symbol])
        first_eligible_dates[symbol] = frame.iloc[required_rows - 1]["date"]

    earliest_eligible = min(first_eligible_dates.values())
    start_ts = pd.Timestamp(start_date).normalize() if start_date else earliest_eligible
    end_ts = pd.Timestamp(end_date).normalize() if end_date else max(all_dates)
    effective_start = max(earliest_eligible, start_ts)

    candidate_dates = sorted(
        date
        for date in all_dates
        if effective_start <= date <= end_ts
        and any(first_eligible_dates[symbol] <= date for symbol in normalized_frames)
    )
    candidate_dates = candidate_dates[::every]
    if max_runs is not None:
        candidate_dates = candidate_dates[-max_runs:]
    if not candidate_dates:
        raise ValueError("No replay dates remain after applying the date filters and cadence.")
    return candidate_dates


def _partition_frames_by_history(
    frames: dict[str, pd.DataFrame],
    *,
    required_rows_by_symbol: dict[str, int],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    eligible_frames: dict[str, pd.DataFrame] = {}
    skipped_symbols: list[dict[str, Any]] = []
    for symbol, frame in frames.items():
        row_count = int(len(frame))
        required_rows = int(required_rows_by_symbol[symbol])
        if row_count < required_rows:
            skipped_symbols.append(
                {
                    "symbol": symbol,
                    "available_rows": row_count,
                    "required_rows": required_rows,
                    "reason": (
                        f"{symbol} only has {row_count} rows, but backfill requires at least "
                        f"{required_rows} rows per ticker."
                    ),
                }
            )
            continue
        eligible_frames[symbol] = frame
    if not eligible_frames:
        details = ", ".join(f"{row['symbol']}={row['available_rows']}" for row in skipped_symbols) or "none"
        raise ValueError(
            "No symbols have enough history for backfill. "
            f"Required rows per ticker: {required_rows_by_symbol}. Available: {details}."
        )
    return eligible_frames, skipped_symbols


def _active_symbols_for_replay_date(
    frames: dict[str, pd.DataFrame],
    *,
    required_rows_by_symbol: dict[str, int],
    replay_date: pd.Timestamp,
) -> list[str]:
    active_symbols: list[str] = []
    for symbol, frame in frames.items():
        normalized_dates = pd.to_datetime(frame["date"]).dt.normalize()
        available_rows = int((normalized_dates <= replay_date).sum())
        if available_rows >= int(required_rows_by_symbol[symbol]):
            active_symbols.append(symbol)
    return active_symbols


def _timestamp_for_replay_date(replay_date: pd.Timestamp) -> str:
    return replay_date.tz_localize("UTC").isoformat()


def run_profile_backfill(
    config_path: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    every: int = 1,
    max_runs: int | None = None,
    reset_output: bool = False,
) -> dict[str, Any]:
    schedule_config_path = Path(config_path).resolve()
    schedule_config = load_profile_schedule_config(schedule_config_path)
    schedule_config_dir = schedule_config_path.parent.resolve()
    matrix_config_path = _resolve_path(
        schedule_config_dir, str(schedule_config["schedule"]["matrix_config"])
    )
    matrix_config = load_profile_matrix_config(matrix_config_path)
    matrix_config_dir = matrix_config_path.parent.resolve()
    symbols = _active_symbols(schedule_config, matrix_config)

    symbol_settings = _symbol_backfill_settings(
        matrix_config,
        matrix_config_dir=matrix_config_dir,
        symbols=symbols,
    )
    source_data_paths = {symbol: Path(settings["data_path"]) for symbol, settings in symbol_settings.items()}
    source_frames = {symbol: load_ohlcv_csv(path) for symbol, path in source_data_paths.items()}
    required_rows_by_symbol = {
        symbol: int(settings["required_rows"]) for symbol, settings in symbol_settings.items()
    }
    eligible_frames, skipped_symbols = _partition_frames_by_history(
        source_frames,
        required_rows_by_symbol=required_rows_by_symbol,
    )
    eligible_required_rows_by_symbol = {
        symbol: required_rows_by_symbol[symbol] for symbol in eligible_frames
    }
    replay_dates = _candidate_replay_dates(
        eligible_frames,
        required_rows_by_symbol=eligible_required_rows_by_symbol,
        start_date=start_date,
        end_date=end_date,
        every=every,
        max_runs=max_runs,
    )

    output_root = _resolve_path(matrix_config_dir, str(matrix_config["matrix"]["output_root"]))
    if reset_output and output_root.exists():
        shutil.rmtree(output_root)

    replay_runs: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="mekubbal-backfill-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        temp_data_dir = temp_dir / "data"
        temp_data_dir.mkdir(parents=True, exist_ok=True)

        matrix_override = deepcopy(matrix_config)
        matrix_override["base_runner"]["data_path_template"] = str(temp_data_dir / "{symbol_lower}.csv")
        for override in matrix_override.get("symbol_overrides", {}).values():
            override["data_path_template"] = str(temp_data_dir / "{symbol_lower}.csv")

        for replay_date in replay_dates:
            active_symbols = _active_symbols_for_replay_date(
                eligible_frames,
                required_rows_by_symbol=eligible_required_rows_by_symbol,
                replay_date=replay_date,
            )
            if not active_symbols:
                continue
            matrix_override["symbols"] = active_symbols
            for symbol, frame in eligible_frames.items():
                snapshot = frame[pd.to_datetime(frame["date"]).dt.normalize() <= replay_date].copy()
                snapshot.to_csv(temp_data_dir / f"{symbol.lower()}.csv", index=False)

            run_summary = run_profile_schedule_config(
                schedule_config,
                config_dir=schedule_config_dir,
                config_label=str(schedule_config_path),
                run_timestamp_utc=_timestamp_for_replay_date(replay_date),
                matrix_config_override=matrix_override,
                matrix_config_dir=matrix_config_dir,
                matrix_config_label=f"{matrix_config_path}@{replay_date.date().isoformat()}",
            )
            replay_runs.append(
                {
                    "replay_date": replay_date.date().isoformat(),
                    "symbols": list(active_symbols),
                    "run_timestamp_utc": str(run_summary["monitor_summary"]["run_timestamp_utc"]),
                    "summary_json_path": str(run_summary["summary_json_path"]),
                    "product_dashboard_path": str(run_summary["product_dashboard_path"]),
                }
            )

    summary = {
        "config_path": str(schedule_config_path),
        "matrix_config_path": str(matrix_config_path),
        "output_root": str(output_root),
        "requested_symbols": symbols,
        "symbols": list(eligible_frames.keys()),
        "skipped_symbols": skipped_symbols,
        "source_data_paths": {symbol: str(path) for symbol, path in source_data_paths.items()},
        "minimum_required_rows": int(max(eligible_required_rows_by_symbol.values())),
        "minimum_required_rows_by_symbol": required_rows_by_symbol,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "every": int(every),
        "max_runs": int(max_runs) if max_runs is not None else None,
        "reset_output": bool(reset_output),
        "runs_replayed": int(len(replay_runs)),
        "first_replay_date": replay_runs[0]["replay_date"],
        "last_replay_date": replay_runs[-1]["replay_date"],
        "replay_runs": replay_runs,
    }
    summary_path = output_root / "reports" / "profile_backfill_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_json_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

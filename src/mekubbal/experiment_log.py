from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_experiment_log_schema(db_path: str | Path) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_type TEXT NOT NULL,
                symbol TEXT,
                data_path TEXT,
                model_path TEXT,
                timesteps INTEGER,
                train_ratio REAL,
                trade_cost REAL,
                cadence TEXT,
                cutoff_date TEXT,
                reward REAL,
                final_equity REAL,
                baseline_equity REAL,
                rows_logged INTEGER,
                metrics_json TEXT NOT NULL
            )
            """
        )


def _metric_float(metrics: Mapping[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _metric_int(metrics: Mapping[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return int(value)
    return None


def log_experiment_run(
    db_path: str | Path,
    run_type: str,
    metrics: Mapping[str, Any],
    *,
    symbol: str | None = None,
    data_path: str | None = None,
    model_path: str | None = None,
    timesteps: int | None = None,
    train_ratio: float | None = None,
    trade_cost: float | None = None,
    cadence: str | None = None,
    cutoff_date: str | None = None,
) -> int:
    ensure_experiment_log_schema(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    reward = _metric_float(metrics, ["policy_total_reward", "reward"])
    final_equity = _metric_float(metrics, ["policy_final_equity", "final_equity"])
    baseline_equity = _metric_float(metrics, ["buy_and_hold_equity"])
    rows_logged = _metric_int(metrics, ["rows_logged"])
    payload = json.dumps(dict(metrics), sort_keys=True)

    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiment_runs (
                created_at, run_type, symbol, data_path, model_path,
                timesteps, train_ratio, trade_cost, cadence, cutoff_date,
                reward, final_equity, baseline_equity, rows_logged, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                run_type,
                symbol,
                data_path,
                model_path,
                timesteps,
                train_ratio,
                trade_cost,
                cadence,
                cutoff_date,
                reward,
                final_equity,
                baseline_equity,
                rows_logged,
                payload,
            ),
        )
        return int(cursor.lastrowid)


def list_experiment_runs(
    db_path: str | Path,
    *,
    limit: int = 20,
    run_type: str | None = None,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be >= 1.")

    ensure_experiment_log_schema(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if run_type:
        clauses.append("run_type = ?")
        params.append(run_type)
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)

    query = """
        SELECT id, created_at, run_type, symbol, timesteps, reward, final_equity,
               baseline_equity, rows_logged, data_path, model_path, cadence, cutoff_date
        FROM experiment_runs
    """
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _connect(db_path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


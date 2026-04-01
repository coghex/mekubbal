# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mekubbal is a reinforcement learning research framework for stock market experimentation using PPO (stable-baselines3). It emphasizes reproducibility, chronological integrity, and structured experimentation over prediction accuracy.

## Build & Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Python >=3.10** required. Uses `src/` layout with setuptools.

## Commands

- **Lint:** `ruff check .` (line-length = 100)
- **All tests:** `pytest -q`
- **Single test file:** `pytest tests/test_env.py -q`
- **Single test:** `pytest tests/test_env.py::test_name -q`

Key CLI entry points (all `mekubbal-*` commands defined in `pyproject.toml [project.scripts]`):
- `mekubbal-download` / `mekubbal-train` / `mekubbal-evaluate` / `mekubbal-paper` — core data/train/eval pipeline
- `mekubbal-control --config configs/research-control.toml` — orchestrated research workflow
- `mekubbal-profile-schedule --config configs/profile-schedule.toml` — recurring production orchestration
- `mekubbal-profile-matrix --config configs/profile-matrix.toml` — multi-symbol x multi-profile runs

## Architecture

**Data pipeline:** `data.py` (yfinance OHLCV download + data gates) → `features.py` (feat_* columns, chronological train/test split) → `env.py` (Gymnasium TradingEnv) → `train.py` (PPO via stable-baselines3) → `evaluate.py` / `paper.py`

**Experimentation layer:** `walk_forward.py` (rolling/expanding folds) → `ablation.py` (v1 vs v2 comparison) → `sweep.py` (penalty grid search) → `selection.py` (model promotion) → `config_hardening.py` (versioned TOML overlays)

**Orchestration:** `control.py` chains walk-forward → ablation → sweep → selection → report from a single TOML config. `multi_symbol.py` batches control across symbols. `profile_runner.py` runs multiple profiles on one symbol; `profile_matrix.py` runs symbols x profiles.

**Production profile workflows:** `profile_selection.py` (candidate/base/active promotion + state persistence) → `profile_monitor.py` (health snapshots + drift alerts) → `profile_rollback.py` (persistent-alert rollback) → `profile_schedule.py` (recurring orchestration). `profile/shadow.py` handles shadow evaluation; `profile/alerts.py` handles alert logic.

**Reporting:** `reporting/` generates static HTML — experiment reports, product dashboards (ticker-first UX), leaderboards with bootstrap CI. Deployed to GitHub Pages via `daily-profile-schedule.yml`.

**CLI wrappers:** `cli/*.py` — each exposes a `main()` function, mapped to console scripts in pyproject.toml.

## Key Conventions

- Feature columns are prefixed with `feat_`; observations are `[feat_* columns, current_position, position_age_norm]`.
- All time-series handling must remain chronological — no shuffling, splits by date order to prevent future leakage.
- Reward = gross return (`target_position * next_return`) minus penalties: trade-cost, risk (`position^2`), switch, downside-volatility, drawdown-spike.
- Default action space: target positions `[-1.0, -0.5, 0.0, 0.5, 1.0]` (customizable via `--position-levels`).
- Data gates in `load_ohlcv_csv`: strict date ordering, duplicate-date rejection, OHLCV numeric checks, outlier return warnings (>20%).
- Symbol aliases: `$BTC → BTC-USD`, `CL → CL=F`, `GC → GC=F`, etc.
- TOML configs support `meta.extends` for inheritance; `config_hardening` generates versioned overlays.
- Diagnostics fields prefixed with `diag_`.
- Tests use monkeypatch to mock PPO models and generate fake OHLCV data for determinism.
- Experiment metadata logged to SQLite (`logs/experiments.db`) by default; `--no-log` disables.

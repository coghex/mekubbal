# Mekubbal

Mekubbal is a learning-focused reinforcement learning project for stock-market experiments in Python.
It helps you run a full research loop: train, evaluate, walk-forward validate, compare variants, sweep penalties, select models, and generate readable reports.

> Not financial advice. This is an educational research project.

## Who this is for

- You want hands-on RL practice (especially PPO-style workflows).
- You care more about good experimentation habits than chasing perfect prediction accuracy.
- You want reproducible, scriptable research runs and easy report generation.

## Quick start (5 minutes)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Download one symbol and run a minimal cycle:

```bash
mekubbal-download --symbol AAPL --start 2018-01-01 --end 2025-01-01 --output data/aapl.csv
mekubbal-train --data data/aapl.csv --model models/aapl_ppo --timesteps 30000
mekubbal-evaluate --data data/aapl.csv --model models/aapl_ppo.zip
mekubbal-paper --data data/aapl.csv --model models/aapl_ppo.zip --output logs/aapl_paper.csv
```

## Core workflow commands

### 1) Reliability first: walk-forward + selection

```bash
mekubbal-walkforward --data data/aapl.csv --models-dir models/walkforward --report logs/walkforward.csv --train-window 252 --test-window 63 --step-window 63 --expanding
mekubbal-select --report logs/walkforward.csv --state models/current_model.json --lookback 3 --min-gap 0.0
```

Regime-aware promotion gate example:

```bash
mekubbal-select --report logs/walkforward.csv --state models/current_model.json --lookback 3 --min-gap 0.0 --min-turbulent-steps 100 --min-turbulent-win-rate 0.5 --min-turbulent-equity-factor 1.0 --max-turbulent-drawdown 0.15
```

### 2) Compare ideas: ablation + sweep

```bash
mekubbal-ablate --data data/aapl.csv --models-dir models/ablation --report logs/ablation_folds.csv --summary logs/ablation_summary.csv --train-window 252 --test-window 63 --step-window 63
mekubbal-sweep --data data/aapl.csv --output-dir logs/sweeps/aapl --report logs/sweeps/aapl/ranking.csv --downside-grid 0,0.005,0.01,0.02 --drawdown-grid 0,0.02,0.05,0.1 --regime-tie-break-tolerance 0.01 --train-window 252 --test-window 63 --step-window 63
```

Sweep ranking is regime-aware: if top deltas are near-equal, turbulent-risk metrics are used as tie-breakers.

### 3) One-shot orchestration + reports

```bash
mekubbal-control --config configs/research-control.toml
mekubbal-report --output logs/reports/aapl.html --walkforward-report logs/walkforward.csv --ablation-summary logs/ablation_summary.csv --sweep-report logs/sweeps/aapl/ranking.csv --selection-state models/current_model.json
```

### 4) Multi-symbol runs

```bash
mekubbal-multi-symbol --base-config configs/research-control.toml --symbols AAPL,MSFT,NVDA --output-root logs/multi_symbol
```

Tabbed dashboard from per-symbol reports:

```bash
mekubbal-report-tabs --output logs/reports/dashboard.html --tab AAPL=logs/reports/aapl.html --tab MSFT=logs/reports/msft.html --tab NVDA=logs/reports/nvda.html
```

### 5) Hardening configs from sweep winners

Single symbol:

```bash
mekubbal-harden-config --base-config configs/research-control.toml --sweep-report logs/sweeps/aapl/ranking.csv --output configs/research-control.hardened.toml --rank 1
```

Multi-symbol hardening in one pass:

```bash
mekubbal-multi-symbol --base-config configs/research-control.toml --symbols AAPL,MSFT,NVDA --output-root logs/multi_symbol --harden-configs --hardened-rank 1 --hardened-profile-template hardened-{symbol_lower}
```

## Useful defaults and conventions

- Action space is target-position based (`[-1, -0.5, 0, 0.5, 1]` by default).
- Features are leakage-safe and chronological.
- Generated experiment logs go to `logs/experiments.db` by default.
- Most CLIs support `--no-log` if you want to disable SQLite run logging.

## Repository map

- `src/mekubbal/`: core modules and CLI entrypoints.
- `configs/`: control and workflow TOML configs.
- `tests/`: unit/integration tests for pipeline modules.

## Development commands

```bash
ruff check .
pytest -q
```

Run targeted tests:

```bash
pytest tests/test_control.py -q
pytest tests/test_multi_symbol.py -q
pytest tests/test_sweep.py -q
pytest tests/test_config_hardening.py -q
```

## Typical project outputs (ignored by git)

- `data/*.csv`
- `models/`
- `logs/`

These are generated artifacts and are intentionally excluded from version control.

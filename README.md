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

Optional lineage tags for manual report builds:

```bash
mekubbal-report --output logs/reports/aapl.html --walkforward-report logs/walkforward.csv --lineage git_commit=$(git rev-parse --short HEAD) --lineage config_profile=hardened-aapl --lineage experiment_run_id=42
```

### 4) Multi-symbol runs

```bash
mekubbal-multi-symbol --base-config configs/research-control.toml --symbols AAPL,MSFT,NVDA --output-root logs/multi_symbol
```

Tabbed dashboard from per-symbol reports:

```bash
mekubbal-report-tabs --output logs/reports/dashboard.html --tab AAPL=logs/reports/aapl.html --tab MSFT=logs/reports/msft.html --tab NVDA=logs/reports/nvda.html
```

Unified dashboard including leaderboards and ticker reports:

```bash
mekubbal-report-tabs --output logs/reports/unified_dashboard.html --leaderboard "Stability=logs/multi_symbol_sector/reports/stability_leaderboard.html" --tab AAPL=logs/reports/aapl.html --tab MSFT=logs/reports/msft.html
```

Generate confidence-aware leaderboard pages (bootstrap CIs + probability of positive gap):

```bash
mekubbal-leaderboards --reports-root logs/multi_symbol_sector/reports --confidence-level 0.95 --bootstrap-samples 2000 --permutation-samples 20000
```

This also generates a `paired_significance_leaderboard` that compares each symbol against the top reference symbol using fold-aligned paired permutation tests.

Compare multiple profile reports for the same symbol with paired significance:

```bash
mekubbal-profile-compare --profile-report base=logs/profiles/base_walkforward.csv --profile-report hardened=logs/profiles/hardened_walkforward.csv --profile-report aggressive=logs/profiles/aggressive_walkforward.csv --output-csv logs/profile_compare/pairwise_significance.csv --output-html logs/profile_compare/pairwise_significance.html
```

Config-driven profile runner (execute profile configs + auto-compare):

```bash
mekubbal-profile-runner --config configs/profile-runner.toml
```

Config-driven profile matrix (multi-symbol x multi-profile + cross-symbol leaderboards):

```bash
mekubbal-profile-matrix --config configs/profile-matrix.toml
```

Auto-select active profile per symbol from matrix outputs (`base` vs `candidate`):

```bash
mekubbal-profile-select --profile-symbol-summary logs/profile_matrix/reports/profile_symbol_summary.csv --state logs/profile_matrix/reports/profile_selection_state.json --base-profile base --candidate-profile candidate --max-candidate-rank 1 --require-candidate-significant
```

Scheduled matrix run with active-profile health snapshot + drift alerts:

```bash
mekubbal-profile-schedule --config configs/profile-schedule.toml
```

Schedule output includes `ticker_health_summary.csv/html` with plain-language per-ticker status and action hints.

Daily GitHub Actions schedule config:

```bash
mekubbal-profile-schedule --config configs/profile-schedule-daily.toml
```

The repository includes `.github/workflows/daily-profile-schedule.yml` to run this daily on GitHub-hosted runners and upload report artifacts.
It now also deploys the latest reports to GitHub Pages (`/logs/profile_matrix_daily/reports/profile_matrix_workspace.html`), with `index.html` redirecting there.

One-time GitHub setup: in **Settings → Pages**, set **Source** to **GitHub Actions**.

Standalone monitoring pass from existing matrix outputs:

```bash
mekubbal-profile-monitor --profile-symbol-summary logs/profile_matrix/reports/profile_symbol_summary.csv --selection-state logs/profile_matrix/reports/profile_selection_state.json
```

This now also supports a plain-language ticker digest (`status`, `recommended_action`, and short summary text) via `--ticker-summary-csv`/`--ticker-summary-html`.

Rollback recommendation/action when drift alerts persist:

```bash
mekubbal-profile-rollback --selection-state logs/profile_matrix/reports/profile_selection_state.json --health-history logs/profile_matrix/reports/active_profile_health_history.csv --min-consecutive-alert-runs 2
```

Threshold sweep for promotion + monitoring rules:

```bash
mekubbal-profile-threshold-sweep --profile-symbol-summary logs/profile_matrix/reports/profile_symbol_summary.csv --health-history logs/profile_matrix/reports/active_profile_health_history.csv --selection-state logs/profile_matrix/reports/profile_selection_state.json
```

The default profile template now uses a distinct candidate config (`configs/research-control.candidate.toml`) so base/candidate comparisons are meaningful out of the box.

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
pytest tests/test_leaderboards.py -q
pytest tests/test_profile_compare.py -q
pytest tests/test_profile_runner.py -q
pytest tests/test_profile_matrix.py -q
pytest tests/test_profile_selection.py -q
pytest tests/test_profile_monitor.py -q
pytest tests/test_profile_schedule.py -q
pytest tests/test_profile_rollback.py -q
pytest tests/test_profile_threshold_sweep.py -q
```

## Typical project outputs (ignored by git)

- `data/*.csv`
- `models/`
- `logs/`

These are generated artifacts and are intentionally excluded from version control.

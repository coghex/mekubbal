# Copilot Instructions for This Repository

## Build, Test, and Lint Commands

Set up environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Primary commands:

- Lint: `ruff check .`
- Run full tests: `pytest -q`
- Run env test file: `pytest tests/test_env.py -q`
- Download data: `mekubbal-download --symbol AAPL --start 2018-01-01 --end 2025-01-01 --output data/aapl.csv`
- Train model: `mekubbal-train --data data/aapl.csv --model models/aapl_ppo --timesteps 30000`
- Evaluate model: `mekubbal-evaluate --data data/aapl.csv --model models/aapl_ppo.zip`
- Paper-trade replay log: `mekubbal-paper --data data/aapl.csv --model models/aapl_ppo.zip --output logs/aapl_paper.csv`
- Periodic retraining: `mekubbal-retrain --data data/aapl.csv --models-dir models/retrain --report logs/retrain.csv --cadence weekly --timesteps 10000 --max-runs 4`
- List experiment runs: `mekubbal-runs --db logs/experiments.db --symbol AAPL --limit 20`
- Diagnostics summary: `mekubbal-diagnostics --mode paper --input logs/aapl_paper.csv`
- Walk-forward validation: `mekubbal-walkforward --data data/aapl.csv --models-dir models/walkforward --report logs/walkforward.csv --train-window 252 --test-window 63 --step-window 63 --expanding`
- Ablation study: `mekubbal-ablate --data data/aapl.csv --models-dir models/ablation --report logs/ablation_folds.csv --summary logs/ablation_summary.csv --train-window 252 --test-window 63 --step-window 63`
- Reward-penalty sweep: `mekubbal-sweep --data data/aapl.csv --output-dir logs/sweeps/aapl --report logs/sweeps/aapl/ranking.csv --downside-grid 0,0.005,0.01,0.02 --drawdown-grid 0,0.02,0.05,0.1 --regime-tie-break-tolerance 0.01`
- Research control workflow: `mekubbal-control --config configs/research-control.toml`
- Multi-symbol control run: `mekubbal-multi-symbol --base-config configs/research-control.toml --symbols AAPL,MSFT,NVDA --output-root logs/multi_symbol`
- Multi-symbol hardened run: `mekubbal-multi-symbol --base-config configs/research-control.toml --symbols AAPL,MSFT,NVDA --output-root logs/multi_symbol --harden-configs --hardened-rank 1 --hardened-profile-template hardened-{symbol_lower}`
- Harden config from sweep: `mekubbal-harden-config --base-config configs/research-control.toml --sweep-report logs/sweeps/aapl/ranking.csv --output configs/research-control.hardened.toml --rank 1`
- Build HTML report: `mekubbal-report --output logs/reports/aapl.html --walkforward-report logs/walkforward.csv --ablation-summary logs/ablation_summary.csv --sweep-report logs/sweeps/aapl/ranking.csv --selection-state models/current_model.json`
- Build HTML report with lineage tags: `mekubbal-report --output logs/reports/aapl.html --walkforward-report logs/walkforward.csv --lineage git_commit=$(git rev-parse --short HEAD) --lineage config_profile=hardened-aapl --lineage experiment_run_id=42`
- Build multi-ticker tabs page: `mekubbal-report-tabs --output logs/reports/dashboard.html --tab AAPL=logs/reports/aapl.html --tab MSFT=logs/reports/msft.html`
- Build unified dashboard (leaderboards + tickers): `mekubbal-report-tabs --output logs/reports/unified_dashboard.html --leaderboard Stability=logs/multi_symbol_sector/reports/stability_leaderboard.html --tab AAPL=logs/reports/aapl.html --tab MSFT=logs/reports/msft.html`
- Generate confidence-aware leaderboards: `mekubbal-leaderboards --reports-root logs/multi_symbol_sector/reports --confidence-level 0.95 --bootstrap-samples 2000 --permutation-samples 20000`
- Compare profiles with paired significance: `mekubbal-profile-compare --profile-report base=logs/profiles/base_walkforward.csv --profile-report hardened=logs/profiles/hardened_walkforward.csv --output-csv logs/profile_compare/pairwise_significance.csv --output-html logs/profile_compare/pairwise_significance.html`
- Config-driven profile runner: `mekubbal-profile-runner --config configs/profile-runner.toml`
- Config-driven profile matrix: `mekubbal-profile-matrix --config configs/profile-matrix.toml`
- Profile auto-selection from matrix outputs: `mekubbal-profile-select --profile-symbol-summary logs/profile_matrix/reports/profile_symbol_summary.csv --state logs/profile_matrix/reports/profile_selection_state.json --base-profile base --candidate-profile candidate --max-candidate-rank 1 --require-candidate-significant`
- Standalone profile monitoring pass: `mekubbal-profile-monitor --profile-symbol-summary logs/profile_matrix/reports/profile_symbol_summary.csv --selection-state logs/profile_matrix/reports/profile_selection_state.json --ticker-summary-csv logs/profile_matrix/reports/ticker_health_summary.csv --ticker-summary-html logs/profile_matrix/reports/ticker_health_summary.html --ensemble-alerts-csv logs/profile_matrix/reports/profile_ensemble_alerts.csv --ensemble-alerts-html logs/profile_matrix/reports/profile_ensemble_alerts.html`
- Profile rollback on persistent alerts (with optional ensemble-event policy): `mekubbal-profile-rollback --selection-state logs/profile_matrix/reports/profile_selection_state.json --health-history logs/profile_matrix/reports/active_profile_health_history.csv --min-consecutive-alert-runs 2`
- Profile threshold sweep: `mekubbal-profile-threshold-sweep --profile-symbol-summary logs/profile_matrix/reports/profile_symbol_summary.csv --health-history logs/profile_matrix/reports/active_profile_health_history.csv --selection-state logs/profile_matrix/reports/profile_selection_state.json`
- Profile v3 ensemble sweep (history-gated recommendation): `mekubbal-profile-ensemble-sweep --profile-symbol-summary logs/profile_matrix/reports/profile_symbol_summary.csv --selection-state logs/profile_matrix/reports/profile_selection_state.json --health-history logs/profile_matrix/reports/active_profile_health_history.csv --output-csv logs/profile_matrix/reports/profile_ensemble_sweep.csv --output-html logs/profile_matrix/reports/profile_ensemble_sweep.html --recommendation-json logs/profile_matrix/reports/profile_ensemble_recommendation.json --min-history-runs 5 --min-history-runs-per-symbol 5`
- Scheduled profile matrix + drift alerts: `mekubbal-profile-schedule --config configs/profile-schedule.toml`
- Optional v3 regime-gated ensemble is configured via `[ensemble_v3]` in profile-schedule configs.
- Schedule runs now also emit a product-facing `reports/product_dashboard.html` view (ticker-first UX with advanced sections for dense reports).
- Daily schedule profile run (GitHub Actions config): `mekubbal-profile-schedule --config configs/profile-schedule-daily.toml`
- GitHub workflow `.github/workflows/daily-profile-schedule.yml` now also deploys reports to GitHub Pages.
- Model selection rule: `mekubbal-select --report logs/walkforward.csv --state models/current_model.json --lookback 3 --min-gap 0.0`
- Regime-aware selection gate example: `mekubbal-select --report logs/walkforward.csv --state models/current_model.json --lookback 3 --min-gap 0.0 --min-turbulent-steps 100 --min-turbulent-win-rate 0.5 --min-turbulent-equity-factor 1.0 --max-turbulent-drawdown 0.15`
- Config-driven initial loop: `mekubbal-loop --config configs/initial-loop.toml`
- Run paper-log test file: `pytest tests/test_paper.py -q`
- Run retraining test file: `pytest tests/test_retrain.py -q`
- Run experiment-log test file: `pytest tests/test_experiment_log.py -q`
- Run walk-forward test file: `pytest tests/test_walk_forward.py -q`
- Run end-to-end smoke test file: `pytest tests/test_smoke_pipeline.py -q`
- Run edge-case test file: `pytest tests/test_edge_cases.py -q`
- Run initial-loop config test file: `pytest tests/test_initial_loop.py -q`
- Run data-gate test file: `pytest tests/test_data_gates.py -q`
- Run selection test file: `pytest tests/test_selection.py -q`
- Run diagnostics test file: `pytest tests/test_diagnostics.py -q`
- Run ablation test file: `pytest tests/test_ablation.py -q`
- Run sweep test file: `pytest tests/test_sweep.py -q`
- Run control test file: `pytest tests/test_control.py -q`
- Run visualization test file: `pytest tests/test_visualization.py -q`
- Run multi-symbol test file: `pytest tests/test_multi_symbol.py -q`
- Run config-hardening test file: `pytest tests/test_config_hardening.py -q`
- Run leaderboards test file: `pytest tests/test_leaderboards.py -q`
- Run profile-compare test file: `pytest tests/test_profile_compare.py -q`
- Run profile-runner test file: `pytest tests/test_profile_runner.py -q`
- Run profile-matrix test file: `pytest tests/test_profile_matrix.py -q`
- Run profile-selection test file: `pytest tests/test_profile_selection.py -q`
- Run profile-monitor test file: `pytest tests/test_profile_monitor.py -q`
- Run profile-schedule test file: `pytest tests/test_profile_schedule.py -q`
- Run profile-rollback test file: `pytest tests/test_profile_rollback.py -q`
- Run profile-threshold-sweep test file: `pytest tests/test_profile_threshold_sweep.py -q`

## High-Level Architecture

Data and training flow:

1. `mekubbal.data`: downloads daily OHLCV data (yfinance) and loads/saves CSV.
2. `mekubbal.features`: converts OHLCV into engineered numeric features and `next_return`, then performs chronological train/test split.
3. `mekubbal.env.TradingEnv`: Gymnasium environment with discrete target-position actions (short to long) and multi-part reward penalties.
4. `mekubbal.train`: trains PPO (`stable-baselines3`) with `MlpPolicy` on the training slice and saves model artifacts.
5. `mekubbal.evaluate`: replays the saved policy on test data and compares against buy-and-hold baseline.
6. `mekubbal.paper`: deterministic policy replay with per-step action/PnL logging and optional append/resume mode.
7. `mekubbal.retrain`: executes weekly/monthly retraining windows with model snapshots and metrics report.
8. `mekubbal.walk_forward`: rolling/expanding walk-forward folds with per-fold model snapshots and report output.
9. `mekubbal.diagnostics`: computes drawdown/turnover/sharpe-like/win-rate diagnostics from episodes and reports.
10. `mekubbal.selection`: promotion rule that advances active model only when recent walk-forward folds beat baseline.
11. `mekubbal.initial_loop`: TOML-driven initial loop automation (optional refresh, train, paper, logging).
12. `mekubbal.reproducibility`: global deterministic seed setup and run-manifest hash generation.
13. `mekubbal.experiment_log`: SQLite-backed experiment run logging and query helpers.
14. `mekubbal.ablation`: walk-forward ablation runner comparing a v1-like control to v2 on aligned folds.
15. `mekubbal.sweep`: grid search over downside/drawdown penalties via repeated ablation runs.
16. `mekubbal.control`: orchestrated control workflow runner that executes walk-forward, ablation, sweep, selection, and report generation from one TOML config.
17. `mekubbal.multi_symbol`: batch runner that applies the control workflow across multiple symbols and writes aggregate summaries.
18. `mekubbal.config_hardening`: creates versioned config overlays from sweep rankings using `meta.extends`.
19. `mekubbal.visualization`: static HTML report builder for key experiment artifacts.
20. `mekubbal.profile_compare`: profile-vs-profile fold-aligned paired significance tooling.
21. `mekubbal.profile_runner`: config-driven orchestration for multiple profiles on one symbol.
22. `mekubbal.profile_matrix`: config-driven orchestration for multiple symbols x profiles with cross-symbol aggregation.
23. `mekubbal.profile_selection`: candidate/base/active profile promotion logic and state persistence.
24. `mekubbal.profile_monitor`: active profile health snapshots and drift/regression alerts.
25. `mekubbal.profile_rollback`: persistent-alert rollback recommendations and optional state mutation.
26. `mekubbal.profile_threshold_sweep`: promotion/monitor threshold grid search with tradeoff ranking.
27. `mekubbal.profile_schedule`: recurring orchestration that runs matrix + monitoring outputs.
28. `mekubbal.cli.*`: command-line wrappers for download/train/evaluate/paper/retrain/walkforward/ablate/sweep/control/multi-symbol/harden-config/report/report-tabs/diagnostics/select/runs/loop/profile workflows.

## Key Conventions in This Codebase

- Feature columns are prefixed with `feat_`; environment observations are `[all feat_* columns, current_position, position_age_norm]`.
- Feature pack v2 uses rolling-normalized momentum/volatility/volume/range features, centered RSI, volatility regime context, and rolling drawdown context.
- Keep time-series handling chronological (no shuffling); train/test splits are by date order to avoid future leakage.
- Reward is defined as: gross return (`target_position * next_return`) minus trade-cost penalty, risk penalty (`position^2` scaled), switch penalty when changing position, downside-volatility penalty, and drawdown-spike penalty.
- Default action semantics in `TradingEnv` map to target positions `[-1.0, -0.5, 0.0, 0.5, 1.0]` (customizable via `--position-levels`).
- CLI commands log run metadata and metrics to `logs/experiments.db` by default (`--no-log` disables).
- Walk-forward folds must remain chronological; each fold trains on an earlier window and tests on the immediately following window.
- Initial loop runs write reproducibility manifests (config/data/model hashes + seed + dependency versions) to `logs/manifests/` by default.
- `load_ohlcv_csv` now enforces data gates (strict date ordering, duplicate-date rejection, OHLCV numeric checks) and warns on large return outliers.
- Model selection keeps previous active model when promotion criteria are not met; regime gate thresholds can require minimum turbulent exposure and performance before promotion.
- Evaluation and paper outputs include diagnostics fields prefixed with `diag_`.

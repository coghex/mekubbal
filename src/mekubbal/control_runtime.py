from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mekubbal.control_config import validate_control_config


def _resolved_window(value: Any, fallback: int | None) -> int | None:
    if value is None:
        return fallback
    return int(value)


def _refresh_data_if_needed(
    *,
    data_cfg: dict[str, Any],
    download_ohlcv_fn: Callable[..., Any],
    save_ohlcv_csv_fn: Callable[..., Any],
) -> str:
    data_path = str(data_cfg["path"])
    if bool(data_cfg["refresh"]):
        downloaded = download_ohlcv_fn(
            symbol=str(data_cfg["symbol"]),
            start=str(data_cfg["start"]),
            end=str(data_cfg["end"]),
        )
        save_ohlcv_csv_fn(downloaded, data_path)
    return data_path


def _build_lineage(
    *,
    config_label: str,
    meta_cfg: dict[str, Any],
    logging_cfg: dict[str, Any],
    git_commit_sha_fn: Callable[[], str | None],
) -> dict[str, Any]:
    lineage = {
        "config_path": str(config_label),
        "config_version": meta_cfg.get("config_version"),
        "config_profile": meta_cfg.get("profile"),
        "symbol": logging_cfg.get("symbol"),
        "git_commit": git_commit_sha_fn(),
    }
    return {key: value for key, value in lineage.items() if value not in (None, "")}


def _run_walkforward_phase(
    *,
    enabled: bool,
    data_path: str,
    walk_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    logging_cfg: dict[str, Any],
    log_db_path: str | None,
    run_walk_forward_validation_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any] | None:
    if not enabled:
        return None
    return run_walk_forward_validation_fn(
        data_path=data_path,
        models_dir=str(walk_cfg["models_dir"]),
        report_path=str(walk_cfg["report_path"]),
        train_window=int(walk_cfg["train_window"]),
        test_window=int(walk_cfg["test_window"]),
        step_window=_resolved_window(walk_cfg.get("step_window"), None),
        expanding=bool(walk_cfg["expanding"]),
        total_timesteps=int(policy_cfg["timesteps"]),
        trade_cost=float(policy_cfg["trade_cost"]),
        risk_penalty=float(policy_cfg["risk_penalty"]),
        switch_penalty=float(policy_cfg["switch_penalty"]),
        position_levels=tuple(float(value) for value in policy_cfg["position_levels"]),
        seed=int(policy_cfg["seed"]),
        symbol=logging_cfg.get("symbol"),
        log_db_path=log_db_path,
    )


def _run_ablation_phase(
    *,
    enabled: bool,
    data_path: str,
    walk_cfg: dict[str, Any],
    ablation_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    logging_cfg: dict[str, Any],
    log_db_path: str | None,
    run_ablation_study_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any] | None:
    if not enabled:
        return None
    return run_ablation_study_fn(
        data_path=data_path,
        models_dir=str(ablation_cfg["models_dir"]),
        report_path=str(ablation_cfg["report_path"]),
        summary_path=str(ablation_cfg["summary_path"]),
        train_window=_resolved_window(ablation_cfg.get("train_window"), int(walk_cfg["train_window"]))
        or int(walk_cfg["train_window"]),
        test_window=_resolved_window(ablation_cfg.get("test_window"), int(walk_cfg["test_window"]))
        or int(walk_cfg["test_window"]),
        step_window=_resolved_window(
            ablation_cfg.get("step_window"),
            _resolved_window(walk_cfg.get("step_window"), None),
        ),
        expanding=(
            bool(walk_cfg["expanding"])
            if ablation_cfg.get("expanding") is None
            else bool(ablation_cfg["expanding"])
        ),
        total_timesteps=int(policy_cfg["timesteps"]),
        trade_cost=float(policy_cfg["trade_cost"]),
        risk_penalty=float(policy_cfg["risk_penalty"]),
        switch_penalty=float(policy_cfg["switch_penalty"]),
        position_levels=tuple(float(value) for value in policy_cfg["position_levels"]),
        seed=int(policy_cfg["seed"]),
        v2_downside_risk_penalty=float(ablation_cfg["v2_downside_penalty"]),
        v2_drawdown_penalty=float(ablation_cfg["v2_drawdown_penalty"]),
        downside_window=int(policy_cfg["downside_window"]),
        symbol=logging_cfg.get("symbol"),
        log_db_path=log_db_path,
    )


def _run_sweep_phase(
    *,
    enabled: bool,
    data_path: str,
    walk_cfg: dict[str, Any],
    sweep_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    logging_cfg: dict[str, Any],
    log_db_path: str | None,
    run_reward_penalty_sweep_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any] | None:
    if not enabled:
        return None
    return run_reward_penalty_sweep_fn(
        data_path=data_path,
        output_dir=str(sweep_cfg["output_dir"]),
        sweep_report_path=str(sweep_cfg["report_path"]),
        downside_penalties=[float(value) for value in sweep_cfg["downside_grid"]],
        drawdown_penalties=[float(value) for value in sweep_cfg["drawdown_grid"]],
        train_window=_resolved_window(sweep_cfg.get("train_window"), int(walk_cfg["train_window"]))
        or int(walk_cfg["train_window"]),
        test_window=_resolved_window(sweep_cfg.get("test_window"), int(walk_cfg["test_window"]))
        or int(walk_cfg["test_window"]),
        step_window=_resolved_window(
            sweep_cfg.get("step_window"),
            _resolved_window(walk_cfg.get("step_window"), None),
        ),
        expanding=(
            bool(walk_cfg["expanding"])
            if sweep_cfg.get("expanding") is None
            else bool(sweep_cfg["expanding"])
        ),
        total_timesteps=int(policy_cfg["timesteps"]),
        trade_cost=float(policy_cfg["trade_cost"]),
        risk_penalty=float(policy_cfg["risk_penalty"]),
        switch_penalty=float(policy_cfg["switch_penalty"]),
        downside_window=int(policy_cfg["downside_window"]),
        position_levels=tuple(float(value) for value in policy_cfg["position_levels"]),
        seed=int(policy_cfg["seed"]),
        symbol=logging_cfg.get("symbol"),
        log_db_path=log_db_path,
        regime_tie_break_tolerance=float(sweep_cfg.get("regime_tie_break_tolerance", 0.01)),
    )


def _run_selection_phase(
    *,
    enabled: bool,
    selection_cfg: dict[str, Any],
    walk_cfg: dict[str, Any],
    run_model_selection_fn: Callable[..., dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    selection_report = (
        str(selection_cfg["report_path"])
        if selection_cfg.get("report_path")
        else str(walk_cfg["report_path"])
    )
    if not enabled:
        return selection_report, None
    selection_metrics = run_model_selection_fn(
        report_path=selection_report,
        state_path=str(selection_cfg["state_path"]),
        lookback_folds=int(selection_cfg["lookback"]),
        min_gap=float(selection_cfg["min_gap"]),
        require_all_recent=not bool(selection_cfg["allow_average_rule"]),
        min_turbulent_step_count=float(selection_cfg["min_turbulent_steps"]),
        min_turbulent_reward_mean=selection_cfg["min_turbulent_reward_mean"],
        min_turbulent_win_rate=selection_cfg["min_turbulent_win_rate"],
        min_turbulent_equity_factor=selection_cfg["min_turbulent_equity_factor"],
        max_turbulent_max_drawdown=selection_cfg["max_turbulent_drawdown"],
    )
    return selection_report, selection_metrics


def _log_control_run(
    *,
    enabled: bool,
    log_db_path: str | None,
    logging_cfg: dict[str, Any],
    data_path: str,
    policy_cfg: dict[str, Any],
    summary: dict[str, Any],
    log_experiment_run_fn: Callable[..., int],
) -> int | None:
    if not enabled or log_db_path is None:
        return None
    return log_experiment_run_fn(
        db_path=log_db_path,
        run_type="control_loop",
        symbol=logging_cfg.get("symbol"),
        data_path=data_path,
        model_path=summary.get("selection", {}).get("active_model_path"),
        timesteps=int(policy_cfg["timesteps"]),
        trade_cost=float(policy_cfg["trade_cost"]),
        metrics=summary,
    )


def _render_control_report(
    *,
    enabled: bool,
    viz_cfg: dict[str, Any],
    walk_cfg: dict[str, Any],
    ablation_cfg: dict[str, Any],
    sweep_cfg: dict[str, Any],
    selection_cfg: dict[str, Any],
    summary: dict[str, Any],
    render_experiment_report_fn: Callable[..., Path | str],
) -> str | None:
    if not enabled:
        return None
    output = render_experiment_report_fn(
        output_path=str(viz_cfg["output_path"]),
        walkforward_report_path=str(walk_cfg["report_path"]) if bool(walk_cfg["enabled"]) else None,
        ablation_summary_path=str(ablation_cfg["summary_path"]) if bool(ablation_cfg["enabled"]) else None,
        sweep_report_path=str(sweep_cfg["report_path"]) if bool(sweep_cfg["enabled"]) else None,
        selection_state_path=str(selection_cfg["state_path"]) if bool(selection_cfg["enabled"]) else None,
        title=str(viz_cfg["title"]),
        lineage=summary.get("lineage"),
    )
    return str(output)


def run_research_control_runtime(
    config: dict[str, Any],
    *,
    config_label: str = "<inline>",
    download_ohlcv_fn: Callable[..., Any],
    save_ohlcv_csv_fn: Callable[..., Any],
    set_global_seed_fn: Callable[[int], Any],
    run_walk_forward_validation_fn: Callable[..., dict[str, Any]],
    run_ablation_study_fn: Callable[..., dict[str, Any]],
    run_reward_penalty_sweep_fn: Callable[..., dict[str, Any]],
    run_model_selection_fn: Callable[..., dict[str, Any]],
    log_experiment_run_fn: Callable[..., int],
    render_experiment_report_fn: Callable[..., Path | str],
    git_commit_sha_fn: Callable[[], str | None],
) -> dict[str, Any]:
    validate_control_config(config)
    data_cfg = config["data"]
    policy_cfg = config["policy"]
    walk_cfg = config["walkforward"]
    ablation_cfg = config["ablation"]
    sweep_cfg = config["sweep"]
    selection_cfg = config["selection"]
    viz_cfg = config["visualization"]
    logging_cfg = config["logging"]
    meta_cfg = config.get("meta", {})
    if meta_cfg is not None and not isinstance(meta_cfg, dict):
        raise ValueError("meta table must be a TOML table.")

    data_path = _refresh_data_if_needed(
        data_cfg=data_cfg,
        download_ohlcv_fn=download_ohlcv_fn,
        save_ohlcv_csv_fn=save_ohlcv_csv_fn,
    )
    set_global_seed_fn(int(policy_cfg["seed"]))

    log_db_path = str(logging_cfg["db_path"]) if bool(logging_cfg["enabled"]) else None
    summary: dict[str, Any] = {
        "config_path": str(config_label),
        "data_path": data_path,
        "lineage": _build_lineage(
            config_label=str(config_label),
            meta_cfg=meta_cfg if isinstance(meta_cfg, dict) else {},
            logging_cfg=logging_cfg,
            git_commit_sha_fn=git_commit_sha_fn,
        ),
    }

    walk_metrics = _run_walkforward_phase(
        enabled=bool(walk_cfg["enabled"]),
        data_path=data_path,
        walk_cfg=walk_cfg,
        policy_cfg=policy_cfg,
        logging_cfg=logging_cfg,
        log_db_path=log_db_path,
        run_walk_forward_validation_fn=run_walk_forward_validation_fn,
    )
    if walk_metrics is not None:
        summary["walkforward"] = walk_metrics

    ablation_metrics = _run_ablation_phase(
        enabled=bool(ablation_cfg["enabled"]),
        data_path=data_path,
        walk_cfg=walk_cfg,
        ablation_cfg=ablation_cfg,
        policy_cfg=policy_cfg,
        logging_cfg=logging_cfg,
        log_db_path=log_db_path,
        run_ablation_study_fn=run_ablation_study_fn,
    )
    if ablation_metrics is not None:
        summary["ablation"] = ablation_metrics

    sweep_metrics = _run_sweep_phase(
        enabled=bool(sweep_cfg["enabled"]),
        data_path=data_path,
        walk_cfg=walk_cfg,
        sweep_cfg=sweep_cfg,
        policy_cfg=policy_cfg,
        logging_cfg=logging_cfg,
        log_db_path=log_db_path,
        run_reward_penalty_sweep_fn=run_reward_penalty_sweep_fn,
    )
    if sweep_metrics is not None:
        summary["sweep"] = sweep_metrics

    _, selection_metrics = _run_selection_phase(
        enabled=bool(selection_cfg["enabled"]),
        selection_cfg=selection_cfg,
        walk_cfg=walk_cfg,
        run_model_selection_fn=run_model_selection_fn,
    )
    if selection_metrics is not None:
        summary["selection"] = selection_metrics

    run_id = _log_control_run(
        enabled=bool(logging_cfg["enabled"]),
        log_db_path=log_db_path,
        logging_cfg=logging_cfg,
        data_path=data_path,
        policy_cfg=policy_cfg,
        summary=summary,
        log_experiment_run_fn=log_experiment_run_fn,
    )
    if run_id is not None:
        summary["experiment_run_id"] = run_id
        summary["lineage"]["experiment_run_id"] = run_id

    visual_report_path = _render_control_report(
        enabled=bool(viz_cfg["enabled"]),
        viz_cfg=viz_cfg,
        walk_cfg=walk_cfg,
        ablation_cfg=ablation_cfg,
        sweep_cfg=sweep_cfg,
        selection_cfg=selection_cfg,
        summary=summary,
        render_experiment_report_fn=render_experiment_report_fn,
    )
    if visual_report_path is not None:
        summary["visual_report_path"] = visual_report_path

    return summary

from __future__ import annotations

from mekubbal.profile.schedule_runtime import prepare_schedule_matrix_context


def test_prepare_schedule_matrix_context_resolves_selection_state_without_shadow(tmp_path):
    matrix_config = tmp_path / "profile-matrix.toml"
    matrix_config.write_text("symbols = [\"AAPL\"]\n", encoding="utf-8")
    output_root = tmp_path / "matrix_out"
    state_path = output_root / "reports" / "profile_selection_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{}", encoding="utf-8")

    context = prepare_schedule_matrix_context(
        schedule_cfg={"matrix_config": matrix_config.name, "symbols": []},
        shadow_cfg={"enabled": False},
        config_dir_path=tmp_path,
        matrix_config_override=None,
        matrix_config_dir=None,
        matrix_config_label=None,
        load_profile_matrix_config_fn=lambda path: {"loaded_from": str(path)},
        run_profile_matrix_fn=lambda path, **kwargs: {
            "output_root": str(output_root),
            "profile_selection": {"state_path": str(state_path)},
            "symbol_summary_path": str(output_root / "reports" / "profile_symbol_summary.csv"),
        },
        run_profile_matrix_config_fn=lambda *args, **kwargs: {},
    )

    assert context["matrix_output_root"] == output_root.resolve()
    assert context["selection_state_path"] == state_path.resolve()
    assert context["shadow_selection_state_path"] is None


def test_prepare_schedule_matrix_context_uses_shadow_promotion_override_and_generated_shadow_state(tmp_path):
    matrix_config = tmp_path / "profile-matrix.toml"
    matrix_config.write_text("symbols = [\"AAPL\"]\n", encoding="utf-8")
    output_root = tmp_path / "matrix_out"
    production_state = output_root / "reports" / "prod_selection_state.json"
    production_state.parent.mkdir(parents=True, exist_ok=True)
    production_state.write_text("{}", encoding="utf-8")
    generated_shadow_state = output_root / "reports" / "generated_shadow_state.json"
    generated_shadow_state.write_text("{}", encoding="utf-8")

    matrix_calls: list[dict[str, object]] = []

    context = prepare_schedule_matrix_context(
        schedule_cfg={"matrix_config": matrix_config.name, "symbols": []},
        shadow_cfg={
            "enabled": True,
            "production_state_path": "reports/prod_selection_state.json",
            "shadow_state_path": "reports/shadow_selection_state.json",
        },
        config_dir_path=tmp_path,
        matrix_config_override=None,
        matrix_config_dir=None,
        matrix_config_label=None,
        load_profile_matrix_config_fn=lambda path: {
            "matrix": {"output_root": str(output_root)},
            "promotion": {"state_path": "reports/profile_selection_state.json"},
        },
        run_profile_matrix_fn=lambda path, **kwargs: matrix_calls.append(kwargs) or {
            "output_root": str(output_root),
            "profile_selection": {"state_path": str(generated_shadow_state)},
            "symbol_summary_path": str(output_root / "reports" / "profile_symbol_summary.csv"),
        },
        run_profile_matrix_config_fn=lambda *args, **kwargs: {},
    )

    assert matrix_calls[0]["promotion_override"] == {
        "enabled": True,
        "state_path": str(output_root / "reports" / "shadow_selection_state.json"),
    }
    assert context["selection_state_path"] == production_state.resolve()
    assert context["shadow_selection_state_path"] == generated_shadow_state.resolve()

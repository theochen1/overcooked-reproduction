import pytest

from human_aware_rl.evaluation import evaluate_paper


def test_strict_mode_never_uses_legacy_find_checkpoint(monkeypatch):
    monkeypatch.setattr(
        evaluate_paper,
        "find_checkpoint_from_run",
        lambda **kwargs: "/tmp/canonical/seed0/ppo_agent/checkpoint_000001",
    )

    def _legacy_scan_forbidden(*args, **kwargs):
        raise AssertionError("legacy find_checkpoint should not be called in strict mode")

    monkeypatch.setattr(evaluate_paper, "find_checkpoint", _legacy_scan_forbidden)
    monkeypatch.setattr(evaluate_paper, "load_jax_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        evaluate_paper,
        "evaluate_agent_pair",
        lambda *args, **kwargs: {"ep_returns": [1.0, 2.0, 3.0]},
    )

    result = evaluate_paper.evaluate_paper_config(
        config_name="sp_sp",
        layout="cramped_room",
        ppo_sp_dir="/tmp/legacy_sp",
        ppo_bc_dir="/tmp/legacy_bc",
        ppo_hp_dir="/tmp/legacy_hp",
        pbt_dir="/tmp/legacy_pbt",
        seed=0,
        strict=True,
        prefer_run_registry=True,
        ppo_data_dir="/tmp/canonical/ppo_runs",
        run_name_templates={"ppo_sp": "ppo_sp__layout-{layout}"},
        agent_dirs={"ppo_sp": "ppo_agent"},
    )

    assert "ep_returns" in result


def test_strict_mode_missing_checkpoint_reports_expected_canonical_path(monkeypatch):
    monkeypatch.setattr(evaluate_paper, "find_checkpoint_from_run", lambda **kwargs: None)

    with pytest.raises(FileNotFoundError) as exc_info:
        evaluate_paper.evaluate_paper_config(
            config_name="sp_sp",
            layout="cramped_room",
            ppo_sp_dir=None,
            ppo_bc_dir=None,
            ppo_hp_dir=None,
            pbt_dir=None,
            seed=0,
            strict=True,
            prefer_run_registry=True,
            ppo_data_dir="/tmp/canonical/ppo_runs",
            run_name_templates={"ppo_sp": "ppo_sp__layout-{layout}"},
            agent_dirs={"ppo_sp": "ppo_agent"},
        )

    msg = str(exc_info.value)
    assert "/tmp/canonical/ppo_runs/ppo_sp__layout-cramped_room/seed0/ppo_agent" in msg

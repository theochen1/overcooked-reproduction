from human_aware_rl_jax_lift.config import get_hparams


def test_ppo_sp_table_lock():
    expected = {
        "simple": (1e-3, 0.1),
        "unident_s": (1e-3, 0.1),
        "random1": (6e-4, 0.1),
        "random0": (8e-4, 0.1),
        "random3": (8e-4, 0.1),
    }
    for layout, (lr, vf) in expected.items():
        h = get_hparams("ppo_sp", layout)
        assert h["learning_rate"] == lr
        assert h["vf_coef"] == vf
    expected_shaping = {
        "simple": int(2.5e6),
        "unident_s": int(2.5e6),
        "random1": int(3.5e6),
        "random0": int(2.5e6),
        "random3": int(2.5e6),
    }
    for layout, rew_h in expected_shaping.items():
        assert get_hparams("ppo_sp", layout)["rew_shaping_horizon"] == rew_h


def test_ppo_bc_table_lock():
    expected = {
        "simple": (1e-3, 3.0, 0.04, int(1e6), (int(5e5), int(3e6))),
        "unident_s": (1e-3, 3.0, 0.05, int(6e6), (int(1e6), int(7e6))),
        "random1": (1e-3, 1.5, 0.05, int(5e6), (int(2e6), int(6e6))),
        "random0": (1.5e-3, 2.0, 0.01, int(4e6), None),
        "random3": (1.5e-3, 3.0, 0.01, int(4e6), (int(1e6), int(4e6))),
    }
    for layout, (lr, ann, vf, rew_h, sp_h) in expected.items():
        h = get_hparams("ppo_bc", layout)
        assert h["learning_rate"] == lr
        assert h["lr_annealing"] == ann
        assert h["vf_coef"] == vf
        assert h["rew_shaping_horizon"] == rew_h
        assert h["self_play_horizon"] == sp_h


def test_pbt_table_lock():
    h = get_hparams("pbt")
    assert h["num_envs"] == 50
    assert h["population_size"] == 3
    assert h["mutation_probability"] == 0.33
    assert h["mutation_factors"] == [0.75, 1.25]
    assert h["iter_per_selection"] == 9
    assert h["num_selection_games"] == 6
from human_aware_rl_jax_lift.reproducibility.paper_hparams import get_hparams


def test_ppo_sp_table_lock():
    expected = {
        "simple": (1e-3, 0.5),
        "unident_s": (1e-3, 0.5),
        "random1": (6e-4, 0.5),
        "random0": (8e-4, 0.5),
        "random3": (8e-4, 0.5),
    }
    for layout, (lr, vf) in expected.items():
        h = get_hparams("ppo_sp", layout)
        assert h["learning_rate"] == lr
        assert h["vf_coef"] == vf


def test_ppo_bc_table_lock():
    expected = {
        "simple": (1e-3, 3.0, 0.5, int(1e6), (int(5e5), int(3e6))),
        "unident_s": (1e-3, 3.0, 0.5, int(6e6), (int(1e6), int(7e6))),
        "random1": (1e-3, 1.5, 0.5, int(5e6), (int(2e6), int(6e6))),
        "random0": (1.5e-3, 2.0, 0.1, int(4e6), None),
        "random3": (1.5e-3, 3.0, 0.1, int(4e6), (int(1e6), int(4e6))),
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

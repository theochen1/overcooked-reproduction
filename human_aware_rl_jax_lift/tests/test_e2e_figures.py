from human_aware_rl_jax_lift.experiments.figure4 import SEEDS as F4_SEEDS
from human_aware_rl_jax_lift.experiments.figure5 import PBT_SEEDS
from human_aware_rl_jax_lift.experiments.figure6 import BC_SEEDS


def test_figure4_seed_set_defined():
    assert len(F4_SEEDS["ppo_sp"]) == 5
    assert len(F4_SEEDS["ppo_bc_train"]) == 5
    assert len(F4_SEEDS["ppo_bc_test"]) == 5
    assert len(F4_SEEDS["ppo_hp"]) == 5


def test_figure5_seed_set_defined():
    assert len(PBT_SEEDS) == 5


def test_figure6_seed_set_defined():
    assert len(BC_SEEDS) == 5

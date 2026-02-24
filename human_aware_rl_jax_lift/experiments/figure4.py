"""Figure 4 reproduction: PPO-SP / PPO-BC / PPO-HP."""

SEEDS = {
    "ppo_sp": [2229, 7649, 7225, 9807, 386],
    "ppo_bc_train": [9456, 1887, 5578, 5987, 516],
    "ppo_bc_test": [2888, 7424, 7360, 4467, 184],
    "ppo_hp": [8355, 5748, 1352, 3325, 8611],
}

PPO_BC_LAYOUT_CONFIGS = {
    "simple": dict(LR=1e-3, VF_COEF=0.5, MINIBATCHES=10, LR_ANNEALING=3, REW_SHAPING_HORIZON=1e6, TOT=8e6, SELF_PLAY_HORIZON=(5e5, 3e6)),
    "unident_s": dict(LR=1e-3, VF_COEF=0.5, MINIBATCHES=12, LR_ANNEALING=3, REW_SHAPING_HORIZON=6e6, TOT=1e7, SELF_PLAY_HORIZON=(1e6, 7e6)),
    "random0": dict(LR=1.5e-3, VF_COEF=0.1, MINIBATCHES=15, LR_ANNEALING=2, REW_SHAPING_HORIZON=4e6, TOT=9e6, SELF_PLAY_HORIZON=None),
    "random1": dict(LR=1e-3, VF_COEF=0.5, MINIBATCHES=15, LR_ANNEALING=1.5, REW_SHAPING_HORIZON=5e6, TOT=1.6e7, SELF_PLAY_HORIZON=(2e6, 6e6)),
    "random3": dict(LR=1.5e-3, VF_COEF=0.1, MINIBATCHES=15, LR_ANNEALING=3, REW_SHAPING_HORIZON=4e6, TOT=1.2e7, SELF_PLAY_HORIZON=(1e6, 4e6)),
}


def run():
    return {
        "seeds": SEEDS,
        "layout_configs": PPO_BC_LAYOUT_CONFIGS,
    }


if __name__ == "__main__":
    run()

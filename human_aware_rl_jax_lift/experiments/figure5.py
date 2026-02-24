"""Figure 5 reproduction: PBT."""

PBT_SEEDS = [8015, 3554, 581, 5608, 4221]
PBT_CONFIG = dict(
    population_size=3,
    resample_prob=0.33,
    mutation_factors=[0.75, 1.25],
    mutable=["LAM", "CLIPPING", "LR", "STEPS_PER_UPDATE", "ENTROPY", "VF_COEF"],
    iter_per_selection=9,
    num_selection_games=6,
)


def run():
    return {"seeds": PBT_SEEDS, "config": PBT_CONFIG}


if __name__ == "__main__":
    run()

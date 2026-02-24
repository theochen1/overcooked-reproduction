"""Figure 6 reproduction: BC quality ablations."""

BC_SEEDS = [5415, 2652, 6440, 1965, 6647]
BC_EPOCHS_BY_LAYOUT = {
    "simple": 100,
    "unident_s": 120,
    "random1": 120,
    "random0": 90,
    "random3": 110,
}


def run():
    return {"seeds": BC_SEEDS, "epochs_by_layout": BC_EPOCHS_BY_LAYOUT}


if __name__ == "__main__":
    run()

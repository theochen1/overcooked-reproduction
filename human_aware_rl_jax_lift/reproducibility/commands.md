# Exact Command Lines

## BC (train split)

`python -m human_aware_rl_jax_lift.scripts.train_bc --layout simple --data_path data/human/anonymized/clean_train_trials.pkl --split train --seeds 5415 2652 6440 1965 6647 --epochs 100`

## BC (test split)

`python -m human_aware_rl_jax_lift.scripts.train_bc --layout simple --data_path data/human/anonymized/clean_test_trials.pkl --split test --seeds 5415 2652 6440 1965 6647 --epochs 100`

## PPO-SP

`python -m human_aware_rl_jax_lift.scripts.train_ppo_sp --layout simple --seeds 2229 7649 7225 9807 386`

## PPO-BC (train partner)

`python -m human_aware_rl_jax_lift.scripts.train_ppo_bc --layout simple --bc_split train --seeds 9456 1887 5578 5987 516`

## PPO-BC (test partner)

`python -m human_aware_rl_jax_lift.scripts.train_ppo_bc --layout simple --bc_split test --seeds 2888 7424 7360 4467 184`

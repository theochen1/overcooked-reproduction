# Exact Command Lines

Run from `human_aware_rl_jax_lift` (module root). BC data must exist: either pre-featurized `data/bc_data/{layout}.pkl` (from `prepare_bc_data.py`) or raw `data/human/anonymized/clean_{train,test}_trials.pkl`.

## BC (train split)

`python scripts/train_bc.py --layout simple --data_path data/bc_data/simple.pkl --split train --seeds 5415 2652 6440 1965 6647 --epochs 100`

## BC (test split)

`python scripts/train_bc.py --layout simple --data_path data/bc_data/simple.pkl --split test --seeds 5415 2652 6440 1965 6647 --epochs 100`

## PPO-SP

`python scripts/train_ppo_sp.py --layout simple --seeds 2229 7649 7225 9807 386`

## PPO-BC (train partner)

`python scripts/train_ppo_bc.py --layout simple --bc_split train --seeds 9456 1887 5578 5987 516`

## PPO-BC (test partner)

`python scripts/train_ppo_bc.py --layout simple --bc_split test --seeds 2888 7424 7360 4467 184`

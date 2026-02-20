![MDP python tests](https://github.com/HumanCompatibleAI/overcooked_ai/workflows/.github/workflows/pythontests.yml/badge.svg) ![overcooked-ai codecov](https://codecov.io/gh/HumanCompatibleAI/overcooked_ai/branch/master/graph/badge.svg) [![PyPI version](https://badge.fury.io/py/overcooked-ai.svg)](https://badge.fury.io/py/overcooked-ai) [!["Open Issues"](https://img.shields.io/github/issues-raw/HumanCompatibleAI/overcooked_ai.svg)](https://github.com/HumanCompatibleAI/minerl/overcooked_ai) [![GitHub issues by-label](https://img.shields.io/github/issues-raw/HumanCompatibleAI/overcooked_ai/bug.svg?color=red)](https://github.com/HumanCompatibleAI/overcooked_ai/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+label%3Abug) [![Downloads](https://pepy.tech/badge/overcooked-ai)](https://pepy.tech/project/overcooked-ai)
[![arXiv](https://img.shields.io/badge/arXiv-1910.05789-bbbbbb.svg)](https://arxiv.org/abs/1910.05789)

# Overcooked-AI 🧑‍🍳🤖

<p align="center">
  <!-- <img src="overcooked_ai_js/images/screenshot.png" width="350"> -->
  <img src="./images/layouts.gif" width="100%"> 
  <i>5 of the available layouts. New layouts are easy to hardcode or generate programmatically.</i>
</p>

## Introduction 🥘

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires placing up to 3 ingredients in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

You can **try out the game [here](https://humancompatibleai.github.io/overcooked-demo/)** (playing with some previously trained DRL agents). To play with your own trained agents using this interface, or to collect more human-AI or human-human data, you can use the code [here](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/overcooked_demo). You can find some human-human and human-AI gameplay data already collected [here](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/human_aware_rl/static/human_data).

**Training Agents:** The `human_aware_rl` directory contains code for training Behavior Cloning (BC) and PPO agents. See the [Behavior Cloning and Reinforcement Learning](#behavior-cloning-and-reinforcement-learning-) section below for details.

This benchmark was build in the context of a 2019 paper: *[On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789)*. Also see our [blog post](https://bair.berkeley.edu/blog/2019/10/21/coordination/).

## Research Papers using Overcooked-AI 📑


- Carroll, Micah, Rohin Shah, Mark K. Ho, Thomas L. Griffiths, Sanjit A. Seshia, Pieter Abbeel, and Anca Dragan. ["On the utility of learning about humans for human-ai coordination."](https://arxiv.org/abs/1910.05789) NeurIPS 2019.
- Charakorn, Rujikorn, Poramate Manoonpong, and Nat Dilokthanakul. ["Investigating Partner Diversification Methods in Cooperative Multi-Agent Deep Reinforcement Learning."](https://www.rujikorn.com/files/papers/diversity_ICONIP2020.pdf) Neural Information Processing. ICONIP 2020.
- Knott, Paul, Micah Carroll, Sam Devlin, Kamil Ciosek, Katja Hofmann, Anca D. Dragan, and Rohin Shah. ["Evaluating the Robustness of Collaborative Agents."](https://arxiv.org/abs/2101.05507) AAMAS 2021.
- Nalepka, Patrick, Jordan P. Gregory-Dunsmore, James Simpson, Gaurav Patil, and Michael J. Richardson. ["Interaction Flexibility in Artificial Agents Teaming with Humans."](https://www.researchgate.net/publication/351533529_Interaction_Flexibility_in_Artificial_Agents_Teaming_with_Humans) Cogsci 2021.
- Fontaine, Matthew C., Ya-Chuan Hsu, Yulun Zhang, Bryon Tjanaka, and Stefanos Nikolaidis. ["On the Importance of Environments in Human-Robot Coordination"](http://arxiv.org/abs/2106.10853) RSS 2021.
- Zhao, Rui, Jinming Song, Hu Haifeng, Yang Gao, Yi Wu, Zhongqian Sun, Yang Wei. ["Maximum Entropy Population Based Training for Zero-Shot Human-AI Coordination"](https://arxiv.org/abs/2112.11701). NeurIPS Cooperative AI Workshop, 2021.
- Sarkar, Bidipta, Aditi Talati, Andy Shih, and Dorsa Sadigh. ["PantheonRL: A MARL Library for Dynamic Training Interactions"](https://iliad.stanford.edu/pdfs/publications/sarkar2022pantheonrl.pdf). AAAI 2022.
- Ribeiro, João G., Cassandro Martinho, Alberto Sardinha, Francisco S. Melo. ["Assisting Unknown Teammates in Unknown Tasks: Ad Hoc Teamwork under Partial Observability"](https://arxiv.org/abs/2201.03538).
- Xihuai Wang, Shao Zhang, Wenhao Zhang, Wentao Dong, Jingxiao Chen, Ying Wen and Weinan Zhang. NeurIPS 2024. ["ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination"](https://arxiv.org/abs/2310.05208v2).


## Installation ☑️

### Installing from PyPI 🗜

You can install the pre-compiled wheel file using pip.
```
pip install overcooked-ai
```
Note that PyPI releases are stable but infrequent. For the most up-to-date development features, build from source. We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the package, so that you can use the provided lockfile to ensure no minimal package version issues.


### Building from source 🔧

Clone the repo 
```
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
```

Using uv (recommended):
```
uv venv
uv sync
```


### Verifying Installation 📈

When building from source, you can verify the installation by running the Overcooked unit test suite. The following commands should all be run from the `overcooked_ai` project root directory:

```
python testing/overcooked_test.py
```




If you're thinking of using the planning code extensively, you should run the full testing suite that verifies all of the Overcooked accessory tools (this can take 5-10 mins): 
```
python -m unittest discover -s testing/ -p "*_test.py"
```

See this [notebook](Overcooked%20Tutorial.ipynb) for a quick guide on getting started using the environment.

## Code Structure Overview 🗺

`overcooked_ai_py` contains:

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically

`agents/`:
- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning/`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic

`overcooked_demo` contains:

`server/`:
- `app.py`: The Flask app 
- `game.py`: The main logic of the game. State transitions are handled by overcooked.Gridworld object embedded in the game environment
- `move_agents.py`: A script that simplifies copying checkpoints to [agents](src/overcooked_demo/server/static/assets/agents/) directory. Instruction of how to use can be found inside the file or by running `python move_agents.py -h`

`up.sh`: Shell script to spin up the Docker server that hosts the game 

`human_aware_rl` contains:

`imitation/`:
- `behavior_cloning.py`: PyTorch module for training, saving, and loading BC models
- `bc_agent.py`: Agent wrapper for using trained BC models in the environment
- `train_bc_models.py`: Script to train BC models on all layouts

`ppo/`:
- `ppo_client.py`: Driver code for training PPO agents with JAX/Flax
- `train_ppo_sp.py`: Script for training self-play PPO agents
- `train_ppo_bc.py`: Script for training PPO with BC partner
- `configs/paper_configs.py`: Hyperparameters from the original paper

`jaxmarl/`:
- `ppo.py`: JAX/Flax implementation of PPO training
- `overcooked_env.py`: JAX-compatible environment wrapper

`visualization/`:
- `play_game.py`: Watch trained agents play with pygame or save as GIF

`human/`:
- `process_data.py`: Script to process human data for DRL algorithms
- `data_processing_utils.py`: Utils for the above

`utils.py`: General utilities


## Raw Data :ledger:

The raw data used during BC training is >100 MB, which makes it inconvenient to distribute via git. The code uses pickled dataframes for training and testing, but in case one needs to original data it can be found [here](https://drive.google.com/drive/folders/1aGV8eqWeOG5BMFdUcVoP2NHU_GFPqi57?usp=share_link) 

## Behavior Cloning and Reinforcement Learning 🤖

The `human_aware_rl` module provides PyTorch-based Behavior Cloning and JAX-based PPO training.

### Installation

Install the ML dependencies:

```bash
# For Behavior Cloning (PyTorch)
pip install -e ".[bc]"

# For full RL training (JAX + PyTorch)
pip install -e ".[harl]"

# With CUDA support
pip install -e ".[harl-cuda]"
```

### Training Behavior Cloning Models

Train BC models on human demonstration data:

```bash
cd src/human_aware_rl

# Train BC models for all 5 layouts
python -m human_aware_rl.imitation.train_bc_models --all_layouts

# Train for a specific layout
python -m human_aware_rl.imitation.train_bc_models --layout cramped_room
```

This trains two models per layout:
- **BC model** (from training data): Used as a partner during PPO training
- **Human Proxy model** (from test data): Used for evaluation

Models are saved to `src/human_aware_rl/bc_runs/`.

### Training PPO Agents

There are two training modes: **fast** (for quick iteration) and **full** (for paper reproduction).

#### Fast Training (Recommended for Testing)

Fast training uses 1M timesteps with early stopping, completing in ~15-30 minutes per layout:

```bash
cd src/human_aware_rl

# Fast self-play training
python -m human_aware_rl.ppo.train_ppo_sp --layout cramped_room --seed 0 --fast

# Fast PPO_BC training (requires BC models first)
python -m human_aware_rl.ppo.train_ppo_bc --layout cramped_room --seed 0 --fast

# Custom timesteps
python -m human_aware_rl.ppo.train_ppo_sp --layout cramped_room --timesteps 500000
```

#### Full Paper Reproduction

Full training uses the paper's hyperparameters (6-8M timesteps), taking 2-4 hours per layout:

```bash
cd src/human_aware_rl

# Self-play PPO (all layouts, 5 seeds each)
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40

# PPO with BC partner
python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0,10,20,30,40
```

#### Training Mode Comparison

| Setting | Fast (`--fast`) | Full (paper) |
|---------|-----------------|--------------|
| Timesteps | 1M | 6.6-7.8M |
| Time per layout | ~15-30 min | ~2-4 hours |
| Early stopping | Yes (30 updates) | Yes (50 updates) |
| Parallel envs | 32 | 32 |
| Expected reward | ~70-80% of full | 100% |

The fast mode is useful for:
- Verifying the training pipeline works
- Quick experimentation with hyperparameters
- Debugging visualization and evaluation code

Use full mode when you need to reproduce the paper's exact results.

### Visualizing Trained Agents

Watch trained agents play the game:

```bash
cd src/human_aware_rl

# Save as GIF and auto-open (recommended)
python -m human_aware_rl.visualization.play_game --bc_self_play --layout cramped_room --gif

# Watch in real-time pygame window
python -m human_aware_rl.visualization.play_game --bc_self_play --layout cramped_room

# Quick demo with random agents (no training required)
python -m human_aware_rl.visualization.play_game --random --layout cramped_room --gif
```

#### Agent Pairings for Evaluation

The paper evaluates agents by pairing them with a **Human Proxy (HP)** - a BC model trained on held-out test data:

```bash
# BC vs Human Proxy
python -m human_aware_rl.visualization.play_game \
    --agent0_type bc --agent0_path bc_runs/train/cramped_room \
    --agent1_type bc --agent1_path bc_runs/test/cramped_room \
    --layout cramped_room --gif

# PPO_SP vs Human Proxy (main paper evaluation)
python -m human_aware_rl.visualization.play_game \
    --agent0_type ppo --agent0_path results/ppo_sp/ppo_sp_cramped_room_seed0/checkpoint_*/params.pkl \
    --agent1_type bc --agent1_path bc_runs/test/cramped_room \
    --layout cramped_room --gif
```

| Model Path | Description |
|------------|-------------|
| `bc_runs/train/{layout}` | BC trained on human training data (PPO_BC partner) |
| `bc_runs/test/{layout}` | Human Proxy for evaluation (held-out test data) |

Available layouts: `cramped_room`, `asymmetric_advantages`, `coordination_ring`, `forced_coordination`, `counter_circuit`

### Reproducing Paper Results (Figure 4)

The evaluation pipeline reproduces Figure 4 from the paper, comparing different agent training methods:

#### Complete Training Pipeline

```bash
cd src/human_aware_rl

# Step 1: Train BC and Human Proxy models
python -m human_aware_rl.imitation.train_bc_models --all_layouts

# Step 2: Train Self-Play PPO agents
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40

# Step 3: Train PPO with BC partner
python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0,10,20,30,40

# Step 4: Train Gold Standard PPO with Human Proxy (optional, for red line in Figure 4)
python -m human_aware_rl.ppo.train_ppo_hp --all_layouts --seeds 0,10,20,30,40

# Step 5: Train PBT agents (for Figure 4b)
python -m human_aware_rl.ppo.train_pbt --all_layouts
```

#### Running Evaluation

```bash
# Check which models are available
python -m human_aware_rl.evaluation.run_paper_evaluation --check_only

# Run full evaluation and generate Figure 4
python -m human_aware_rl.evaluation.run_paper_evaluation --all

# Or evaluate specific figures
python -m human_aware_rl.evaluation.run_paper_evaluation --figure 4a  # Self-play comparison
python -m human_aware_rl.evaluation.run_paper_evaluation --figure 4b  # PBT comparison
```

#### Figure 4 Evaluation Configs

**Figure 4(a) - Self-Play Comparison:**
| Config | Description | Color |
|--------|-------------|-------|
| PPO_HP + HP | Gold standard (PPO trained with HP) | Red dotted line |
| SP + SP | Self-play baseline | White bars |
| SP + HP | Self-play agent + Human Proxy | Teal bars |
| PPO_BC + HP | BC-trained PPO + Human Proxy | Orange bars |
| BC + HP | BC agent + Human Proxy | Gray bars |

**Figure 4(b) - PBT Comparison:**
- Same structure but with PBT agents instead of Self-Play

#### Quick Evaluation (Fewer Seeds)

```bash
# Fast evaluation with single seed
python -m human_aware_rl.evaluation.run_paper_evaluation --all --seeds 0 --num_games 5
```

### Running Tests

```bash
cd src/human_aware_rl
./run_tests.sh
```

## Further Issues and questions ❓

If you have issues or questions, you can contact [Micah Carroll](https://micahcarroll.github.io) at mdc@berkeley.edu.

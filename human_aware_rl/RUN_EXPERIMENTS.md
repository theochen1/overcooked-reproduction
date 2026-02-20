# Running Overcooked Human-Aware RL Experiments

This guide explains how to reproduce the results from the paper "On the Utility of Learning about Humans for Human-AI Coordination" (NeurIPS 2019).

## Quick Start

### 1. Deploy to EC2

Run the deployment script from your local machine:

```bash
cd /Users/theochen/Desktop/overcooked-reproduction/human_aware_rl
chmod +x deploy_to_ec2.sh
./deploy_to_ec2.sh
```

### 2. SSH into EC2

```bash
ssh -i "/Users/theochen/Downloads/overcooked-dpail.pem" ec2-user@98.86.229.193
```

### 3. Build Docker Image

```bash
cd ~/overcooked/human_aware_rl
sudo docker build -t overcooked-rl -f Dockerfile.ec2 .
```

This will take 10-15 minutes to build.

### 4. Run Container

**Interactive mode (recommended for testing):**
```bash
sudo docker run -it --name overcooked overcooked-rl
```

**Background mode (for long experiments):**
```bash
sudo docker run -d --name overcooked-bg overcooked-rl tail -f /dev/null
sudo docker exec -it overcooked-bg bash
```

## Experiment Overview

The paper has several types of experiments. Run them in this order for best results:

### Phase 1: Behavioral Cloning (BC) - Required First!

BC models are needed as training partners for PPO experiments.

```bash
# Inside the Docker container
cd /workspace/human_aware_rl/human_aware_rl

# Run BC experiments (trains BC models from human data)
python experiments/bc_experiments.py
```

**Estimated time:** 1-2 hours

### Phase 2: PPO Self-Play (PPO-SP)

Trains agents using self-play (agent plays against copies of itself).

```bash
bash experiments/ppo_sp_experiments.sh
```

**Estimated time:** 6-12 hours per layout (5 layouts × 5 seeds each)

### Phase 3: Population Based Training (PBT)

Trains a population of agents that play against each other.

```bash
bash experiments/pbt_experiments.sh
```

**Estimated time:** 8-15 hours per layout

### Phase 4: PPO with BC Partner (PPO-BC)

Trains agents to play with BC models of humans.

```bash
bash experiments/ppo_bc_experiments.sh
```

**Estimated time:** 8-12 hours per layout

### Phase 5: PPO with Human Model (PPO-HM)

Trains agents to play with a hand-crafted human model.

```bash
bash experiments/ppo_hm_experiments.sh
```

**Estimated time:** 8-12 hours per layout

## Running Individual Experiments

Each experiment script contains multiple commands. You can run individual ones:

### PPO Self-Play Example (Simple Layout)

```bash
python ppo/ppo.py with \
    EX_NAME="ppo_sp_simple" \
    layout_name="simple" \
    REW_SHAPING_HORIZON=2.5e6 \
    LR=1e-3 \
    PPO_RUN_TOT_TIMESTEPS=6e6 \
    OTHER_AGENT_TYPE="sp" \
    SEEDS="[2229]" \
    VF_COEF=1 \
    TIMESTAMP_DIR=False
```

### PBT Example (Simple Layout)

```bash
python pbt/pbt.py with \
    fixed_mdp \
    layout_name="simple" \
    EX_NAME="pbt_simple" \
    TOTAL_STEPS_PER_AGENT=8e6 \
    REW_SHAPING_HORIZON=3e6 \
    LR=2e-3 \
    POPULATION_SIZE=3 \
    SEEDS="[8015]" \
    NUM_SELECTION_GAMES=6 \
    VF_COEF=0.5 \
    MINIBATCHES=10 \
    TIMESTAMP_DIR=False
```

## Layouts

The paper uses 5 layouts:
- `simple` - Basic layout for initial testing
- `unident_s` - Unidentified layout variant
- `random0`, `random1`, `random3` - Procedurally generated layouts

## Monitoring Experiments

### View logs in real-time

```bash
# If running in background
sudo docker logs -f overcooked-bg
```

### Check GPU usage (if using GPU instance)

```bash
nvidia-smi
```

### Screen for persistent sessions

```bash
# Start a screen session
screen -S experiments

# Run your experiment
bash experiments/ppo_sp_experiments.sh

# Detach: Ctrl+A, then D
# Reattach later
screen -r experiments
```

## Saving Results

Results are saved in the container. To copy them out:

```bash
# From EC2 host (not inside container)
sudo docker cp overcooked:/workspace/human_aware_rl/human_aware_rl/data ./experiment_results
```

Then copy to your local machine:

```bash
# From your local machine
scp -i "/Users/theochen/Downloads/overcooked-dpail.pem" -r \
    ec2-user@98.86.229.193:~/experiment_results ./
```

## Troubleshooting

### Container already exists

```bash
sudo docker rm overcooked
```

### Out of disk space

```bash
# Clean up Docker
sudo docker system prune -a
```

### TensorFlow errors

Ensure you're using TensorFlow 1.13.1:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Memory issues

For large experiments, consider using a larger EC2 instance (t2.xlarge or bigger).

## Experiment Parameters Reference

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `PPO_RUN_TOT_TIMESTEPS` | Total training steps | 6e6 - 16e6 |
| `REW_SHAPING_HORIZON` | When to stop reward shaping | 1e6 - 7e6 |
| `LR` | Learning rate | 6e-4 - 3e-3 |
| `SEEDS` | Random seeds for reproducibility | List of 5 seeds |
| `OTHER_AGENT_TYPE` | Partner type | sp, bc_train, bc_test, hm |
| `VF_COEF` | Value function coefficient | 0.1 - 1.0 |

## Full Reproduction Timeline

For complete paper reproduction on a single machine:

1. **Day 1**: BC experiments + 1 PPO-SP layout
2. **Days 2-3**: Remaining PPO-SP layouts
3. **Days 4-6**: PBT experiments
4. **Days 7-9**: PPO-BC experiments
5. **Days 10-11**: PPO-HM experiments
6. **Day 12**: Analysis and visualization (run Jupyter notebook)

Total: ~2 weeks for full reproduction on a single GPU instance.


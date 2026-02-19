# Critical Differences: JAX Implementation vs Original TensorFlow Implementation

This document outlines the key differences between the original TensorFlow implementation in `human_aware_rl/` and the JAX implementation in `overcooked_ai-master/src/human_aware_rl/jaxmarl/`.

## 1. OBSERVATION ENCODING (MAJOR DIFFERENCE)

### Old Implementation (`human_aware_rl/overcooked_ai`)
- **Observation shape: (5, 5, 20) channels**
- Features:
  - 10 player features: 2 positions + 8 orientations (4 directions × 2 players)
  - 5 base map features: `pot_loc`, `counter_loc`, `onion_disp_loc`, `dish_disp_loc`, `serve_loc`
  - 5 variable features: `onions_in_pot`, `onions_cook_time`, `onion_soup_loc`, `dishes`, `onions`
- **NO tomato support** - only onion soups
- **NO urgency feature** - no time pressure signal
- Uses simple `cook_time` tracking

### New Implementation (`overcooked_ai-master/src/overcooked_ai_py`)  
- **Observation shape: (5, 5, 26) channels**
- Features:
  - 10 player features (same)
  - 6 base map features: adds `tomato_disp_loc`
  - 9 variable features: `onions_in_pot`, `tomatoes_in_pot`, `onions_in_soup`, `tomatoes_in_soup`, `soup_cook_time_remaining`, `soup_done`, `dishes`, `onions`, `tomatoes`
  - 1 urgency feature: `urgency` (activates when `horizon - timestep < 40`)
- **HAS tomato support** - mixed soups possible
- **HAS urgency feature** - signals end of episode

**IMPACT**: The neural network expects different input shapes. A model trained on 20 channels cannot work with 26 channels, and the feature meanings differ.

## 2. LAYOUT CONFIGURATION DIFFERENCES

### Old Layouts (`human_aware_rl`)
```json
{
    "grid": "...",
    "start_order_list": None,
    "cook_time": 20,
    "num_items_for_soup": 3,
    "delivery_reward": 20
}
```

### New Layouts (`overcooked_ai-master`)
```json
{
    "grid": "...",
    "start_bonus_orders": [],
    "start_all_orders": [
        {"ingredients": ["onion", "onion", "onion"]}
    ]
}
```

**IMPACT**: The layout configuration schema has changed. The new version uses a recipe-based order system.

## 3. `old_dynamics` FLAG

### Old Implementation
- **No `old_dynamics` flag** - there is only one dynamics mode
- Soups automatically start cooking when 3 ingredients are added

### New Implementation
- **Has `old_dynamics` flag** to enable backward compatibility
- When `old_dynamics=True`: auto-cook when 3 ingredients added (like old behavior)
- When `old_dynamics=False`: requires explicit "interact" action to start cooking
- **Problem**: The new code has assertions that fail with `old_dynamics=True` on some layouts:
  ```python
  assert all([len(order["ingredients"]) == 3 for order in self.start_all_orders]), 
         "Only accept orders with 3 items when using the old_dynamics"
  ```

## 4. ADVANTAGE NORMALIZATION (SUBTLE DIFFERENCE)

### Old Implementation (baselines PPO)
```python
# In model.py train() - normalizes PER MINIBATCH
advs = returns - values
advs = (advs - advs.mean()) / (advs.std() + 1e-8)
```
The advantages are normalized **per minibatch** inside the `train()` function.

### JAX Implementation
```python
# Normalized at BATCH level before minibatch splitting
advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
# Then split into minibatches
```
The comment says "CRITICAL FIX" - but this behavior is actually DIFFERENT from original.

**IMPACT**: Per-minibatch normalization vs per-batch normalization can lead to different gradient statistics.

## 5. VALUE FUNCTION LOSS

### Old Implementation
```python
vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, -CLIPRANGE, CLIPRANGE)
vf_losses1 = tf.square(vpred - R)
vf_losses2 = tf.square(vpredclipped - R)
vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
```
Uses value function clipping with `0.5` coefficient.

### JAX Implementation
```python
if clip_vf:
    values_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    critic_loss1 = (values - returns) ** 2
    critic_loss2 = (values_clipped - returns) ** 2
    critic_loss = 0.5 * jnp.maximum(critic_loss1, critic_loss2).mean()
else:
    critic_loss = 0.5 * ((values - returns) ** 2).mean()
```
Similar, but `clip_vf` is configurable (could be disabled).

## 6. ENTROPY COEFFICIENT

### Old Implementation
- Uses fixed entropy coefficient throughout training
- Default: `ent_coef=0.1` (from ppo.py config)

### JAX Implementation  
- Uses **annealed** entropy coefficient
- `entropy_coeff_start=0.2 → entropy_coeff_end=0.1` over `entropy_coeff_horizon=3e5` steps
- This is controlled by `use_entropy_annealing` flag

**IMPACT**: Different exploration behavior during training.

## 7. LEARNING RATE SCHEDULE

### Old Implementation
```python
# Linear annealing from lr to lr/lr_reduction_factor
lr = lambda prop: (start_lr / lr_reduction_factor) + (start_lr - start_lr/lr_reduction_factor) * prop
```
Where `prop = 1.0 - (update - 1.0) / nupdates` (starts at 1, ends at 0)

### JAX Implementation
```python
frac = 1.0 - self.total_timesteps / self.config.total_timesteps
new_lr = self.config.learning_rate * frac
```
Linear decay from `lr` to `0`.

## 8. SELF-PLAY MECHANISM

### Old Implementation
- Uses trajectory-level self-play randomization
- `gym_env.self_play_randomization` controls probability
- Other agent can be: BC, Human Model, Random, or Self-Play copy

### JAX Implementation
- Uses `bc_schedule` for mixing BC and self-play
- Self-play actions sampled from same policy for agent 1
- No trajectory-level randomization option

## 9. REWARD SHAPING (CRITICAL DIFFERENCE!)

### Old Implementation
- Uses `shaped_r` from environment step - **SINGLE SCALAR** for both agents
- **BOTH agents receive the SAME total shaped reward**, regardless of which agent performed the action
- This is key for cooperative learning: if agent 1 places an item in pot, agent 0 ALSO gets that reward
```python
# In RewardShapingEnv.step_wait()
dense_reward = infos[env_num]['shaped_r']  # Single scalar - sum of all shaped rewards
shaped_rew = rew[env_num] + float(dense_reward) * self.reward_shaping_factor
```

### JAX Implementation (BEFORE FIX)
- Used `shaped_r_by_agent` - **PER-AGENT** shaped rewards
- Each agent only received shaped reward for their OWN actions
- This broke cooperative learning: agents couldn't learn to coordinate because they didn't get credit for partner's actions
```python
dense_reward = info.get("shaped_r_by_agent", (0, 0))  # Per-agent!
shaped_reward_0 = sparse_reward + factor * dense_reward[0]  # Only agent 0's rewards
shaped_reward_1 = sparse_reward + factor * dense_reward[1]  # Only agent 1's rewards
```

**IMPACT**: This is one of the MOST CRITICAL differences! Using per-agent rewards prevents cooperative learning because agents don't get credit for their partner's contributions to the team goal.

### JAX Implementation (AFTER FIX ✅)
```python
# Use TOTAL shaped reward - both agents see the same reward
shaped_r_by_agent = info.get("shaped_r_by_agent", (0, 0))
total_shaped_reward = sum(shaped_r_by_agent)
dense_reward = (total_shaped_reward, total_shaped_reward)  # Both agents get total
```

---

## FIXES APPLIED ✅

All critical differences have been fixed in the JAX implementation:

### 1. ✅ Observation Encoding
- Added `lossless_state_encoding_legacy()` method producing 20-channel encoding
- Added `use_legacy_encoding=True` flag in `OvercookedJaxEnvConfig`
- Validated encodings match exactly between old and new implementations

### 2. ✅ Layout Files
- Created `random0_legacy.layout` (forced_coordination)
- Created `random3_legacy.layout` (counter_circuit)
- Both use original format with `cook_time=20`, `num_items_for_soup=3`, `delivery_reward=20`

### 3. ✅ Advantage Normalization
- Changed from batch-level to per-minibatch normalization
- Now matches original baselines `model.train()` behavior

### 4. ✅ Fixed Entropy Coefficient
- Set `ent_coef=0.1` (matching original)
- Set `use_entropy_annealing=False` by default

### 5. ✅ Learning Rate Schedule
- Set `use_lr_annealing=False` (original uses LR_ANNEALING=1 = no annealing)
- Set `learning_rate=1e-3` (matching original LR)

### 6. ✅ Other Hyperparameters Fixed
- `clip_eps=0.05` (was 0.2)
- `vf_coef=0.1` (was 0.5)
- `max_grad_norm=0.1` (was 0.5)
- `gae_lambda=0.98` (was 0.95)
- `num_envs=30` (was 32)
- `num_minibatches=6` (was 10)

### 7. ✅ Reward Computation (CRITICAL FIX!)
- Changed from per-agent shaped rewards to **SHARED total shaped reward**
- Both agents now receive the SAME total shaped reward (sum of both agents' contributions)
- This matches the original TensorFlow implementation where `shaped_r` was a single scalar
- Without this fix, agents couldn't learn to coordinate because they didn't get credit for partner's actions

### Usage for Paper Reproduction
```bash
cd overcooked_ai-master/src
python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout random0_legacy
python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout random3_legacy
```

---

## LAYOUT NAME MAPPINGS

| Old Name (human_aware_rl) | New Name (overcooked_ai-master) |
|---------------------------|--------------------------------|
| `random0` | `forced_coordination` |
| `random3` | `counter_circuit` or `counter_circuit_o_1order` |
| `simple` | `cramped_room` |
| `unident_s` | `asymmetric_advantages` |
| `random1` | `coordination_ring` |


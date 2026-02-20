# PPL-Based Models for Overcooked AI

This module provides Probabilistic Programming Language (PPL) based alternatives to traditional neural network approaches for behavior cloning and imitation learning in Overcooked AI.

## Overview

Traditional approaches like BC and PPO learn black-box mappings from states to actions. PPL-based models offer:

1. **Uncertainty Quantification**: Know when the model is uncertain
2. **Interpretability**: Explicit theories of behavior (goals, rationality)
3. **Compositionality**: Build complex models from simple components
4. **Sample Efficiency**: Leverage strong inductive biases

## Available Models

### 1. Bayesian Behavior Cloning (`bayesian_bc.py`)

A Bayesian neural network that maintains uncertainty over its weights.

**Key features:**
- Weight priors for regularization
- Variational inference (SVI) for training
- Posterior predictive sampling for uncertainty
- Detects out-of-distribution states via high entropy

```python
from human_aware_rl.ppl import train_bayesian_bc

# Train
results = train_bayesian_bc("cramped_room", num_epochs=100)

# Use
from human_aware_rl.ppl import BayesianBCAgent
agent = BayesianBCAgent.from_saved(model_dir, featurize_fn)
action, info = agent.action(state)
print(f"Uncertainty: {info['entropy']}")
```

### 2. Rational Agent Model (`rational_agent.py`)

Models humans as approximately rational agents:

```
P(action | state) ∝ exp(β * Q(state, action))
```

**Key features:**
- Learns Q-function (implicit utilities/preferences)
- Infers rationality parameter β from data
- Interpretable: low β = noisy, high β = optimal

```python
from human_aware_rl.ppl import train_rational_agent

# Train with learned rationality
results = train_rational_agent("cramped_room", learn_beta=True)

# The learned β tells you how consistent humans are
print(f"Learned β = {results['betas'][-1]}")  # ~1-5 for humans
```

### 3. Hierarchical BC (`hierarchical_bc.py`)

Assumes behavior is driven by latent subgoals:

```
1. Sample goal g ~ P(g | state)
2. Sample action a ~ P(a | state, g)
```

**Key features:**
- Unsupervised goal discovery
- Goal-conditioned policies
- Interpretable: inspect what goals were inferred

```python
from human_aware_rl.ppl import train_hierarchical_bc

# Train with 8 latent goals
results = train_hierarchical_bc("cramped_room", num_goals=8)

# Inspect learned goals
from human_aware_rl.ppl import HierarchicalBCAgent
agent = HierarchicalBCAgent.from_saved(model_dir, featurize_fn)
action, info = agent.action(state)
print(f"Inferred goal: {info['inferred_goal']}")
```

### 4. WebPPL Bridge (`webppl_bridge.py`)

Interface for WebPPL models (for cognitive science research).

**Key features:**
- Run WebPPL models from Python
- Memoization patterns for efficiency
- RSA-style pragmatic models

```python
from human_aware_rl.ppl.webppl_bridge import create_webppl_agent

# Create a softmax-rational agent
agent = create_webppl_agent(
    model_type="softmax_rational",
    softmax_beta=2.0,
)
```

## Training

### Single Model

```bash
# Train Bayesian BC on cramped_room
python -m human_aware_rl.ppl.bayesian_bc --layout cramped_room --epochs 100

# Train Rational Agent
python -m human_aware_rl.ppl.rational_agent --layout cramped_room --learn_beta

# Train Hierarchical BC with 12 goals
python -m human_aware_rl.ppl.hierarchical_bc --layout cramped_room --num_goals 12
```

### All Models

```bash
# Train all models on one layout
python -m human_aware_rl.ppl.train_ppl --layout cramped_room

# Train all models on all layouts
python -m human_aware_rl.ppl.train_ppl --all_layouts --all_models
```

## Evaluation

```bash
# Evaluate PPL models
python -m human_aware_rl.ppl.evaluate_ppl --layout cramped_room

# Compare with standard BC
python -m human_aware_rl.ppl.compare_with_baselines --all_layouts --markdown
```

## Theoretical Background

### Softmax Rationality

The rational agent model is based on the observation that humans make decisions that approximately maximize expected utility, but with noise. The softmax policy:

```
P(a | s) = exp(β * Q(s, a)) / Σ_a' exp(β * Q(s, a'))
```

Has the following properties:
- As β → 0: Actions become uniform random
- As β → ∞: Actions become optimal (argmax Q)
- For moderate β: Bounded rationality

This is also known as:
- Boltzmann exploration (RL)
- Luce choice rule (psychology)
- Energy-based models (ML)

### Hierarchical Models

The hierarchical model assumes behavior is structured:

1. **Goals/Intentions**: High-level objectives (e.g., "get onion")
2. **Actions**: Low-level movements to achieve goals

This decomposition provides:
- Better generalization (same goal, different contexts)
- Interpretability (we can ask "what is the agent trying to do?")
- Efficient inference (factorized posterior)

### Bayesian Deep Learning

The Bayesian BC model places priors over neural network weights:

```
P(w) = Normal(0, σ²)  # Prior
P(D | w) = Π Categorical(a | softmax(f_w(s)))  # Likelihood
P(w | D) ∝ P(D | w) P(w)  # Posterior
```

Benefits:
- Uncertainty quantification
- Automatic regularization
- Principled model selection

## WebPPL Models

For researchers familiar with WebPPL/Church, the `webppl_bridge.py` module allows running WebPPL code from Python. Example WebPPL model:

```javascript
// Softmax-rational agent
var agent = function(state, beta) {
    return Infer({method: 'enumerate'}, function() {
        var action = uniformDraw(actions);
        var utility = getUtility(state, action);
        factor(beta * utility);
        return action;
    });
};
```

Requirements:
- Node.js
- WebPPL: `npm install -g webppl`

## Dependencies

Add to your `requirements.txt`:

```
pyro-ppl>=1.8.0
numpyro>=0.13.0
```

## References

1. **Bounded Rationality**: Simon, H. A. (1957). Models of Man.
2. **Softmax Q-learning**: Bridle, J. S. (1990). Training stochastic model recognition algorithms.
3. **Inverse RL**: Ng, A. Y., & Russell, S. J. (2000). Algorithms for inverse reinforcement learning.
4. **Options Framework**: Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs.
5. **Goal Inference**: Baker, C. L., Saxe, R., & Tenenbaum, J. B. (2009). Action understanding as inverse planning.
6. **WebPPL**: Goodman, N. D., & Stuhlmüller, A. (2014). The Design and Implementation of Probabilistic Programming Languages.
7. **Bayesian Deep Learning**: Blundell, C., et al. (2015). Weight Uncertainty in Neural Networks.

## Citation

If you use these models in your research, please cite:

```bibtex
@misc{overcooked_ppl,
  title={PPL-Based Models for Overcooked AI},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/overcooked_ai}}
}
```

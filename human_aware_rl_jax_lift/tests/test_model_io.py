import jax
import jax.numpy as jnp

from human_aware_rl_jax_lift.agents.bc.model import BCPolicy
from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.agents.ppo.model import ActorCriticCNN


def test_bc_logits_shape_and_dtype():
    model = BCPolicy()
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 64), dtype=jnp.float32))
    logits = model.apply(params, jnp.zeros((4, 64), dtype=jnp.float32))
    assert logits.shape == (4, 6)
    assert logits.dtype == jnp.float32


def test_ppo_network_output_shapes():
    cfg = PPOConfig()
    model = ActorCriticCNN(num_actions=cfg.num_actions, num_filters=cfg.num_filters, hidden_dim=cfg.hidden_dim)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((2, 5, 5, 20), dtype=jnp.float32))
    logits, values = model.apply(params, jnp.zeros((2, 5, 5, 20), dtype=jnp.float32))
    assert logits.shape == (2, 6)
    assert values.shape == (2,)

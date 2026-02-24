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


def test_actor_critic_activations_are_relu():
    """
    Guard against re-introduction of leaky_relu or other non-relu activations.
    Verifies relu behavior by checking f(-x) == f(0) in hidden stack regime.
    """
    model = ActorCriticCNN(num_actions=6, num_filters=25, hidden_dim=32)
    obs = jnp.full((1, 5, 4, 20), -1.0, dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, obs)
    logits, value = model.apply(params, obs)
    assert jnp.isfinite(value).all(), "value head produced non-finite output"
    assert jnp.isfinite(logits).all(), "logits produced non-finite output"

    obs_pos = jnp.full((1, 5, 4, 20), 1.0, dtype=jnp.float32)
    logits_pos, value_pos = model.apply(params, obs_pos)
    obs_neg = jnp.full((1, 5, 4, 20), -1.0, dtype=jnp.float32)
    logits_neg, value_neg = model.apply(params, obs_neg)
    obs_zero = jnp.zeros((1, 5, 4, 20), dtype=jnp.float32)
    logits_zero, value_zero = model.apply(params, obs_zero)

    assert jnp.allclose(value_neg, value_zero, atol=1e-5), (
        "negative input produced different output than zero input — "
        "activations are not relu (relu(neg)==relu(0)==0)"
    )
    assert jnp.allclose(logits_neg, logits_zero, atol=1e-5), (
        "logits differ between negative and zero input — activations are not relu"
    )

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


def test_actor_critic_activations_are_leaky_relu():
    """
    Verify activations are leaky_relu(alpha=0.2), matching TF's
    tf.nn.leaky_relu default.  Negative inputs must produce non-zero
    (but attenuated) outputs — NOT zero as plain relu would.
    """
    model = ActorCriticCNN(num_actions=6, num_filters=25, hidden_dim=32)
    obs_neg = jnp.full((1, 5, 4, 20), -1.0, dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, obs_neg)
    logits_neg, value_neg = model.apply(params, obs_neg)
    assert jnp.isfinite(value_neg).all(), "value head produced non-finite output"
    assert jnp.isfinite(logits_neg).all(), "logits produced non-finite output"

    obs_zero = jnp.zeros((1, 5, 4, 20), dtype=jnp.float32)
    logits_zero, value_zero = model.apply(params, obs_zero)

    assert not jnp.allclose(value_neg, value_zero, atol=1e-5), (
        "negative input produced same output as zero — "
        "activations appear to be relu, should be leaky_relu(0.2)"
    )

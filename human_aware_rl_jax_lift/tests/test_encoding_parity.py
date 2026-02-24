from human_aware_rl_jax_lift.encoding.bc_features import featurize_state_64
from human_aware_rl_jax_lift.encoding.ppo_masks import lossless_state_encoding_20
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.state import make_initial_state


def test_ppo_masks_shape():
    terrain = parse_layout("simple")
    state = make_initial_state(terrain)
    o0, o1 = lossless_state_encoding_20(terrain, state)
    assert o0.shape[-1] == 20
    assert o1.shape[-1] == 20


def test_bc_features_vector_shape():
    terrain = parse_layout("simple")
    state = make_initial_state(terrain)
    f0, f1 = featurize_state_64(terrain, state)
    assert f0.ndim == 1
    assert f1.ndim == 1

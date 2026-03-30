"""Handcrafted BC featurization (62-dim when tf_compat, 64-dim otherwise)."""

import jax.numpy as jnp
import numpy as np

# Module-level cache for path-cost results.
# Key: (mlp_id, pos_and_or, sorted_locations_tuple) — all discrete, bounded per layout.
# This avoids repeated BFS calls for the same (source, targets) during eval rollouts.
_plan_cache: dict = {}

from human_aware_rl_jax_lift.env.state import (
    OBJ_DISH,
    OBJ_NONE,
    OBJ_ONION,
    OBJ_SOUP,
    OvercookedState,
    SOUP_ONION,
    TERRAIN_COUNTER,
    Terrain,
)


# JAX ori idx: 0=N, 1=S, 2=E, 3=W
_DIRECTION_BY_OR = [(0, -1), (0, 1), (1, 0), (-1, 0)]


def _closest_delta_path_cost(mlp, player_pos, player_or, positions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Closest selection via graph path cost (matches TF get_deltas_to_closest_location).

    Results are cached by (mlp identity, pos_and_or, sorted target locations) since all
    inputs are discrete and bounded per layout. This eliminates repeated BFS calls for
    the same query during eval rollouts.
    """
    locations = [tuple(pos) for pos, m in zip(positions, mask) if m]
    if not locations:
        return np.zeros((2,), dtype=np.float32)
    or_idx = int(player_or)
    direction = _DIRECTION_BY_OR[or_idx]
    pos_and_or = (tuple(map(int, player_pos)), direction)

    cache_key = (id(mlp), pos_and_or, tuple(sorted(locations)))
    cached = _plan_cache.get(cache_key)
    if cached is not None:
        return cached

    _, closest_loc = mlp.mp.min_cost_to_feature(pos_and_or, locations, with_argmin=True)
    if closest_loc is None:
        result = np.zeros((2,), dtype=np.float32)
    else:
        # pos_distance(closest, player) = (dx, dy) = delta from player to closest
        result = (np.array(closest_loc, dtype=np.float32)
                  - np.array(player_pos, dtype=np.float32))
    _plan_cache[cache_key] = result
    return result


def _closest_delta_dist_table(
    dist_table: jnp.ndarray,
    player_pos: jnp.ndarray,
    player_or: jnp.ndarray,
    positions: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """O(1) closest-location via precomputed path-cost distance table.

    dist_table[src_y, src_x, or_idx, tgt_y, tgt_x] = path cost.
    Fully JAX-traceable — safe to use under jax.vmap.
    """
    src_x, src_y = player_pos[0], player_pos[1]
    tgt_xs = positions[:, 0]
    tgt_ys = positions[:, 1]
    costs = dist_table[src_y, src_x, player_or, tgt_ys, tgt_xs]  # [N]
    large = jnp.array(1e6, dtype=jnp.float32)
    masked_costs = jnp.where(mask, costs, large)
    idx = jnp.argmin(masked_costs)
    best_cost = masked_costs[idx]
    # Return (0, 0) when no reachable target exists (matches mlp path / TF behaviour
    # where min_cost_to_feature returns closest_loc=None → delta=(0,0)).
    reachable = jnp.any(mask) & (best_cost < large)
    delta = (positions[idx] - player_pos).astype(jnp.float32)
    return jnp.where(reachable, delta, jnp.zeros(2, dtype=jnp.float32))


def get_mlp_for_layout(layout_name: str):
    """Load and cache MediumLevelPlanner for a layout (for path-cost BC feature parity)."""
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

    if not hasattr(get_mlp_for_layout, "_cache"):
        get_mlp_for_layout._cache = {}
    if layout_name not in get_mlp_for_layout._cache:
        mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, start_order_list=None)
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
        get_mlp_for_layout._cache[layout_name] = (mdp, mlp)
    return get_mlp_for_layout._cache[layout_name][1]


def build_dist_table(mlp, terrain) -> jnp.ndarray:
    """Precompute dist_table[H, W, 4, H, W] of BFS path costs (one-time startup cost).

    dist_table[src_y, src_x, or_idx, tgt_y, tgt_x] = path cost from
    (src_x, src_y) facing _DIRECTION_BY_OR[or_idx] to target (tgt_x, tgt_y).
    Non-reachable pairs get cost 1e6.

    Only fills entries for walkable source positions and for target positions that
    actually appear in the terrain feature arrays (pots, counters, dispensers, etc.).
    All other entries remain 1e6.
    """
    grid_np = np.asarray(terrain.grid)
    H, W = grid_np.shape
    table = np.full((H, W, 4, H, W), 1e6, dtype=np.float32)

    walkable = np.asarray(terrain.walkable_mask)  # [H, W]

    # Collect the unique target positions featurize_state_64 ever queries.
    target_set: set = set()
    for attr in ("onion_disp_positions", "dish_disp_positions",
                 "counter_positions", "pot_positions", "serve_positions"):
        arr = np.asarray(getattr(terrain, attr))  # [N, 2]  (x, y)
        for row in arr:
            target_set.add((int(row[0]), int(row[1])))
    targets = list(target_set)

    for src_y in range(H):
        for src_x in range(W):
            if not walkable[src_y, src_x]:
                continue
            for or_idx in range(4):
                direction = _DIRECTION_BY_OR[or_idx]
                src_pos_and_or = ((src_x, src_y), direction)
                for tgt_x, tgt_y in targets:
                    c = mlp.mp.min_cost_to_feature(src_pos_and_or, [(tgt_x, tgt_y)])
                    table[src_y, src_x, or_idx, tgt_y, tgt_x] = min(float(c), 1e6)

    return jnp.array(table)


def _held_obj_one_hot(obj_id) -> np.ndarray:
    # Dispatch to numpy when obj_id is already numpy (eval/mlp path).
    if isinstance(obj_id, (np.ndarray, np.integer, int)):
        oi = int(obj_id)
        return np.array([float(oi == OBJ_ONION), float(oi == OBJ_SOUP), float(oi == OBJ_DISH)], dtype=np.float32)
    return jnp.array(
        [jnp.where(obj_id == OBJ_ONION, 1.0, 0.0),
         jnp.where(obj_id == OBJ_SOUP, 1.0, 0.0),
         jnp.where(obj_id == OBJ_DISH, 1.0, 0.0)],
        dtype=jnp.float32,
    )


def _orientation_one_hot(o) -> np.ndarray:
    if isinstance(o, (np.ndarray, np.integer, int)):
        return np.eye(4, dtype=np.float32)[int(o)]
    return jnp.eye(4, dtype=jnp.float32)[o]


def _wall_features(terrain: Terrain, pos) -> np.ndarray:
    # Dispatch to numpy when pos is already numpy (eval/mlp path) to avoid
    # GPU→CPU syncs from indexing numpy terrain with JAX scalars.
    xp = np if isinstance(pos, np.ndarray) else jnp
    dirs = xp.array([[0, -1], [0, 1], [1, 0], [-1, 0]], dtype=xp.int32)
    h, w = terrain.grid.shape
    out = []
    for d in dirs:
        p = pos + d
        in_bounds = (p[0] >= 0) & (p[1] >= 0) & (p[0] < w) & (p[1] < h)
        clipped = xp.array(
            [xp.clip(p[0], 0, w - 1), xp.clip(p[1], 0, h - 1)],
            dtype=xp.int32,
        )
        is_wall = ~in_bounds | (~terrain.walkable_mask[clipped[1], clipped[0]])
        out.append(xp.where(is_wall, 1.0, 0.0))
    return xp.array(out, dtype=xp.float32)


def _facing_empty_counter(terrain: Terrain, state: OvercookedState, i: int, tf_compat: bool = False):
    """Check if player faces a counter.

    When ``tf_compat=True``, checks only whether the player faces a counter
    (any-counter). When False, also verifies no object sits on the counter.
    """
    pos = state.player_pos[i]
    xp = np if isinstance(pos, np.ndarray) else jnp
    dirs = xp.array([[0, -1], [0, 1], [1, 0], [-1, 0]], dtype=xp.int32)
    face_pos = pos + dirs[state.player_or[i]]
    in_bounds = (
        (face_pos[0] >= 0)
        & (face_pos[1] >= 0)
        & (face_pos[0] < terrain.grid.shape[1])
        & (face_pos[1] < terrain.grid.shape[0])
    )
    is_counter = in_bounds & (terrain.grid[face_pos[1], face_pos[0]] == TERRAIN_COUNTER)
    if tf_compat:
        result = is_counter
    else:
        # Correct: check that the faced counter has no object on it.
        # counter_positions is [max_counters, 2], counter_obj is [max_counters].
        # Find if face_pos matches any counter and check its object.
        cpos = terrain.counter_positions  # [N, 2]
        match = (cpos[:, 0] == face_pos[0]) & (cpos[:, 1] == face_pos[1])
        # Sum matched object ids; unmatched slots contribute 0 (OBJ_NONE)
        matched_obj = xp.sum(match.astype(xp.int32) * state.counter_obj)
        has_obj = matched_obj != OBJ_NONE
        result = is_counter & ~has_obj
    return xp.array([xp.where(result, 1.0, 0.0)], dtype=xp.float32)


def _facing_and_walls_legacy_order(terrain: Terrain, state: OvercookedState, i: int, tf_compat: bool = False) -> jnp.ndarray:
    """Facing-empty-counter + wall features in TF order.

    TF inserts facing at index == player orientation, then wall_0..wall_3.
    So when or=0: [facing, w0, w1, w2, w3]; when or=1: [w0, facing, w1, w2, w3]; etc.
    """
    facing = _facing_empty_counter(terrain, state, i, tf_compat=tf_compat)[0]
    walls = _wall_features(terrain, state.player_pos[i])
    or_i = state.player_or[i]
    # At index or_i: facing; elsewhere: wall[k] for k<or_i, wall[k-1] for k>or_i
    xp = np if isinstance(or_i, (np.ndarray, np.integer, int)) else jnp
    k = xp.arange(5, dtype=xp.int32)
    wall_idx = k - (k > or_i).astype(xp.int32)
    return xp.where(k == or_i, facing, walls[wall_idx]).astype(xp.float32)


def featurize_state_64(terrain: Terrain, state: OvercookedState, mlp=None, dist_table=None, tf_compat=False):
    """Return `(features_p0, features_p1)` vectors following legacy feature layout.

    Exactly one of the following must be provided:
    - `dist_table`: JAX array [H, W, 4, H, W] from build_dist_table(); pure JAX,
      vmappable, GPU-safe.  Use this at eval time.
    - `mlp`: MediumLevelPlanner; BFS path cost (CPU only, no vmap).  Used by
      prepare_bc_data.py during data preparation.
    """

    def player_feats(i: int):
        pos = state.player_pos[i]
        held = state.held_obj[i]
        pot_type = state.pot_state[:, 0]
        pot_items = state.pot_state[:, 1]
        pot_cook = state.pot_state[:, 2]
        pot_valid = terrain.pot_mask

        empty_pot_mask     = pot_valid & (pot_type == 0)
        one_onion_pot_mask = pot_valid & (pot_type == SOUP_ONION) & (pot_items == 1)
        two_onion_pot_mask = pot_valid & (pot_type == SOUP_ONION) & (pot_items == 2)
        cooking_pot_mask   = pot_valid & (pot_type == SOUP_ONION) & (pot_items == 3) & (pot_cook < terrain.cook_time)
        ready_pot_mask     = pot_valid & (pot_type == SOUP_ONION) & (pot_items == 3) & (pot_cook >= terrain.cook_time)

        counter_onion_mask = (state.counter_obj == OBJ_ONION) & terrain.counter_mask
        counter_dish_mask  = (state.counter_obj == OBJ_DISH)  & terrain.counter_mask
        counter_soup_mask  = (state.counter_obj == OBJ_SOUP)  & terrain.counter_mask

        or_i = state.player_or[i]

        if dist_table is not None:
            # ── Pure JAX: O(1) table lookup — vmappable, GPU-safe. ─────────────
            def _cl(positions, mask):
                return _closest_delta_dist_table(dist_table, pos, or_i, positions, mask)

            onion_pos  = jnp.concatenate([terrain.onion_disp_positions, terrain.counter_positions], axis=0)
            onion_mask = jnp.concatenate([terrain.onion_disp_mask,      counter_onion_mask],        axis=0)
            onion_delta = _cl(onion_pos, onion_mask)

            dish_pos   = jnp.concatenate([terrain.dish_disp_positions, terrain.counter_positions], axis=0)
            dish_mask  = jnp.concatenate([terrain.dish_disp_mask,      counter_dish_mask],         axis=0)
            dish_delta  = _cl(dish_pos, dish_mask)

            soup_delta  = _cl(terrain.counter_positions, counter_soup_mask)
            serve_delta = _cl(terrain.serve_positions,   terrain.serve_mask)
            pot_deltas  = [
                _cl(terrain.pot_positions, empty_pot_mask),
                _cl(terrain.pot_positions, one_onion_pot_mask),
                _cl(terrain.pot_positions, two_onion_pot_mask),
                _cl(terrain.pot_positions, cooking_pot_mask),
                _cl(terrain.pot_positions, ready_pot_mask),
            ]
            zeros2      = jnp.zeros(2, dtype=jnp.float32)
            onion_delta = jnp.where(held == OBJ_ONION, zeros2, onion_delta)
            dish_delta  = jnp.where(held == OBJ_DISH,  zeros2, dish_delta)
            soup_delta  = jnp.where(held == OBJ_SOUP,  zeros2, soup_delta)
            wall_or_facing = (
                _wall_features(terrain, pos) if tf_compat
                else _facing_and_walls_legacy_order(terrain, state, i, tf_compat=tf_compat)
            )
            feats = [
                _orientation_one_hot(or_i),
                _held_obj_one_hot(held),
                onion_delta,
                pot_deltas[0], pot_deltas[1], pot_deltas[2], pot_deltas[3], pot_deltas[4],
                dish_delta, soup_delta, serve_delta,
                wall_or_facing,
            ]
            return jnp.concatenate(feats, axis=0)

        elif mlp is not None:
            # ── Python BFS path — CPU only, used by prepare_bc_data.py. ────────
            def _cl(positions, mask):
                return _closest_delta_path_cost(
                    mlp, np.asarray(pos), int(or_i), np.asarray(positions), np.asarray(mask)
                )

            onion_pos  = np.concatenate([np.asarray(terrain.onion_disp_positions), np.asarray(terrain.counter_positions)], axis=0)
            onion_mask = np.concatenate([np.asarray(terrain.onion_disp_mask),      np.asarray(counter_onion_mask)],        axis=0)
            onion_delta = _cl(onion_pos, onion_mask)

            dish_pos   = np.concatenate([np.asarray(terrain.dish_disp_positions), np.asarray(terrain.counter_positions)], axis=0)
            dish_mask  = np.concatenate([np.asarray(terrain.dish_disp_mask),      np.asarray(counter_dish_mask)],         axis=0)
            dish_delta  = _cl(dish_pos, dish_mask)

            soup_delta  = _cl(terrain.counter_positions, counter_soup_mask)
            serve_delta = _cl(terrain.serve_positions,   terrain.serve_mask)
            pot_deltas  = [
                _cl(terrain.pot_positions, empty_pot_mask),
                _cl(terrain.pot_positions, one_onion_pot_mask),
                _cl(terrain.pot_positions, two_onion_pot_mask),
                _cl(terrain.pot_positions, cooking_pot_mask),
                _cl(terrain.pot_positions, ready_pot_mask),
            ]
            zeros2      = np.zeros(2, dtype=np.float32)
            onion_delta = np.where(np.asarray(held) == OBJ_ONION, zeros2, onion_delta)
            dish_delta  = np.where(np.asarray(held) == OBJ_DISH,  zeros2, dish_delta)
            soup_delta  = np.where(np.asarray(held) == OBJ_SOUP,  zeros2, soup_delta)
            wall_or_facing = (
                _wall_features(terrain, np.asarray(pos)) if tf_compat
                else np.asarray(_facing_and_walls_legacy_order(terrain, state, i, tf_compat=tf_compat))
            )
            feats = [
                _orientation_one_hot(state.player_or[i]),
                _held_obj_one_hot(held),
                onion_delta,
                pot_deltas[0], pot_deltas[1], pot_deltas[2], pot_deltas[3], pot_deltas[4],
                dish_delta, soup_delta, serve_delta,
                wall_or_facing,
            ]
            return np.concatenate([np.asarray(f) for f in feats], axis=0)

        else:
            raise ValueError("featurize_state_64 requires either dist_table or mlp.")

    p0 = player_feats(0)
    p1 = player_feats(1)

    if mlp is not None:
        # numpy CPU path
        rel10 = (np.asarray(state.player_pos[1]) - np.asarray(state.player_pos[0])).astype(np.float32)
        rel01 = (np.asarray(state.player_pos[0]) - np.asarray(state.player_pos[1])).astype(np.float32)
        abs0  = np.asarray(state.player_pos[0]).astype(np.float32)
        # When tf_compat=True, P1's absolute position field uses P0's position.
        abs1  = np.asarray(state.player_pos[0 if tf_compat else 1]).astype(np.float32)
        return np.concatenate([p0, p1, rel10, abs0], axis=0), np.concatenate([p1, p0, rel01, abs1], axis=0)
    else:
        # JAX path (dist_table)
        rel10 = (state.player_pos[1] - state.player_pos[0]).astype(jnp.float32)
        rel01 = (state.player_pos[0] - state.player_pos[1]).astype(jnp.float32)
        abs0  = state.player_pos[0].astype(jnp.float32)
        # When tf_compat=True, P1's absolute position field uses P0's position.
        abs1  = state.player_pos[0 if tf_compat else 1].astype(jnp.float32)
        return jnp.concatenate([p0, p1, rel10, abs0], axis=0), jnp.concatenate([p1, p0, rel01, abs1], axis=0)

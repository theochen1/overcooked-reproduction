"""Adapters between JAX-lift state and legacy planner state."""

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from human_aware_rl_jax_lift.env.state import (
    OBJ_DISH,
    OBJ_ONION,
    OBJ_SOUP,
    OBJ_TOMATO,
    OvercookedState as JaxOvercookedState,
    SOUP_ONION,
)

from overcooked_ai_py.agents.agent import Agent, AgentPair, CoupledPlanningAgent, EmbeddedPlanningAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Direction
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedState, PlayerState
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS


_ORIENTATION_BY_IDX = {
    0: Direction.NORTH,
    1: Direction.SOUTH,
    2: Direction.EAST,
    3: Direction.WEST,
}


def _soup_type_name(soup_type_id: int) -> str:
    return "onion" if int(soup_type_id) == int(SOUP_ONION) else "tomato"


def _obj_name(obj_id: int) -> str:
    if obj_id == OBJ_ONION:
        return "onion"
    if obj_id == OBJ_TOMATO:
        return "tomato"
    if obj_id == OBJ_DISH:
        return "dish"
    if obj_id == OBJ_SOUP:
        return "soup"
    raise ValueError(f"Unsupported object id: {obj_id}")


def jax_to_legacy_state(terrain, state: JaxOvercookedState, order_list=None) -> OvercookedState:
    """Convert JAX-lift `OvercookedState` to legacy `overcooked_ai_py` state."""
    players = []
    for i in range(2):
        pos = tuple(np.asarray(state.player_pos[i]).tolist())
        ori_idx = int(state.player_or[i])
        ori = _ORIENTATION_BY_IDX[ori_idx]
        held_obj = int(state.held_obj[i])
        if held_obj == 0:
            held = None
        elif held_obj == OBJ_SOUP:
            soup_payload = np.asarray(state.held_soup[i]).tolist()
            held = ObjectState(
                "soup",
                pos,
                state=(
                    _soup_type_name(int(soup_payload[0])),
                    int(soup_payload[1]),
                    int(soup_payload[2]),
                ),
            )
        else:
            held = ObjectState(_obj_name(held_obj), pos)
        players.append(PlayerState(pos, ori, held))

    objects = {}
    # Counter objects
    for i, valid in enumerate(np.asarray(terrain.counter_mask).tolist()):
        if not valid:
            continue
        pos = tuple(np.asarray(terrain.counter_positions[i]).tolist())
        obj_id = int(state.counter_obj[i])
        if obj_id == 0:
            continue
        if obj_id == OBJ_SOUP:
            soup_payload = np.asarray(state.counter_soup[i]).tolist()
            obj = ObjectState(
                "soup",
                pos,
                state=(
                    _soup_type_name(int(soup_payload[0])),
                    int(soup_payload[1]),
                    int(soup_payload[2]),
                ),
            )
        else:
            obj = ObjectState(_obj_name(obj_id), pos)
        objects[pos] = obj

    # Pot soups (legacy planners reason over soups in pots as objects)
    for i, valid in enumerate(np.asarray(terrain.pot_mask).tolist()):
        if not valid:
            continue
        pot_payload = np.asarray(state.pot_state[i]).tolist()
        soup_type_id, num_items, cook_time = map(int, pot_payload)
        if soup_type_id == 0 or num_items <= 0:
            continue
        pos = tuple(np.asarray(terrain.pot_positions[i]).tolist())
        objects[pos] = ObjectState(
            "soup",
            pos,
            state=(_soup_type_name(soup_type_id), num_items, cook_time),
        )

    return OvercookedState(players=players, objects=objects, order_list=order_list)


class DeterministicPolicyAgent(Agent):
    """Legacy planner-compatible agent wrapping a deterministic callback."""

    def __init__(self, policy_fn: Callable[[OvercookedState, int], object]):
        self.policy_fn = policy_fn

    def action(self, state):
        return self.policy_fn(state, self.agent_index)


@dataclass
class PlanningEvalHarness:
    """CP/PBC evaluation harness around legacy planning stack."""

    layout_name: str
    horizon: int = 400
    force_compute: bool = False

    def __post_init__(self):
        mdp_params = {"layout_name": self.layout_name, "start_order_list": None}
        env_params = {"horizon": self.horizon}
        self._ae = AgentEvaluator(
            mdp_params=mdp_params,
            env_params=env_params,
            force_compute=self.force_compute,
            mlp_params=NO_COUNTERS_PARAMS,
        )

    @property
    def evaluator(self) -> AgentEvaluator:
        return self._ae

    def evaluate_cp(self, num_games: int = 100, delivery_horizon: int = 2, display: bool = False):
        """Evaluate CP pair (CoupledPlanningAgent + CoupledPlanningAgent)."""
        a0 = CoupledPlanningAgent(self._ae.mlp, delivery_horizon=delivery_horizon)
        a1 = CoupledPlanningAgent(self._ae.mlp, delivery_horizon=delivery_horizon)
        a0.mlp.env = self._ae.env
        a1.mlp.env = self._ae.env
        pair = AgentPair(a0, a1)
        return self._ae.evaluate_agent_pair(pair, num_games=num_games, display=display, info=True)

    def evaluate_pbc(
        self,
        other_agent: Agent,
        num_games: int = 100,
        pbc_index: int = 0,
        delivery_horizon: int = 2,
        display: bool = False,
    ):
        """
        Evaluate PBC agent against a fixed partner model.

        `other_agent` is treated as deterministic during embedded planning.
        """
        planner_partner = copy.deepcopy(other_agent)
        rollout_partner = copy.deepcopy(other_agent)
        pbc = EmbeddedPlanningAgent(planner_partner, self._ae.mlp, self._ae.env, delivery_horizon=delivery_horizon)
        pair = AgentPair(pbc, rollout_partner) if pbc_index == 0 else AgentPair(rollout_partner, pbc)
        return self._ae.evaluate_agent_pair(pair, num_games=num_games, display=display, info=True)

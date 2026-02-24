"""Population-based training orchestration."""

import copy
import random
from dataclasses import dataclass
from typing import Dict, List

from .config import PBTConfig


def _mutate_value(key: str, value, factors):
    if key == "LAM":
        eps = min((1.0 - value) / 2.0, value / 2.0)
        direction = random.choice([-1.0, 1.0])
        return max(0.0, min(1.0, value + direction * eps))
    factor = random.choice(factors)
    out = value * factor
    if key == "STEPS_PER_UPDATE":
        return max(1, int(out))
    return out


@dataclass
class PopulationMember:
    params: Dict[str, float]
    fitness: float = 0.0

    def mutate(self, cfg: PBTConfig):
        for k in cfg.mutable_keys:
            if k in self.params and random.random() < cfg.resample_prob:
                self.params[k] = _mutate_value(k, self.params[k], cfg.mutation_factors)


class PBTTrainer:
    def __init__(self, base_hparams: Dict[str, float], config: PBTConfig | None = None):
        self.config = config or PBTConfig()
        self.population = [PopulationMember(copy.deepcopy(base_hparams)) for _ in range(self.config.population_size)]

    def exploit_and_explore(self) -> dict[int, int]:
        """Exploit+explore and return {replaced_idx: source_idx}."""
        ranked = sorted(enumerate(self.population), key=lambda x: x[1].fitness)
        k = max(1, self.config.population_size // 4)
        worst = ranked[:k]
        best = ranked[-k:]
        copy_map: dict[int, int] = {}
        for widx, w in worst:
            bidx, src = random.choice(best)
            w.params = copy.deepcopy(src.params)
            w.mutate(self.config)
            copy_map[widx] = bidx
        return copy_map

    def update_fitness(self, fitnesses: List[float]):
        for m, f in zip(self.population, fitnesses):
            m.fitness = float(f)

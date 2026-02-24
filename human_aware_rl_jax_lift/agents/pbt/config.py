"""PBT configuration from legacy appendix settings."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PBTConfig:
    population_size: int = 3
    resample_prob: float = 0.33
    mutation_factors: List[float] = field(default_factory=lambda: [0.75, 1.25])
    mutable_keys: List[str] = field(
        default_factory=lambda: ["LAM", "CLIPPING", "LR", "STEPS_PER_UPDATE", "ENTROPY", "VF_COEF"]
    )
    iter_per_selection: int = 9
    num_selection_games: int = 6

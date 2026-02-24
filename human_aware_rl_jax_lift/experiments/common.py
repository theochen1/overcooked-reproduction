"""Shared experiment helpers."""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class EvalSummary:
    mean: float
    std: float
    stderr: float


def summarize(xs: Iterable[float]) -> EvalSummary:
    arr = np.asarray(list(xs), dtype=np.float64)
    if arr.size == 0:
        return EvalSummary(0.0, 0.0, 0.0)
    return EvalSummary(float(arr.mean()), float(arr.std()), float(arr.std() / max(np.sqrt(arr.size), 1.0)))

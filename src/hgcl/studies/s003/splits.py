"""Deterministic S003 label-budget construction."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Hashable, Mapping


@dataclass(frozen=True)
class Budget:
    seed: int
    fraction: float
    fit_ids: frozenset[Hashable]
    validation_ids: frozenset[Hashable]
    refit_ids: frozenset[Hashable]
    digest: str


def build_label_budgets(
    labels: Mapping[Hashable, int],
    *,
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71),
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 1.0),
) -> dict[tuple[int, float], Budget]:
    by_class = {value: sorted(key for key, label in labels.items() if label == value) for value in (0, 1)}
    if any(len(ids) < 2 for ids in by_class.values()):
        raise ValueError("each known class needs fit and validation examples")
    result = {}
    for seed in seeds:
        pools = {}
        for label, ids in by_class.items():
            shuffled = list(ids)
            random.Random(seed * 1009 + label).shuffle(shuffled)
            fit_count = min(max(1, math.floor(len(shuffled) * 0.8)), len(shuffled) - 1)
            pools[label] = (shuffled[:fit_count], shuffled[fit_count:])
        for fraction in fractions:
            fit_ids: set[Hashable] = set()
            validation_ids: set[Hashable] = set()
            for fit_pool, validation_pool in pools.values():
                fit_ids.update(fit_pool[: max(1, math.ceil(len(fit_pool) * fraction))])
                validation_ids.update(validation_pool[: max(1, math.ceil(len(validation_pool) * fraction))])
            refit = fit_ids | validation_ids
            canonical = {"seed": seed, "fraction": fraction, "fit": sorted(map(str, fit_ids)), "validation": sorted(map(str, validation_ids))}
            digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
            result[(seed, fraction)] = Budget(seed, fraction, frozenset(fit_ids), frozenset(validation_ids), frozenset(refit), digest)
    return result

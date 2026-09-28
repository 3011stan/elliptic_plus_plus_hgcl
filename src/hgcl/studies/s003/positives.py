"""Snapshot-local positive and negative set construction."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class PositiveSets:
    positives: tuple[frozenset[int], ...]
    negatives: tuple[frozenset[int], ...]
    valid_anchors: tuple[int, ...]


def cosine_knn(x: torch.Tensor, tx_ids: tuple[str, ...], *, k: int = 10) -> tuple[tuple[int, ...], ...]:
    if x.shape[0] != len(tx_ids):
        raise ValueError("tx_ids must align with rows")
    if len(tx_ids) <= 1:
        return tuple(() for _ in tx_ids)
    similarity = functional.normalize(x, dim=1) @ functional.normalize(x, dim=1).T
    count = min(k, len(tx_ids) - 1)
    rows = []
    for anchor in range(len(tx_ids)):
        candidates = [index for index in range(len(tx_ids)) if index != anchor]
        candidates.sort(key=lambda index: (-float(similarity[anchor, index]), str(tx_ids[index])))
        rows.append(tuple(candidates[:count]))
    return tuple(rows)


def build_positive_sets(node_count: int, edge_index: torch.Tensor, knn: tuple[tuple[int, ...], ...]) -> PositiveSets:
    if len(knn) != node_count:
        raise ValueError("KNN rows must match node count")
    successors = [set() for _ in range(node_count)]
    for source, target in edge_index.T.tolist():
        successors[source].add(target)
    universe = set(range(node_count))
    positives = []
    negatives = []
    for anchor in range(node_count):
        positive = {anchor, *successors[anchor], *knn[anchor]}
        positives.append(frozenset(positive))
        negatives.append(frozenset(universe - positive))
    valid = tuple(index for index, negative in enumerate(negatives) if negative)
    return PositiveSets(tuple(positives), tuple(negatives), valid)

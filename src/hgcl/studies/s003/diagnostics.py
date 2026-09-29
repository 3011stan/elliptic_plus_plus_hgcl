"""Representation diagnostics (alignment, uniformity, effective rank) for S003."""

from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np
import torch


def alignment(
    embeddings: torch.Tensor,
    positive_pairs: torch.Tensor | Sequence[tuple[int, int]],
    *,
    alpha: float = 2.0,
) -> float:
    """Compute representation alignment on positive pairs: E_{(i,j)}[||u_i - u_j||_2^alpha]."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return 0.0

    if isinstance(positive_pairs, torch.Tensor):
        if positive_pairs.ndim != 2 or positive_pairs.shape[1] == 0:
            return 0.0
        # Filter out self-pairs (i == j)
        mask = positive_pairs[0] != positive_pairs[1]
        if not mask.any():
            return 0.0
        src = positive_pairs[0][mask]
        dst = positive_pairs[1][mask]
    else:
        filtered = [(int(u), int(v)) for u, v in positive_pairs if int(u) != int(v)]
        if not filtered:
            return 0.0
        src = torch.tensor([u for u, _ in filtered], dtype=torch.long, device=embeddings.device)
        dst = torch.tensor([v for _, v in filtered], dtype=torch.long, device=embeddings.device)

    # L2 normalize embeddings
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = embeddings / norms

    diff = normalized[src] - normalized[dst]
    dist = torch.norm(diff, p=2, dim=1).pow(alpha)
    return float(dist.mean().item())


def uniformity(embeddings: torch.Tensor, *, t: float = 2.0) -> float:
    """Compute representation uniformity on the unit hypersphere: log E_{i!=j}[exp(-t * ||u_i - u_j||_2^2)]."""
    n = embeddings.shape[0]
    if n < 2:
        return 0.0

    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = embeddings / norms

    # Pairwise squared Euclidean distance: ||u_i - u_j||^2 = 2 - 2 * (u_i . u_j)
    cosine_sim = torch.clamp(torch.mm(normalized, normalized.t()), min=-1.0, max=1.0)
    sq_dist = torch.clamp(2.0 - 2.0 * cosine_sim, min=0.0)

    kernel = torch.exp(-t * sq_dist)
    off_diagonal_mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
    avg_kernel = kernel[off_diagonal_mask].mean()
    return float(torch.log(avg_kernel.clamp_min(1e-12)).item())


def effective_rank(embeddings: torch.Tensor) -> float:
    """Compute Roy & Vetterli effective rank: exp(-sum p_k ln p_k) where p_k = sigma_k / sum sigma."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        return 0.0

    singular_values = torch.linalg.svdvals(embeddings.to(torch.float64))
    sv_sum = singular_values.sum()
    if float(sv_sum.item()) <= 1e-12:
        return 0.0

    probabilities = singular_values / sv_sum
    positive_p = probabilities[probabilities > 1e-12]
    entropy = -torch.sum(positive_p * torch.log(positive_p))
    return float(torch.exp(entropy).item())


def compute_snapshot_diagnostics(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    positive_sets: Sequence[Sequence[int]] | None = None,
) -> dict[str, float]:
    """Compute alignment, uniformity, and effective rank for a single snapshot."""
    n = embeddings.shape[0]
    if n == 0:
        return {"alignment": 0.0, "uniformity": 0.0, "effective_rank": 0.0, "anchor_count": 0}

    if positive_sets is not None:
        pairs = []
        for i, pos_list in enumerate(positive_sets):
            for j in pos_list:
                if i != j:
                    pairs.append((i, int(j)))
        align = alignment(embeddings, pairs)
    else:
        align = alignment(embeddings, edge_index)

    unif = uniformity(embeddings)
    erank = effective_rank(embeddings)
    return {
        "alignment": align,
        "uniformity": unif,
        "effective_rank": erank,
        "anchor_count": n,
    }


def aggregate_diagnostics_across_snapshots(
    snapshot_diagnostics: Mapping[int, Mapping[str, float]],
) -> dict[str, float]:
    """Calculate weighted average across snapshots weighted by the anchor count."""
    total_anchors = sum(int(diag.get("anchor_count", 0)) for diag in snapshot_diagnostics.values())
    if total_anchors == 0:
        return {"alignment": 0.0, "uniformity": 0.0, "effective_rank": 0.0, "total_anchors": 0}

    weighted_alignment = sum(
        float(diag.get("alignment", 0.0)) * int(diag.get("anchor_count", 0))
        for diag in snapshot_diagnostics.values()
    ) / total_anchors
    weighted_uniformity = sum(
        float(diag.get("uniformity", 0.0)) * int(diag.get("anchor_count", 0))
        for diag in snapshot_diagnostics.values()
    ) / total_anchors
    weighted_effective_rank = sum(
        float(diag.get("effective_rank", 0.0)) * int(diag.get("anchor_count", 0))
        for diag in snapshot_diagnostics.values()
    ) / total_anchors

    return {
        "alignment": float(weighted_alignment),
        "uniformity": float(weighted_uniformity),
        "effective_rank": float(weighted_effective_rank),
        "total_anchors": total_anchors,
    }


def aggregate_diagnostics_across_seeds(
    seed_summaries: Mapping[int, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute mean and standard deviation across seeds for each diagnostic."""
    metrics = ("alignment", "uniformity", "effective_rank")
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        vals = [float(summary[metric]) for summary in seed_summaries.values() if metric in summary]
        if not vals:
            result[metric] = {"mean": 0.0, "std": 0.0}
        elif len(vals) == 1:
            result[metric] = {"mean": float(vals[0]), "std": 0.0}
        else:
            result[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
            }
    return result

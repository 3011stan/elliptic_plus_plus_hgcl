"""S003 contrastive graph and feature transformations."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ContrastiveView:
    x: torch.Tensor
    edge_index: torch.Tensor
    masked_features: int


@dataclass(frozen=True)
class ContrastivePair:
    names: tuple[str, str]
    views: tuple[ContrastiveView, ContrastiveView]


def make_contrastive_views(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *, seed: int,
    feature_drop: float = 0.1,
    edge_drop: float = 0.1,
) -> ContrastivePair:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    feature_mask = torch.rand(x.shape, generator=generator) < feature_drop
    stochastic_x = x.clone()
    stochastic_x[feature_mask] = 0
    keep = torch.rand(edge_index.shape[1], generator=generator) >= edge_drop
    blocks = ((0, 93), (93, 165), (165, 182))
    block_index = int(torch.randint(0, len(blocks), (1,), generator=generator))
    start, end = blocks[block_index]
    block_x = x.clone()
    block_x[:, start:end] = 0
    return ContrastivePair(
        ("stochastic", "functional_blocks"),
        (
            ContrastiveView(stochastic_x, edge_index[:, keep], int(feature_mask.sum())),
            ContrastiveView(block_x, edge_index.clone(), end - start),
        ),
    )

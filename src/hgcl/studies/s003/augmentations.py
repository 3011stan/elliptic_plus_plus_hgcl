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


def make_two_stochastic_views(
    x: torch.Tensor, edge_index: torch.Tensor, *, seed: int, feature_drop: float = 0.1, edge_drop: float = 0.1
) -> ContrastivePair:
    views = []
    for offset in (0, 1):
        generator = torch.Generator(device="cpu").manual_seed(seed + offset)
        mask = torch.rand(x.shape, generator=generator) < feature_drop
        augmented = x.clone()
        augmented[mask] = 0
        keep = torch.rand(edge_index.shape[1], generator=generator) >= edge_drop
        views.append(ContrastiveView(augmented, edge_index[:, keep], int(mask.sum())))
    return ContrastivePair(("stochastic_1", "stochastic_2"), tuple(views))


def masking_columns(policy: str, *, seed: int, feature_count: int = 182) -> tuple[int, ...]:
    sizes = (93, 72, 17)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = int(torch.randint(0, 3, (1,), generator=generator))
    if policy == "functional_blocks":
        starts = (0, 93, 165)
        return tuple(range(starts[selected], starts[selected] + sizes[selected]))
    permutation = torch.randperm(feature_count, generator=generator).tolist()
    if policy == "random_individual":
        return tuple(sorted(permutation[: sizes[selected]]))
    if policy == "random_groups":
        groups = (permutation[:93], permutation[93:165], permutation[165:182])
        return tuple(sorted(groups[selected]))
    raise ValueError(f"unknown masking policy: {policy}")


def make_s003_views(
    x: torch.Tensor, edge_index: torch.Tensor, *, seed: int, masking_policy: str = "functional_blocks", edge_drop: float = 0.1
) -> ContrastivePair:
    stochastic = make_two_stochastic_views(x, edge_index, seed=seed, edge_drop=edge_drop).views[0]
    columns = masking_columns(masking_policy, seed=seed)
    masked = x.clone()
    masked[:, list(columns)] = 0
    return ContrastivePair(("stochastic", masking_policy), (stochastic, ContrastiveView(masked, edge_index.clone(), len(columns))))


def maybe_add_reverse_edges(edge_index: torch.Tensor, *, enabled: bool, approved: bool) -> torch.Tensor:
    if enabled and not approved:
        raise ValueError("reverse-edge ablation requires explicit resource approval")
    if not enabled:
        return edge_index
    combined = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    return torch.unique(combined.T, dim=0).T.contiguous()

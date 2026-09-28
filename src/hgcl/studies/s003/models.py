"""S003 encoders, projectors, and contrastive objectives."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional
from torch_geometric.nn import GCNConv, GINConv, SAGEConv

from .positives import PositiveSets


class DirectedGINEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 2) -> None:
        super().__init__()
        if layers != 2 or hidden_dim != 128:
            raise ValueError("S003 encoder contract is GIN 2x128")
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for index in range(layers):
            width = input_dim if index == 0 else hidden_dim
            mlp = nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convolutions.append(GINConv(mlp, train_eps=True, flow="source_to_target"))
            self.normalizations.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            x = functional.relu(normalization(convolution(x, edge_index)))
        return x


class S003ContrastiveModel(nn.Module):
    def __init__(self, input_dim: int = 182) -> None:
        super().__init__()
        self.encoder = DirectedGINEncoder(input_dim)
        self.projector = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def forward(self, left_x: torch.Tensor, left_edges: torch.Tensor, right_x: torch.Tensor, right_edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.projector(self.encode(left_x, left_edges)), self.projector(self.encode(right_x, right_edges))

    @torch.no_grad()
    def frozen_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        result = self.encode(x, edge_index).detach()
        self.train(was_training)
        return result


class DirectedSupervisedGNN(nn.Module):
    """Two-layer directed 128d encoder with a node-classification head."""

    def __init__(self, kind: str, input_dim: int = 182) -> None:
        super().__init__()
        if kind not in {"gcn", "graphsage", "gin"}:
            raise ValueError(f"unsupported supervised GNN: {kind}")
        self.kind = kind
        self.layers = 2
        self.hidden_dim = 128
        if kind == "gcn":
            self.convolutions = nn.ModuleList(
                [GCNConv(input_dim, 128, flow="source_to_target"), GCNConv(128, 128, flow="source_to_target")]
            )
        elif kind == "graphsage":
            self.convolutions = nn.ModuleList(
                [SAGEConv(input_dim, 128, flow="source_to_target"), SAGEConv(128, 128, flow="source_to_target")]
            )
        else:
            self.convolutions = nn.ModuleList(
                [
                    GINConv(nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128)), train_eps=True, flow="source_to_target"),
                    GINConv(nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)), train_eps=True, flow="source_to_target"),
                ]
            )
        self.normalizations = nn.ModuleList([nn.LayerNorm(128), nn.LayerNorm(128)])
        self.classifier = nn.Linear(128, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            x = functional.relu(normalization(convolution(x, edge_index)))
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, edge_index)).squeeze(-1)


def _directional_loss(anchor: torch.Tensor, target: torch.Tensor, sets: PositiveSets, temperature: float) -> torch.Tensor:
    if not sets.valid_anchors:
        raise ValueError("epoch has no valid anchor")
    similarities = functional.normalize(anchor, dim=1) @ functional.normalize(target, dim=1).T / temperature
    losses = []
    for index in sets.valid_anchors:
        candidates = sorted(sets.positives[index] | sets.negatives[index])
        positives = sorted(sets.positives[index])
        losses.append(torch.logsumexp(similarities[index, candidates], dim=0) - torch.logsumexp(similarities[index, positives], dim=0))
    return torch.stack(losses).mean()


def symmetric_multi_positive_loss(left: torch.Tensor, right: torch.Tensor, sets: PositiveSets, *, temperature: float = 0.2) -> torch.Tensor:
    return 0.5 * (_directional_loss(left, right, sets, temperature) + _directional_loss(right, left, sets, temperature))

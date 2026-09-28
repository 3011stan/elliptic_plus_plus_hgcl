"""Label-free pretraining and frozen-encoder downstream training."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import time
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score, matthews_corrcoef

from .augmentations import make_contrastive_views
from .models import S003ContrastiveModel, symmetric_multi_positive_loss
from .positives import build_positive_sets, cosine_knn


@dataclass(frozen=True)
class PretrainingResult:
    losses: tuple[float, ...]
    epochs_completed: int
    valid_anchors: tuple[int, ...]


class DownstreamMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@dataclass(frozen=True)
class DownstreamSelection:
    model: DownstreamMLP
    threshold: float
    f1_illicit: float
    mcc: float
    hyperparameters: dict[str, float | int]
    input_dim: int
    encoder_parameters: tuple[nn.Parameter, ...]


@dataclass(frozen=True)
class RefitClassifier:
    model: DownstreamMLP
    threshold: float
    input_dim: int


def pretrain_ssl(
    model: S003ContrastiveModel,
    snapshots: Iterable[tuple[int, tuple[str, ...], torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    seed: int,
    knn_k: int = 10,
    device: str | torch.device = "cpu",
    max_seconds: float | None = None,
    checkpoint_hook: Callable[[int, S003ContrastiveModel, torch.optim.Optimizer], None] | None = None,
) -> PretrainingResult:
    if epochs <= 0:
        raise ValueError("SSL epochs must be fixed and positive")
    torch.manual_seed(seed)
    device = torch.device(device)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    materialized = list(snapshots)
    started = time.monotonic()
    epoch_losses: list[float] = []
    anchors_by_epoch: list[int] = []
    for epoch in range(epochs):
        losses = []
        valid_anchor_count = 0
        for step, tx_ids, x_cpu, edges_cpu in materialized:
            if x_cpu.shape[0] < 2:
                continue
            pair = make_contrastive_views(x_cpu, edges_cpu, seed=seed + epoch * 1009 + step)
            knn = cosine_knn(x_cpu, tx_ids, k=knn_k)
            positive_sets = build_positive_sets(x_cpu.shape[0], edges_cpu, knn)
            if not positive_sets.valid_anchors:
                continue
            left, right = pair.views
            optimizer.zero_grad(set_to_none=True)
            z_left, z_right = model(left.x.to(device), left.edge_index.to(device), right.x.to(device), right.edge_index.to(device))
            loss = symmetric_multi_positive_loss(z_left, z_right, positive_sets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite SSL loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            valid_anchor_count += len(positive_sets.valid_anchors)
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                raise TimeoutError("SSL resource guard exceeded max_seconds")
        if not losses:
            raise ValueError("epoch has no valid anchor")
        epoch_losses.append(float(np.mean(losses)))
        anchors_by_epoch.append(valid_anchor_count)
        if checkpoint_hook is not None:
            checkpoint_hook(epoch + 1, model, optimizer)
    return PretrainingResult(tuple(epoch_losses), epochs, tuple(anchors_by_epoch))


def _train_candidate(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    fit_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    *, hidden: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[DownstreamMLP, float, float, float]:
    torch.manual_seed(seed)
    model = DownstreamMLP(inputs.shape[1], hidden)
    fit_labels = labels[fit_indices].float()
    positives = max(1, int(fit_labels.sum()))
    negatives = max(1, len(fit_labels) - positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best: tuple[float, float, float, dict[str, torch.Tensor]] | None = None
    stale = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs[fit_indices]), fit_labels)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(inputs[validation_indices])).numpy()
        expected = labels[validation_indices].numpy()
        candidates = []
        for threshold in (0.3, 0.5, 0.7):
            predicted = (scores >= threshold).astype(int)
            candidates.append((f1_score(expected, predicted, pos_label=1, zero_division=0), matthews_corrcoef(expected, predicted), threshold))
        current = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
        if best is None or current[:2] > best[:2]:
            best = (*current, {key: value.detach().clone() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best is not None
    model.load_state_dict(best[3])
    return model, best[2], best[0], best[1]


def search_downstream(
    embeddings: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    fit_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    epochs: int,
    patience: int,
    seed: int,
) -> DownstreamSelection:
    inputs = torch.cat((embeddings.detach(), features.detach()), dim=1)
    candidates = []
    for hidden in (64, 128):
        for learning_rate in (1e-3, 3e-4):
            for weight_decay in (1e-5, 1e-4):
                model, threshold, f1, mcc = _train_candidate(
                    inputs, labels, fit_indices, validation_indices,
                    hidden=hidden, learning_rate=learning_rate, weight_decay=weight_decay,
                    epochs=epochs, patience=patience, seed=seed,
                )
                candidates.append((f1, mcc, -hidden, -learning_rate, -weight_decay, model, threshold, hidden, learning_rate, weight_decay))
    best = max(candidates, key=lambda item: item[:5])
    frozen_marker = nn.Parameter(embeddings.detach().clone(), requires_grad=False)
    return DownstreamSelection(
        model=best[5], threshold=best[6], f1_illicit=best[0], mcc=best[1],
        hyperparameters={"hidden": best[7], "learning_rate": best[8], "weight_decay": best[9]},
        input_dim=inputs.shape[1], encoder_parameters=(frozen_marker,),
    )


def refit_downstream(
    selection: DownstreamSelection,
    embeddings: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    refit_indices: torch.Tensor,
    epochs: int,
    seed: int,
) -> RefitClassifier:
    torch.manual_seed(seed)
    inputs = torch.cat((embeddings.detach(), features.detach()), dim=1)
    hidden = int(selection.hyperparameters["hidden"])
    model = DownstreamMLP(inputs.shape[1], hidden)
    selected_labels = labels[refit_indices].float()
    positives = max(1, int(selected_labels.sum()))
    negatives = max(1, len(selected_labels) - positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(selection.hyperparameters["learning_rate"]),
        weight_decay=float(selection.hyperparameters["weight_decay"]),
    )
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs[refit_indices]), selected_labels)
        loss.backward()
        optimizer.step()
    return RefitClassifier(deepcopy(model), selection.threshold, inputs.shape[1])

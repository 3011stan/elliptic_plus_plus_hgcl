from __future__ import annotations

import pytest
import torch

from hgcl.studies.s003.models import S003ContrastiveModel, symmetric_multi_positive_loss
from hgcl.studies.s003.positives import PositiveSets
from hgcl.studies.s003.training import pretrain_ssl, search_downstream


def test_gin_2x128_shared_encoder_and_finite_gradients() -> None:
    model = S003ContrastiveModel(input_dim=182)
    assert model.encoder.layers == 2 and model.encoder.hidden_dim == 128
    x = torch.randn(4, 182)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    left, right = model(x, edge_index, x * 0.9, edge_index)
    sets = PositiveSets(
        positives=tuple(frozenset({i}) for i in range(4)),
        negatives=tuple(frozenset(set(range(4)) - {i}) for i in range(4)),
        valid_anchors=(0, 1, 2, 3),
    )
    loss = symmetric_multi_positive_loss(left, right, sets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_epoch_without_valid_anchor_is_invalid() -> None:
    z = torch.randn(1, 128)
    sets = PositiveSets((frozenset({0}),), (frozenset(),), ())
    with pytest.raises(ValueError, match="anchor"):
        symmetric_multi_positive_loss(z, z, sets)


def test_fixed_epoch_ssl_and_frozen_downstream_selection() -> None:
    torch.manual_seed(7)
    model = S003ContrastiveModel(input_dim=182)
    snapshots = [
        (1, tuple(str(i) for i in range(6)), torch.randn(6, 182), torch.tensor([[0, 1, 2], [1, 2, 3]]))
    ]
    result = pretrain_ssl(model, snapshots, epochs=2, seed=11, knn_k=1)
    assert len(result.losses) == 2
    assert result.epochs_completed == 2
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())

    embeddings = torch.randn(20, 128)
    features = torch.randn(20, 182)
    labels = torch.tensor([0, 1] * 10)
    selection = search_downstream(
        embeddings, features, labels,
        fit_indices=torch.arange(0, 12), validation_indices=torch.arange(12, 20),
        epochs=3, patience=2, seed=11,
    )
    assert selection.threshold in {0.3, 0.5, 0.7}
    assert selection.input_dim == 310
    assert not any(parameter.requires_grad for parameter in selection.encoder_parameters)

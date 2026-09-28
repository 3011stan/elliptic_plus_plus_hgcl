from __future__ import annotations

import pytest
import torch

from hgcl.studies.s003.baselines import AdapterContext, GraphData, SupervisedGNNAdapter
from hgcl.studies.s003.models import DirectedSupervisedGNN


@pytest.mark.parametrize(
    ("kind", "method_id"),
    [("gcn", "gcn_supervised"), ("graphsage", "graphsage_supervised"), ("gin", "gin_supervised")],
)
def test_supervised_gnn_is_directed_2x128_and_trainable(kind: str, method_id: str) -> None:
    torch.manual_seed(11)
    count = 20
    ids = tuple(f"tx-{index}" for index in range(count))
    labels = torch.tensor([0, 1] * 10)
    features = torch.randn(count, 182)
    features[:, 0] = labels.float() * 4 - 2
    edges = torch.tensor([list(range(count - 1)), list(range(1, count))])
    data = GraphData(ids, features, labels, edges)
    context = AdapterContext(
        11,
        0.01,
        frozenset(ids[:12]),
        frozenset(ids[12:16]),
        frozenset(ids[:16]),
        "a" * 64,
    )
    adapter = SupervisedGNNAdapter(kind, epochs=2)
    fitted = adapter.fit(data, context)
    scores = adapter.predict_scores(fitted, data)[0]

    assert fitted.method_id == method_id
    assert fitted.hyperparameters["layers"] == 2
    assert fitted.hyperparameters["hidden"] == 128
    assert scores.shape == (count,)
    assert all(layer.flow == "source_to_target" for layer in fitted.estimator.convolutions)


def test_supervised_gnn_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        DirectedSupervisedGNN("gat")

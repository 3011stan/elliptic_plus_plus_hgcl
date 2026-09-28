from __future__ import annotations

import torch

from hgcl.studies.s003.augmentations import make_two_stochastic_views
from hgcl.studies.s003.baselines import AdapterContext, GCPALAdapter, GraphData, InspectionLDGIAdapter


def _graph() -> tuple[GraphData, AdapterContext]:
    torch.manual_seed(11)
    ids = tuple(f"tx-{i}" for i in range(20))
    labels = torch.tensor([0, 1] * 10)
    x = torch.randn(20, 182)
    edges = torch.tensor([list(range(19)), list(range(1, 20))])
    return GraphData(ids, x, labels, edges), AdapterContext(11, 0.01, frozenset(ids[:12]), frozenset(ids[12:16]), frozenset(ids[:16]), "a" * 64)


def test_gcpal_uses_two_stochastic_views() -> None:
    data, _ = _graph()
    pair = make_two_stochastic_views(data.features, data.edge_index, seed=11)
    assert pair.names == ("stochastic_1", "stochastic_2")
    assert pair.views[0].x.shape == pair.views[1].x.shape == (20, 182)


def test_inspection_l_preserves_gin_dgi_rf100_and_declares_migration() -> None:
    data, context = _graph()
    fitted = InspectionLDGIAdapter(ssl_epochs=1, rf_estimators=10).fit(data, context)
    assert fitted.method_id == "inspection_l_dgi"
    assert fitted.encoder.layers == 2 and fitted.encoder.hidden_dim == 128
    assert fitted.downstream.estimator.n_estimators == 10
    assert "182 features" in fitted.migration_notes[0]


def test_gcpal_preserves_knn_multi_positive_and_h_concat_x() -> None:
    data, context = _graph()
    adapter = GCPALAdapter(ssl_epochs=1, downstream_epochs=1, knn_k=10)
    fitted = adapter.fit(data, context)
    assert fitted.method_id == "gcpal"
    assert fitted.encoder.encoder.layers == 2
    assert fitted.downstream.estimator.input_dim == 310
    assert adapter.characteristic_components == frozenset({"gin_2x128", "two_stochastic_views", "knn_k10", "multi_positive_loss", "mlp_2layer_h_concat_x"})

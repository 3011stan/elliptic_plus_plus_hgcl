from __future__ import annotations

import pytest
import torch

from hgcl.studies.s003.augmentations import masking_columns, maybe_add_reverse_edges
from hgcl.studies.s003.baselines import AdapterContext, GraphData, S003TxGCLAdapter


def _data():
    ids = tuple(f"tx-{i}" for i in range(20)); labels = torch.tensor([0, 1] * 10)
    data = GraphData(ids, torch.randn(20, 182), labels, torch.tensor([list(range(19)), list(range(1, 20))]))
    context = AdapterContext(11, 0.01, frozenset(ids[:12]), frozenset(ids[12:16]), frozenset(ids[:16]), "a" * 64)
    return data, context


@pytest.mark.parametrize("policy", ["functional_blocks", "random_groups", "random_individual"])
def test_masking_controls_have_exact_block_cardinality(policy: str) -> None:
    assert len(masking_columns(policy, seed=11)) in {93, 72, 17}


def test_reverse_edges_are_explicitly_gated() -> None:
    edges = torch.tensor([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="approval"):
        maybe_add_reverse_edges(edges, enabled=True, approved=False)
    assert maybe_add_reverse_edges(edges, enabled=True, approved=True).shape[1] == 4


@pytest.mark.parametrize(("representation", "width"), [("x_only", 182), ("h_only", 128), ("h_concat_x", 310)])
def test_s003_representations_and_primary_adapter(representation: str, width: int) -> None:
    data, context = _data()
    fitted = S003TxGCLAdapter(ssl_epochs=1, downstream_epochs=1, representation=representation).fit(data, context)
    assert fitted.downstream.estimator.input_dim == width
    assert fitted.downstream.hyperparameters["representation"] == representation

from __future__ import annotations

import torch

from hgcl.studies.s003.augmentations import make_contrastive_views


def test_exactly_two_deterministic_representations_and_protected_metadata() -> None:
    x = torch.ones((5, 182))
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    pair = make_contrastive_views(x, edges, seed=11, feature_drop=0.1, edge_drop=0.1)
    same = make_contrastive_views(x, edges, seed=11, feature_drop=0.1, edge_drop=0.1)
    assert pair.names == ("stochastic", "functional_blocks")
    assert len(pair.views) == 2
    assert torch.equal(pair.views[0].x, same.views[0].x)
    assert pair.views[1].masked_features in {17, 72, 93}
    assert pair.views[0].x.shape[1] == 182


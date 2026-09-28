from __future__ import annotations

import torch

from hgcl.studies.s003.positives import build_positive_sets, cosine_knn


def test_snapshot_local_knn_successors_and_disjoint_negatives() -> None:
    x = torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1.0], [-1.0, 0]])
    ids = ("d", "c", "b", "a")
    knn = cosine_knn(x, ids, k=10)
    assert all(len(row) == 3 for row in knn)
    sets = build_positive_sets(4, torch.tensor([[0, 2], [1, 0]]), knn)
    assert 1 in sets.positives[0]
    assert 2 not in sets.positives[0] or 2 in knn[0]
    for positive, negative in zip(sets.positives, sets.negatives):
        assert positive.isdisjoint(negative)
    assert sets.valid_anchors == tuple(i for i, n in enumerate(sets.negatives) if n)


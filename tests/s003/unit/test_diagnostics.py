from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from hgcl.studies.s003.diagnostics import (
    aggregate_diagnostics_across_seeds,
    aggregate_diagnostics_across_snapshots,
    alignment,
    compute_snapshot_diagnostics,
    effective_rank,
    uniformity,
)


def test_alignment_on_identical_and_orthogonal_pairs() -> None:
    # 1. Identical representations on positive pairs have alignment == 0.0
    u = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    pairs = torch.tensor([[0, 2], [1, 3]])  # 0 paired with 1, 2 paired with 3
    align = alignment(u, pairs)
    assert pytest.approx(align, abs=1e-6) == 0.0

    # 2. Orthogonal representations on positive pairs have distance squared == 2.0
    pairs_ortho = torch.tensor([[0], [2]])  # [1,0] paired with [0,1]
    align_ortho = alignment(u, pairs_ortho)
    assert pytest.approx(align_ortho, abs=1e-5) == 2.0

    # 3. Empty positive pairs or empty embeddings return 0.0
    assert alignment(torch.empty((0, 2)), pairs) == 0.0
    assert alignment(u, torch.empty((2, 0), dtype=torch.long)) == 0.0

    # 4. Self pairs (i == j) are filtered out
    self_pairs = torch.tensor([[0, 1], [0, 1]])
    assert alignment(u, self_pairs) == 0.0


def test_uniformity_on_collapsed_and_dispersed_representations() -> None:
    # 1. Perfectly collapsed representations (all points identical)
    collapsed = torch.ones((10, 16))
    unif_collapsed = uniformity(collapsed)
    # When all points are identical, ||u_i - u_j|| = 0 for all i != j.
    # exp(-2 * 0) = 1.0, log(1.0) = 0.0
    assert pytest.approx(unif_collapsed, abs=1e-5) == 0.0

    # 2. Orthogonal / dispersed representations (points spread out)
    # Unit basis vectors in 4D
    basis = torch.eye(4)
    unif_basis = uniformity(basis)
    # For any i != j, ||e_i - e_j||^2 = 2. exp(-2 * 2) = exp(-4). log(exp(-4)) = -4.0
    assert pytest.approx(unif_basis, abs=1e-5) == -4.0
    assert unif_basis < unif_collapsed

    # 3. Degenerate inputs (N < 2) return 0.0
    assert uniformity(torch.randn(1, 16)) == 0.0
    assert uniformity(torch.empty((0, 16))) == 0.0


def test_effective_rank_on_full_rank_and_rank_one() -> None:
    # 1. Rank-1 matrix: all rows identical
    rank_one = torch.ones((20, 10))
    erank_1 = effective_rank(rank_one)
    assert pytest.approx(erank_1, abs=1e-5) == 1.0

    # 2. Orthogonal isotropic matrix (e.g. eye(10) with 10 equal singular values)
    orthogonal = torch.eye(10)
    erank_ortho = effective_rank(orthogonal)
    # All 10 singular values are 1.0, p_k = 1/10, entropy = -10 * (1/10 * ln(1/10)) = ln(10)
    # exp(ln(10)) = 10.0
    assert pytest.approx(erank_ortho, abs=1e-5) == 10.0

    # 3. Random Gaussian matrix (20x10): effective rank should be between 1.0 and 10.0
    torch.manual_seed(42)
    random_mat = torch.randn(50, 10)
    erank_rnd = effective_rank(random_mat)
    assert 1.0 <= erank_rnd <= 10.0

    # 4. Zero matrix or empty returns 0.0
    assert effective_rank(torch.zeros((10, 10))) == 0.0
    assert effective_rank(torch.empty((0, 10))) == 0.0


def test_snapshot_and_cross_snapshot_aggregation() -> None:
    embeddings = torch.randn(25, 128)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    diag = compute_snapshot_diagnostics(embeddings, edge_index)

    assert "alignment" in diag
    assert "uniformity" in diag
    assert "effective_rank" in diag
    assert diag["anchor_count"] == 25
    assert 1.0 <= diag["effective_rank"] <= 128.0

    # Cross-snapshot aggregation weighted by anchor count
    snapshots = {
        35: {"alignment": 0.5, "uniformity": -2.0, "effective_rank": 8.0, "anchor_count": 100},
        36: {"alignment": 1.0, "uniformity": -1.0, "effective_rank": 12.0, "anchor_count": 300},
    }
    agg = aggregate_diagnostics_across_snapshots(snapshots)
    # weights: 100/400 = 0.25 and 300/400 = 0.75
    assert pytest.approx(agg["alignment"], abs=1e-5) == 0.25 * 0.5 + 0.75 * 1.0
    assert pytest.approx(agg["uniformity"], abs=1e-5) == 0.25 * -2.0 + 0.75 * -1.0
    assert pytest.approx(agg["effective_rank"], abs=1e-5) == 0.25 * 8.0 + 0.75 * 12.0
    assert agg["total_anchors"] == 400


def test_cross_seed_aggregation() -> None:
    seed_summaries = {
        11: {"alignment": 0.8, "uniformity": -1.5, "effective_rank": 10.0},
        23: {"alignment": 0.9, "uniformity": -1.6, "effective_rank": 10.5},
        37: {"alignment": 0.85, "uniformity": -1.55, "effective_rank": 10.2},
    }
    cross_seed = aggregate_diagnostics_across_seeds(seed_summaries)
    assert pytest.approx(cross_seed["alignment"]["mean"], abs=1e-5) == 0.85
    assert cross_seed["alignment"]["std"] > 0.0
    assert pytest.approx(cross_seed["effective_rank"]["mean"], abs=1e-5) == (10.0 + 10.5 + 10.2) / 3.0

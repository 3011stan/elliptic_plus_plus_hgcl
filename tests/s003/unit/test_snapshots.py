from __future__ import annotations

import torch

from hgcl.studies.s003.data import build_snapshots, discover_source, read_source


def test_directed_snapshot_mapping_and_edge_accounting(s003_fixture_root) -> None:
    data = read_source(discover_source(s003_fixture_root))
    snapshots = build_snapshots(data, steps=range(1, 4))
    first = snapshots[0]
    assert first.message_flow == "source_to_target"
    assert first.tx_ids == tuple(sorted(first.tx_ids))
    assert torch.equal(first.edge_index, torch.tensor([[0], [1]]))
    assert first.manifest["duplicate_edges_removed"] == 0
    assert first.manifest["self_loops_removed"] == 0
    assert snapshots[2].edge_index.shape == (2, 0)
    assert snapshots[2].x.shape == (0, 182)


def test_duplicate_and_self_loop_edges_are_removed(s003_fixture_root) -> None:
    data = read_source(discover_source(s003_fixture_root))
    data.edges.extend([("1001", "1002"), ("1001", "1001")])
    first = build_snapshots(data, steps=[1])[0]
    assert first.manifest["duplicate_edges_removed"] == 1
    assert first.manifest["self_loops_removed"] == 1

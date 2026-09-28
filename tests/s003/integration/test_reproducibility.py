from __future__ import annotations

import torch

from hgcl.studies.s003.config import load_config
from hgcl.studies.s003.data import build_snapshots, discover_source, read_source
from hgcl.studies.s003.evaluation import decisions_from_scores
from hgcl.studies.s003.pipeline import load_prepared, persist_prepared, prepare_dataset


def test_manifests_scores_and_classes_reload_deterministically(s003_fixture_root) -> None:
    source = discover_source(s003_fixture_root)
    first = build_snapshots(read_source(source), steps=range(1, 4))
    second = build_snapshots(read_source(source), steps=range(1, 4))
    assert [item.digest for item in first] == [item.digest for item in second]
    scores = torch.tensor([0.1, 0.500001, 0.9])
    reloaded = scores + torch.tensor([1e-7, -1e-7, 1e-7])
    assert torch.allclose(scores, reloaded, rtol=1e-5, atol=2e-6)
    assert decisions_from_scores(scores, 0.5).tolist() == decisions_from_scores(reloaded, 0.5).tolist()


def test_prepared_artifacts_reload_with_exact_manifests(tmp_path, s003_fixture_root) -> None:
    config = load_config("configs/s003/smoke.yaml")
    evidence = {name: {"causal": True, "source": "fixture-v1"} for name in ("local", "aggregate", "augmented")}
    prepared = prepare_dataset(config, data_root=s003_fixture_root, causal_evidence=evidence)
    path = persist_prepared(prepared, config, artifact_root=tmp_path / "artifacts" / "s003")
    reloaded = load_prepared(path, config)
    assert reloaded.data_digest == prepared.data_digest
    assert [item.digest for item in reloaded.snapshots] == [item.digest for item in prepared.snapshots]
    assert reloaded.test_labels.labels is None

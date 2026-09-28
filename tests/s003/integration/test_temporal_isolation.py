from __future__ import annotations

from hgcl.studies.s003.config import load_config
import torch

from hgcl.studies.s003.data import PreparedSnapshot
from hgcl.studies.s003.pipeline import dry_run_contract, prepare_dataset, run_shadow_dry_run


def test_profile_contract_never_opens_real_test_labels() -> None:
    smoke = dry_run_contract(load_config("configs/s003/smoke.yaml"), original_data=True)
    dry = dry_run_contract(load_config("configs/s003/dry-run.yaml"), original_data=True)
    assert smoke["max_nodes_per_snapshot"] == 256
    assert smoke["ssl_epochs"] == 2 and smoke["downstream_epochs"] == 3
    assert dry["max_nodes_per_snapshot"] is None
    assert dry["ssl_epochs"] == 10 and dry["downstream_epochs"] == 20
    assert smoke["snapshot_batch_size"] == dry["snapshot_batch_size"] == 1
    assert smoke["test_labels_opened"] is False


def test_prepare_seals_test_labels_and_samples_by_stable_hash(s003_fixture_root) -> None:
    prepared = prepare_dataset(
        load_config("configs/s003/smoke.yaml"),
        data_root=s003_fixture_root,
        causal_evidence={name: {"causal": True, "source": "fixture-dictionary-v1"} for name in ("local", "aggregate", "augmented")},
    )
    assert len(prepared.snapshots) == 49
    assert all(snapshot.labels is None for snapshot in prepared.snapshots if snapshot.time_step >= 35)
    assert prepared.test_labels.labels is None
    assert prepared.feature_audit.passed
    assert prepared.test_access_count == 0


def test_end_to_end_shadow_training_uses_only_steps_1_to_34() -> None:
    torch.manual_seed(11)
    snapshots = []
    for step in (1, 2, 30, 31):
        count = 20
        ids = tuple(f"{step}-{index}" for index in range(count))
        x = torch.randn(count, 182)
        edges = torch.tensor([list(range(count - 1)), list(range(1, count))])
        labels = torch.tensor([index % 2 for index in range(count)])
        snapshots.append(PreparedSnapshot(step, ids, x, edges, labels, {"time_step": step}, str(step) * 64))
    result = run_shadow_dry_run(load_config("configs/s003/smoke.yaml"), snapshots, device="cpu")
    assert result["epochs"] == {"ssl": 2, "downstream": 3}
    assert result["fit_steps"] == [1, 2]
    assert result["shadow_steps"] == [30, 31]
    assert result["test_labels_opened"] is False
    assert result["metrics"]["pooling"] == "concatenated_known_predictions"

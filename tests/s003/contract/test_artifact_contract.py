from __future__ import annotations

import json
from pathlib import Path

import pytest

from hgcl.studies.s003.artifacts import (
    ArtifactError,
    ArtifactStore,
    DryRunApproval,
    EvaluationCell,
    EvaluationCohort,
    compute_evidence_digest,
)


def _digest(char: str = "a") -> str:
    return char * 64


def test_atomic_json_and_binary_registration_is_relative_and_immutable(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts" / "s003")
    path = store.write_json("runs/s003-run-001/run.json", {"state": "planned"})
    assert json.loads(path.read_text())["state"] == "planned"
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(ArtifactError, match="exists"):
        store.write_json("runs/s003-run-001/run.json", {"state": "running"})
    parquet = store.write_bytes("prepared/test/nodes.parquet", b"PAR1fixture")
    assert parquet.read_bytes() == b"PAR1fixture"
    with pytest.raises(ArtifactError):
        store.write_json("../escape.json", {})


def test_common_envelope_and_file_hashes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts" / "s003")
    envelope = store.envelope(
        artifact_type="run",
        config_digest=_digest("a"),
        data_digest=_digest("b"),
        code_revision={"commit": "c" * 40, "dirty": False},
        payload={"run_id": "s003-run-001"},
    )
    assert envelope["schema_version"] == 1
    assert envelope["study_id"] == "s003"
    path = store.write_json("runs/s003-run-001/envelope.json", envelope)
    assert store.file_record(path)["sha256"] == store.sha256(path)


def test_dry_run_approval_is_digest_bound() -> None:
    approval = DryRunApproval(
        approval_id="s003-approval-001",
        approved_by="researcher",
        data_digest=_digest("a"),
        dry_run_config_digest=_digest("b"),
        lab_config_digest=_digest("c"),
        code_revision="d" * 40,
        evidence_digest=_digest("e"),
        design_digest=_digest("f"),
        max_projected_duration_seconds=3600,
    )
    approval.assert_compatible(
        data_digest=_digest("a"),
        lab_config_digest=_digest("c"),
        code_revision="d" * 40,
        design_digest=_digest("f"),
    )
    with pytest.raises(ArtifactError, match="lab_config_digest"):
        approval.assert_compatible(
            data_digest=_digest("a"),
            lab_config_digest=_digest("0"),
            code_revision="d" * 40,
            design_digest=_digest("f"),
        )


def test_dry_run_approval_reverse_edge_ablation_and_validation() -> None:
    # 1. Reverse edge ablation defaults to False
    approval = DryRunApproval(
        approval_id="s003-approval-001",
        approved_by="researcher",
        data_digest=_digest("a"),
        dry_run_config_digest=_digest("b"),
        lab_config_digest=_digest("c"),
        code_revision="d" * 40,
        evidence_digest=_digest("e"),
        design_digest=_digest("f"),
        max_projected_duration_seconds=3600,
    )
    assert approval.reverse_edge_ablation is False

    # 2. Compatibility check rejects when reverse edge ablation is required but not approved
    with pytest.raises(ArtifactError, match="reverse_edge_ablation is not approved"):
        approval.assert_compatible(
            data_digest=_digest("a"),
            lab_config_digest=_digest("c"),
            code_revision="d" * 40,
            design_digest=_digest("f"),
            require_reverse_edge_ablation=True,
        )

    # 3. Approved reverse edge ablation passes compatibility
    approved_p2 = DryRunApproval(
        approval_id="s003-approval-002",
        approved_by="researcher",
        data_digest=_digest("a"),
        dry_run_config_digest=_digest("b"),
        lab_config_digest=_digest("c"),
        code_revision="d" * 40,
        evidence_digest=_digest("e"),
        design_digest=_digest("f"),
        max_projected_duration_seconds=3600,
        reverse_edge_ablation=True,
    )
    approved_p2.assert_compatible(
        data_digest=_digest("a"),
        lab_config_digest=_digest("c"),
        code_revision="d" * 40,
        design_digest=_digest("f"),
        require_reverse_edge_ablation=True,
    )

    # 4. Empty approved_by is rejected
    with pytest.raises(ArtifactError, match="approved_by cannot be empty"):
        DryRunApproval(
            approval_id="s003-approval-003",
            approved_by="",
            data_digest=_digest("a"),
            dry_run_config_digest=_digest("b"),
            lab_config_digest=_digest("c"),
            code_revision="d" * 40,
            evidence_digest=_digest("e"),
            design_digest=_digest("f"),
            max_projected_duration_seconds=3600,
        )


def test_dry_run_approval_serialization_round_trip() -> None:
    approval = DryRunApproval(
        approval_id="s003-approval-001",
        approved_by="researcher",
        data_digest=_digest("a"),
        dry_run_config_digest=_digest("b"),
        lab_config_digest=_digest("c"),
        code_revision="d" * 40,
        evidence_digest=_digest("e"),
        design_digest=_digest("f"),
        max_projected_duration_seconds=7200,
        reverse_edge_ablation=True,
    )
    payload = approval.to_dict()
    restored = DryRunApproval.from_dict(payload)
    assert restored.approval_id == approval.approval_id
    assert restored.approved_by == approval.approved_by
    assert restored.data_digest == approval.data_digest
    assert restored.dry_run_config_digest == approval.dry_run_config_digest
    assert restored.lab_config_digest == approval.lab_config_digest
    assert restored.code_revision == approval.code_revision
    assert restored.evidence_digest == approval.evidence_digest
    assert restored.design_digest == approval.design_digest
    assert restored.max_projected_duration_seconds == approval.max_projected_duration_seconds
    assert restored.reverse_edge_ablation is True


def test_compute_evidence_digest() -> None:
    payload1 = {"report": "valid", "p1_cells": 205}
    payload2 = {"p1_cells": 205, "report": "valid"}  # Key order difference
    payload3 = {"report": "valid", "p1_cells": 204}

    digest1 = compute_evidence_digest(payload1)
    digest2 = compute_evidence_digest(payload2)
    digest3 = compute_evidence_digest(payload3)

    assert len(digest1) == 64
    assert digest1 == digest2  # Canonical order invariant
    assert digest1 != digest3


def test_evaluation_cohort_seals_205_cells_and_releases_once() -> None:
    cells = [EvaluationCell(key=f"cell-{i:03d}", state="failed") for i in range(205)]
    cells[0] = EvaluationCell(
        key="cell-000",
        state="selected",
        run_id="s003-run-000",
        weights_digest=_digest("a"),
        threshold=0.5,
        config_digest=_digest("b"),
    )
    cohort = EvaluationCohort(
        cohort_id="s003-cohort-001",
        design_digest=_digest("c"),
        test_store_digest=_digest("d"),
        cells=tuple(cells),
    )
    cohort.seal()
    cohort.release()
    with pytest.raises(ArtifactError, match="released"):
        cohort.release()
    with pytest.raises(ArtifactError, match="sealed"):
        cohort.add_cell(EvaluationCell(key="extra", state="failed"))

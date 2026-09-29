from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml

from hgcl.studies.s003.artifacts import (
    ArtifactError,
    DryRunApproval,
    compute_evidence_digest,
)
from hgcl.studies.s003.baselines import GraphData
from hgcl.studies.s003.cli import run
from hgcl.studies.s003.config import load_config
from hgcl.studies.s003.data import PreparedSnapshot
from hgcl.studies.s003.evaluation import TestLabelStore
from hgcl.studies.s003.pipeline import (
    PreparedDataset,
    approve_dry_run,
    find_compatible_approval,
    generate_matrix_report,
    run_matrix_pipeline,
)


def _setup_dry_run_fixture(tmp_path: Path, *, training_performed: bool = False) -> tuple[Path, dict, str]:
    run_dir = tmp_path / "artifacts" / "s003" / "runs" / "s003-dry-001"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_payload = {
        "study_id": "s003",
        "run_id": "s003-dry-001",
        "profile": "dry-run" if not training_performed else "smoke",
        "config_digest": "1" * 64,
        "data_digest": "2" * 64,
        "design_digest": "3" * 64,
        "training_performed": training_performed,
        "test_labels_materialized": False,
        "expected_p1_cells": 205,
        "cells": [f"cell-{i}" for i in range(205)],
    }
    evidence_digest = compute_evidence_digest(run_payload)
    run_file = run_dir / "run.json"
    run_file.write_text(json.dumps({"payload": run_payload}), encoding="utf-8")
    return run_dir, run_payload, evidence_digest


def test_approve_dry_run_success_and_digest_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, run_payload, evidence_digest = _setup_dry_run_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    approval_file = tmp_path / "approval-input.json"
    approval_data = {
        "approval_id": "s003-app-001",
        "approved_by": "researcher",
        "data_digest": "2" * 64,
        "dry_run_config_digest": "1" * 64,
        "lab_config_digest": "4" * 64,
        "code_revision": "0" * 40,
        "evidence_digest": evidence_digest,
        "design_digest": "3" * 64,
        "max_projected_duration_seconds": 7200,
        "reverse_edge_ablation": False,
    }
    approval_file.write_text(json.dumps(approval_data), encoding="utf-8")

    code = run(["approve-dry-run", "--run", str(run_dir), "--approval-file", str(approval_file)])
    assert code == 0

    saved_approval = run_dir / "approval.json"
    assert saved_approval.is_file()
    payload = json.loads(saved_approval.read_text(encoding="utf-8"))["payload"]
    assert payload["approval_id"] == "s003-app-001"
    assert payload["evidence_digest"] == evidence_digest


def test_approve_dry_run_fail_closed_on_digest_mismatch(tmp_path: Path) -> None:
    run_dir, run_payload, evidence_digest = _setup_dry_run_fixture(tmp_path)

    # 1. Corrupted evidence digest fails closed
    bad_evidence_file = tmp_path / "bad-evidence.json"
    bad_data = {
        "approval_id": "s003-app-002",
        "approved_by": "researcher",
        "data_digest": "2" * 64,
        "dry_run_config_digest": "1" * 64,
        "lab_config_digest": "4" * 64,
        "code_revision": "0" * 40,
        "evidence_digest": "f" * 64,  # Incorrect
        "design_digest": "3" * 64,
        "max_projected_duration_seconds": 7200,
    }
    bad_evidence_file.write_text(json.dumps(bad_data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="evidence_digest"):
        approve_dry_run(run_dir, bad_evidence_file)

    # 2. Corrupted design digest fails closed
    bad_design_file = tmp_path / "bad-design.json"
    bad_data["evidence_digest"] = evidence_digest
    bad_data["design_digest"] = "f" * 64
    bad_design_file.write_text(json.dumps(bad_data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="design_digest"):
        approve_dry_run(run_dir, bad_design_file)

    # 3. Attempting to approve a run with training_performed=True fails closed
    smoke_dir, _, smoke_ev = _setup_dry_run_fixture(tmp_path / "smoke", training_performed=True)
    smoke_approval_file = tmp_path / "smoke-approval.json"
    bad_data["design_digest"] = "3" * 64
    bad_data["evidence_digest"] = smoke_ev
    smoke_approval_file.write_text(json.dumps(bad_data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="dry-run cannot have performed training|only a dry-run"):
        approve_dry_run(smoke_dir, smoke_approval_file)


def test_matrix_blocked_without_compatible_approval(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "s003"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ArtifactError, match="matrix execution blocked: no compatible approved dry-run found"):
        find_compatible_approval(
            artifacts_root,
            data_digest="2" * 64,
            lab_config_digest="4" * 64,
            code_revision="0" * 40,
            design_digest="3" * 64,
        )


def test_matrix_and_report_with_tiny_non_scientific_fixture(tmp_path: Path) -> None:
    # 1. Setup minimal synthetic dataset
    ids = tuple(f"tx-{i}" for i in range(20))
    features = torch.randn(20, 182)
    labels = torch.tensor([0, 1] * 10)
    edge_index = torch.tensor([list(range(19)), list(range(1, 20))], dtype=torch.long)
    snap = PreparedSnapshot(1, ids, features, edge_index, labels, {}, "a" * 64)
    test_store = TestLabelStore.seal({35: {"tx-test": 1}})
    prep = PreparedDataset(
        source=SimpleNamespace(data_digest="2" * 64),
        snapshots=(snap,),
        training_snapshots=(snap,),
        preprocessing=SimpleNamespace(),
        feature_audit=SimpleNamespace(),
        test_labels=test_store,
        data_digest="2" * 64,
    )

    # 2. Setup dry-run approval
    run_dir, run_payload, evidence_digest = _setup_dry_run_fixture(tmp_path)
    approval_file = tmp_path / "approval.json"
    approval_data = {
        "approval_id": "s003-app-test",
        "approved_by": "researcher",
        "data_digest": "2" * 64,
        "dry_run_config_digest": "1" * 64,
        "lab_config_digest": "4" * 64,
        "code_revision": "0" * 40,
        "evidence_digest": evidence_digest,
        "design_digest": "3" * 64,
        "max_projected_duration_seconds": 3600,
    }
    approval_file.write_text(json.dumps(approval_data), encoding="utf-8")
    approve_dry_run(run_dir, approval_file)

    # 3. Create mock config
    mock_config = SimpleNamespace(
        profile="lab",
        digest="4" * 64,
        code_revision="0" * 40,
        raw={
            "baselines": {"methods": ["mlp_x", "s003_txgcl"]},
            "labels": {"fractions": [0.01], "seeds": [11]},
            "paths": {"artifacts_root": str(tmp_path / "artifacts" / "s003")},
        },
    )

    # 4. Run matrix pipeline with small cell override
    matrix_res = run_matrix_pipeline(
        mock_config,
        prep,
        matrix_id="s003-matrix-tiny",
        artifacts_root=tmp_path / "artifacts" / "s003",
        cells_override=["main:mlp_x:0.01:11", "main:s003_txgcl:0.01:11"],
        execute_cells=True,
        ssl_epochs=1,
        downstream_epochs=1,
    )
    assert matrix_res["status"] == "complete"
    assert matrix_res["cells_count"] == 2
    assert matrix_res["approval_id"] == "s003-app-test"

    # Invariant: test labels are never accessed
    assert prep.test_labels.access_log == []
    assert prep.test_labels.released_cohort is None

    # 5. Run report generation on the matrix output
    matrix_dir = tmp_path / "artifacts" / "s003" / "matrices" / "s003-matrix-tiny"
    report_res = generate_matrix_report(matrix_dir)
    assert report_res["status"] == "complete"
    assert report_res["accounting"]["total_cells"] == 2
    assert report_res["accounting"]["selected"] == 2

    report_json = matrix_dir / "report.json"
    assert report_json.is_file()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["matrix_id"] == "s003-matrix-tiny"
    assert payload["accounting"]["selected"] == 2


def test_approve_dry_run_auto_without_approval_file(tmp_path: Path) -> None:
    run_dir, run_payload, evidence_digest = _setup_dry_run_fixture(tmp_path)
    res = approve_dry_run(
        run_dir,
        approval_file=None,
        approved_by="stan",
        lab_config_path=Path(__file__).parents[3] / "configs" / "s003" / "lab.yaml",
    )
    assert res["status"] == "complete"
    assert res["approved"] is True
    assert (run_dir / "approval.json").is_file()
    approval_doc = json.loads((run_dir / "approval.json").read_text(encoding="utf-8"))
    assert approval_doc["payload"]["approved_by"] == "stan"
    assert approval_doc["payload"]["data_digest"] == run_payload["data_digest"]

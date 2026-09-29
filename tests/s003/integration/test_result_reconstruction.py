from __future__ import annotations

import json
from pathlib import Path
import pytest

from hgcl.studies.s003.cli import run
from hgcl.studies.s003.pipeline import generate_matrix_report, resume_run
from hgcl.studies.s003.provenance import (
    EnvironmentManifest,
    LineageReconstruction,
    ProvenanceError,
    RunCheckpoint,
    reconstruct_result_lineage,
)


def _digest(val: str = "a") -> str:
    import hashlib
    return hashlib.sha256(val.encode("utf-8")).hexdigest()


def test_row_level_lineage_and_coverage_in_report(tmp_path: Path) -> None:
    """Test row-level lineage and explicit incomplete-state coverage in generate_matrix_report (FR-026, FR-027, SC-001)."""
    matrix_dir = tmp_path / "artifacts" / "s003" / "matrices" / "s003-matrix-test"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    data_hash = _digest("data")
    config_hash = _digest("config")
    weights_0 = _digest("weights-0")
    weights_1 = _digest("weights-1")

    # Create matrix.json with 4 cells: selected, failed, invalid, interrupted
    cells = [
        {
            "key": "main:s003_txgcl:0.01:11",
            "state": "selected",
            "run_id": "s003-run-001",
            "weights_digest": weights_0,
            "threshold": 0.45,
            "config_digest": config_hash,
        },
        {
            "key": "main:mlp_x:0.01:11",
            "state": "selected",
            "run_id": "s003-run-002",
            "weights_digest": weights_1,
            "threshold": 0.50,
            "config_digest": config_hash,
        },
        {
            "key": "main:gcn_supervised:0.01:11",
            "state": "failed",
            "run_id": "s003-run-003",
            "weights_digest": None,
            "threshold": None,
            "config_digest": config_hash,
            "failure_kind": "runtime",
            "failure_message": "CUDA out of memory",
        },
        {
            "key": "main:xgboost:0.01:11",
            "state": "invalid",
            "run_id": "s003-run-004",
            "weights_digest": None,
            "threshold": None,
            "config_digest": config_hash,
            "failure_kind": "data",
            "failure_message": "Single class present in sample",
        },
        {
            "key": "representation:h_only:0.01:11",
            "state": "interrupted",
            "run_id": "s003-run-005",
            "weights_digest": None,
            "threshold": None,
            "config_digest": config_hash,
            "failure_kind": "preemption",
            "failure_message": "Worker interrupted by scheduler",
        },
    ]

    matrix_file = matrix_dir / "matrix.json"
    matrix_file.write_text(
        json.dumps({
            "schema_version": 1,
            "study_id": "s003",
            "matrix_id": "s003-matrix-test",
            "design_digest": _digest("design"),
            "data_digest": data_hash,
            "cells": cells,
        }),
        encoding="utf-8",
    )

    # Create evaluations.json for the two selected cells
    evaluations_file = matrix_dir / "evaluations.json"
    evaluations_file.write_text(
        json.dumps({
            "s003-run-001": {
                "test_labels_digest": _digest("test-labels"),
                "scores_digest": _digest("scores-001"),
                "pooled": {"f1_illicit": 0.78, "mcc": 0.72},
            },
            "s003-run-002": {
                "test_labels_digest": _digest("test-labels"),
                "scores_digest": _digest("scores-002"),
                "pooled": {"f1_illicit": 0.65, "mcc": 0.60},
            },
        }),
        encoding="utf-8",
    )

    # Generate report
    report_res = generate_matrix_report(matrix_dir)
    assert report_res["status"] == "complete"

    # Verify coverage.json
    coverage_file = matrix_dir / "coverage.json"
    assert coverage_file.is_file()
    cov_data = json.loads(coverage_file.read_text(encoding="utf-8"))
    assert cov_data["total_expected"] == 5
    assert cov_data["accounting"]["selected"] == 2
    assert cov_data["accounting"]["failed"] == 1
    assert cov_data["accounting"]["invalid"] == 1
    assert cov_data["accounting"]["interrupted"] == 1
    assert len(cov_data["cells"]) == 5

    # Verify failure reasons are explicitly preserved
    failed_cell = next(c for c in cov_data["cells"] if c["key"] == "main:gcn_supervised:0.01:11")
    assert failed_cell["state"] == "failed"
    assert failed_cell["failure_kind"] == "runtime"
    assert "CUDA out of memory" in failed_cell["failure_message"]

    invalid_cell = next(c for c in cov_data["cells"] if c["key"] == "main:xgboost:0.01:11")
    assert invalid_cell["state"] == "invalid"
    assert invalid_cell["failure_kind"] == "data"

    interrupted_cell = next(c for c in cov_data["cells"] if c["key"] == "representation:h_only:0.01:11")
    assert interrupted_cell["state"] == "interrupted"

    # Verify report.json row_lineage
    report_file = matrix_dir / "report.json"
    assert report_file.is_file()
    rep_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert len(rep_data["row_lineage"]) == 5

    row_0 = next(r for r in rep_data["row_lineage"] if r["run_id"] == "s003-run-001")
    assert row_0["cell_key"] == "main:s003_txgcl:0.01:11"
    assert row_0["weights_digest"] == weights_0
    assert row_0["threshold"] == 0.45
    assert row_0["metrics"]["pooled"]["f1_illicit"] == 0.78


def test_resume_cli_output_and_state_transitions(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test resume CLI command, structured output, and state enforcement (FR-027)."""
    run_dir = tmp_path / "artifacts" / "s003" / "runs" / "s003-run-resumable"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = run_dir / "run.json"

    # 1. Resuming an interrupted run succeeds
    run_json.write_text(
        json.dumps({
            "run_id": "s003-run-resumable",
            "state": "interrupted",
            "transitions": [
                {"from_state": "planned", "to_state": "running"},
                {"from_state": "running", "to_state": "interrupted", "reason": "timeout"},
            ],
        }),
        encoding="utf-8",
    )

    exit_code = run(["resume", "--run", str(run_dir)])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["study_id"] == "s003"
    assert out["command"] == "resume"
    assert out["status"] == "resumed"
    assert out["state"] == "running"

    # Verify updated transitions in run.json
    updated_run = json.loads(run_json.read_text(encoding="utf-8"))
    assert updated_run["state"] == "running"
    assert updated_run["transitions"][-1]["from_state"] == "interrupted"
    assert updated_run["transitions"][-1]["to_state"] == "running"

    # 2. Attempting to resume a completed run fails with exit code 4
    run_json.write_text(
        json.dumps({"run_id": "s003-run-resumable", "state": "completed"}),
        encoding="utf-8",
    )
    exit_code_completed = run(["resume", "--run", str(run_dir)])
    assert exit_code_completed == 4
    err = json.loads(capsys.readouterr().err)
    assert err["status"] == "error"
    assert "only interrupted runs can be resumed" in err["error"]["message"]

    # 3. Attempting to resume a failed run fails with exit code 4
    run_json.write_text(
        json.dumps({"run_id": "s003-run-resumable", "state": "failed"}),
        encoding="utf-8",
    )
    exit_code_failed = run(["resume", "--run", str(run_dir)])
    assert exit_code_failed == 4
    err_failed = json.loads(capsys.readouterr().err)
    assert "only interrupted runs can be resumed" in err_failed["error"]["message"]


def test_end_to_end_arbitrary_result_reconstruction(tmp_path: Path) -> None:
    """End-to-end scenario: select arbitrary row from report and reconstruct full lineage (SC-006)."""
    artifacts_root = tmp_path / "artifacts" / "s003"
    matrix_dir = artifacts_root / "matrices" / "s003-matrix-eval"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    data_hash = _digest("source-dataset")
    config_hash = _digest("config-production")
    weights_hash = _digest("weights-checkpoint")
    labels_hash = _digest("sealed-test-labels")
    scores_hash = _digest("neural-predictions")

    chosen_run_id = "s003-run-main-s003_txgcl-0.01-37"
    chosen_cell_key = "main:s003_txgcl:0.01:37"

    matrix_file = matrix_dir / "matrix.json"
    matrix_file.write_text(
        json.dumps({
            "schema_version": 1,
            "study_id": "s003",
            "matrix_id": "s003-matrix-eval",
            "data_digest": data_hash,
            "design_digest": _digest("design-p1"),
            "cells": [
                {
                    "key": chosen_cell_key,
                    "state": "selected",
                    "run_id": chosen_run_id,
                    "weights_digest": weights_hash,
                    "threshold": 0.42,
                    "config_digest": config_hash,
                }
            ],
        }),
        encoding="utf-8",
    )

    eval_file = matrix_dir / "evaluations.json"
    eval_file.write_text(
        json.dumps({
            chosen_run_id: {
                "test_labels_digest": labels_hash,
                "scores_digest": scores_hash,
                "pooled": {
                    "f1_illicit": 0.812,
                    "mcc": 0.775,
                    "precision_illicit": 0.84,
                    "recall_illicit": 0.785,
                    "pr_auc_illicit": 0.86,
                },
            }
        }),
        encoding="utf-8",
    )

    # Generate report
    report_res = generate_matrix_report(matrix_dir)
    assert report_res["status"] == "complete"

    # Reconstruct lineage for chosen_run_id
    lineage = reconstruct_result_lineage(
        chosen_run_id,
        artifacts_root=artifacts_root,
        matrix_dir=matrix_dir,
    )

    # Assert all inputs and decisions are accurately reconstructed
    assert lineage.run_id == chosen_run_id
    assert lineage.cell_key == chosen_cell_key
    assert lineage.study_id == "s003"
    assert lineage.method == "s003_txgcl"
    assert lineage.fraction == 0.01
    assert lineage.seed == 37
    assert lineage.variant == "complete"
    assert lineage.data_digest == data_hash
    assert lineage.config_digest == config_hash
    assert lineage.weights_digest == weights_hash
    assert lineage.threshold == 0.42
    assert lineage.test_labels_digest == labels_hash
    assert lineage.scores_digest == scores_hash
    assert lineage.metrics["f1_illicit"] == 0.812
    assert lineage.metrics["mcc"] == 0.775
    assert lineage.isolated_from_s002 is True
    assert "torch" in lineage.environment["dependencies"]
    assert len(lineage.source_artifacts) >= 2

    # Attempting to reconstruct unknown run fails
    with pytest.raises(ProvenanceError, match="not found in provided paths"):
        reconstruct_result_lineage("s003-run-nonexistent", matrix_dir=matrix_dir)


def test_completed_artifact_immutability(tmp_path: Path) -> None:
    """Verify that completed artifacts, sealed cohorts, and terminal run states are immutable (SC-003, SC-006)."""
    from hgcl.studies.s003.artifacts import ArtifactError, ArtifactStore, EvaluationCell, EvaluationCohort
    from hgcl.studies.s003.provenance import RunStateMachine

    store = ArtifactStore(tmp_path / "artifacts" / "s003")

    # 1. Overwriting a completed artifact fails closed
    original_payload = {
        "status": "completed",
        "run_id": "s003-run-immutable",
        "metrics": {"f1_illicit": 0.80},
    }
    artifact_path = store.write_json("runs/s003-run-immutable/run.json", original_payload)
    original_digest = store.sha256(artifact_path)

    # Attempting to rewrite the artifact raises ArtifactError
    with pytest.raises(ArtifactError, match="artifact already exists"):
        store.write_json("runs/s003-run-immutable/run.json", {"status": "completed", "tampered": True})

    # File contents and hash remain unaltered
    assert store.sha256(artifact_path) == original_digest
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == original_payload

    # 2. EvaluationCohort immutability after sealing
    cells = [EvaluationCell(key=f"cell-{i:03d}", state="failed") for i in range(205)]
    cells[0] = EvaluationCell(
        key="cell-000",
        state="selected",
        run_id="s003-run-000",
        weights_digest=_digest("weights"),
        threshold=0.5,
        config_digest=_digest("config"),
    )
    cohort = EvaluationCohort(
        cohort_id="s003-cohort-sealed",
        design_digest=_digest("design"),
        test_store_digest=_digest("test_store"),
        cells=tuple(cells),
    )
    cohort.seal()
    assert cohort.state == "sealed"

    # Cannot add cells once sealed
    with pytest.raises(ArtifactError, match="cohort is sealed"):
        cohort.add_cell(EvaluationCell(key="cell-extra", state="failed"))

    # Cannot seal again
    with pytest.raises(ArtifactError, match="cannot seal cohort in state sealed"):
        cohort.seal()

    cohort.release()
    assert cohort.state == "released"

    # Cannot release again
    with pytest.raises(ArtifactError, match="already released"):
        cohort.release()

    # 3. Terminal state machine immutability
    sm = RunStateMachine(run_id="s003-run-term")
    sm.start()
    sm.select()
    sm.start_evaluation()
    sm.complete("finished")
    assert sm.state == "completed"
    assert sm.is_terminal is True

    with pytest.raises(ProvenanceError, match="terminal state 'completed'"):
        sm.start()
    with pytest.raises(ProvenanceError, match="terminal state 'completed'"):
        sm.resume()
    with pytest.raises(ProvenanceError, match="terminal state 'completed'"):
        sm.fail("runtime", "late failure")


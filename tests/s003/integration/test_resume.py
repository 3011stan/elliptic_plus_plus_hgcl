from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from hgcl.studies.s003.artifacts import EvaluationCell, EvaluationCohort
from hgcl.studies.s003.evaluation import (
    EvaluationAccessError,
    TestLabelStore,
    evaluate_frozen_members,
    evaluate_technical_rerun,
)
from hgcl.studies.s003.matrix import MatrixScheduleError, MatrixScheduler
from hgcl.studies.s003.provenance import (
    EvaluationManifest,
    ProvenanceError,
    ReloadEquivalenceReport,
    RunCheckpoint,
    RunStateMachine,
    assert_reload_equivalence,
    check_reload_equivalence,
    create_technical_rerun_manifest,
    validate_technical_rerun,
)


def _digest(val: str = "a") -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()


def test_strict_resume_lifecycle_and_incompatible_checkpoint(tmp_path: Path) -> None:
    """Test strict resume transitions in state machine and checkpoint compatibility verification."""
    # 1. State machine only allows resume from interrupted
    sm = RunStateMachine(run_id="s003-run-strict")
    assert sm.state == "planned"
    with pytest.raises(ProvenanceError, match="can only resume an interrupted run"):
        sm.resume()

    sm.start()
    assert sm.state == "running"
    with pytest.raises(ProvenanceError, match="can only resume an interrupted run"):
        sm.resume()

    sm.interrupt("preemption")
    assert sm.state == "interrupted"

    sm.resume("resumed by scheduler")
    assert sm.state == "running"

    sm.select("best validation f1")
    with pytest.raises(ProvenanceError, match="can only resume an interrupted run"):
        sm.resume()

    sm.start_evaluation()
    sm.complete("finished")
    assert sm.is_terminal is True
    with pytest.raises(ProvenanceError, match="cannot transition from terminal state"):
        sm.resume()

    # 2. Checkpoint compatibility verification
    ckpt = RunCheckpoint(
        run_id="s003-run-strict",
        epoch=10,
        phase="downstream",
        data_digest=_digest("1"),
        config_digest=_digest("2"),
        code_revision="3" * 40,
        dependencies={"torch": "2.6.0"},
        rng_states=RunCheckpoint.capture_rng_states(),
        model_state_dict={"weight": torch.randn(4, 4)},
    )
    ckpt_path = tmp_path / "checkpoint.pt"
    ckpt.save_atomic(ckpt_path)

    loaded = RunCheckpoint.load(ckpt_path)
    # Compatible resume succeeds
    loaded.assert_compatible(data_digest=_digest("1"), config_digest=_digest("2"), code_revision="3" * 40)

    # Incompatible resume fails closed
    with pytest.raises(ProvenanceError, match="data_digest mismatch"):
        loaded.assert_compatible(data_digest=_digest("9"), config_digest=_digest("2"), code_revision="3" * 40)
    with pytest.raises(ProvenanceError, match="config_digest mismatch"):
        loaded.assert_compatible(data_digest=_digest("1"), config_digest=_digest("9"), code_revision="3" * 40)
    with pytest.raises(ProvenanceError, match="code_revision mismatch"):
        loaded.assert_compatible(data_digest=_digest("1"), config_digest=_digest("2"), code_revision="9" * 40)


def test_matrix_scheduler_strict_resume_and_incomplete_accounting() -> None:
    """Test MatrixScheduler strict resume rules and incomplete cell accounting."""
    dummy_labels = TestLabelStore.seal({35: {"tx1": 1}})
    cell_keys = [f"cell-{i}" for i in range(205)]
    design_digest = hashlib.sha256("\n".join(cell_keys).encode("utf-8")).hexdigest()

    scheduler = MatrixScheduler(
        cell_keys,
        design_digest=design_digest,
        test_labels=dummy_labels,
        expected_count=205,
    )

    # Cannot resume planned cell
    with pytest.raises(MatrixScheduleError, match="only an interrupted cell can resume"):
        scheduler.resume("cell-0")

    # Start -> Interrupt -> Resume
    scheduler.start("cell-0")
    assert scheduler.cells["cell-0"].state == "running"
    with pytest.raises(MatrixScheduleError, match="only an interrupted cell can resume"):
        scheduler.resume("cell-0")

    scheduler.interrupt("cell-0", "interrupted by test")
    assert scheduler.cells["cell-0"].state == "interrupted"

    scheduler.resume("cell-0")
    assert scheduler.cells["cell-0"].state == "running"

    # Select cell -> immutable, cannot resume
    scheduler.select(
        "cell-0",
        run_id="s003-run-0",
        weights_digest=_digest("a"),
        threshold=0.5,
        config_digest=_digest("b"),
    )
    with pytest.raises(MatrixScheduleError, match="only an interrupted cell can resume"):
        scheduler.resume("cell-0")

    # Terminal failed cell cannot resume
    scheduler.start("cell-1")
    scheduler.terminate("cell-1", state="failed", kind="runtime", message="OOM")
    with pytest.raises(MatrixScheduleError, match="only an interrupted cell can resume"):
        scheduler.resume("cell-1")

    # Terminal invalid cell cannot resume
    scheduler.start("cell-2")
    scheduler.terminate("cell-2", state="invalid", kind="data", message="missing endpoints")
    with pytest.raises(MatrixScheduleError, match="only an interrupted cell can resume"):
        scheduler.resume("cell-2")


def test_reload_equivalence_numeric_tolerances_and_decisions() -> None:
    """Test reload equivalence according to FR-042 (rtol=1e-5, atol=2e-6, identical classes)."""
    # 1. Exact identical scores
    scores_a = torch.tensor([0.1, 0.49, 0.50, 0.51, 0.95], dtype=torch.float64)
    scores_b = torch.tensor([0.1, 0.49, 0.50, 0.51, 0.95], dtype=torch.float64)
    threshold = 0.50

    report = check_reload_equivalence(scores_a, scores_b, threshold)
    assert report.equivalent is True
    assert report.max_abs_diff == 0.0
    assert report.max_rel_diff == 0.0
    assert report.decisions_match is True
    assert report.rtol == 1e-5
    assert report.atol == 2e-6

    # 2. Within tolerances without decision flip
    # Perturbation 1e-7 < atol (2e-6) and rtol (1e-5)
    scores_c = scores_a + 1e-7
    report_c = assert_reload_equivalence(scores_a, scores_c, threshold=0.50)
    assert report_c.equivalent is True
    assert report_c.max_abs_diff <= 2e-6

    # 3. Beyond tolerance fails
    scores_d = scores_a + 1e-4  # exceeds atol and rtol
    report_d = check_reload_equivalence(scores_a, scores_d, threshold)
    assert report_d.equivalent is False
    assert report_d.details["within_tolerance"] is False
    with pytest.raises(ProvenanceError, match="reload equivalence check failed"):
        assert_reload_equivalence(scores_a, scores_d, threshold)

    # 4. Within tolerance but decisions flip across threshold fails
    scores_orig = torch.tensor([0.4999999], dtype=torch.float64)
    scores_reloaded = torch.tensor([0.5000001], dtype=torch.float64)  # diff = 2e-7 < atol, but flips threshold 0.5
    report_flip = check_reload_equivalence(scores_orig, scores_reloaded, threshold=0.5)
    assert report_flip.details["within_tolerance"] is True
    assert report_flip.decisions_match is False
    assert report_flip.equivalent is False
    with pytest.raises(ProvenanceError, match="decisions_match=False"):
        assert_reload_equivalence(scores_orig, scores_reloaded, threshold=0.5)


def test_model_checkpoint_save_reload_equivalence(tmp_path: Path) -> None:
    """Test full checkpoint save and reload equivalence on PyTorch module."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
    test_inputs = torch.randn(50, 10)

    with torch.no_grad():
        original_scores = torch.sigmoid(model(test_inputs)).squeeze(-1)

    threshold = 0.50
    checkpoint = RunCheckpoint(
        run_id="s003-run-model",
        epoch=100,
        phase="downstream",
        data_digest=_digest("a"),
        config_digest=_digest("b"),
        code_revision="c" * 40,
        dependencies={"torch": "2.6.0"},
        rng_states=RunCheckpoint.capture_rng_states(),
        model_state_dict=model.state_dict(),
    )
    checkpoint_file = tmp_path / "model_checkpoint.pt"
    checkpoint.save_atomic(checkpoint_file)

    # Reload into fresh model instance
    reloaded_ckpt = RunCheckpoint.load(checkpoint_file)
    fresh_model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
    fresh_model.load_state_dict(reloaded_ckpt.model_state_dict)

    with torch.no_grad():
        reloaded_scores = torch.sigmoid(fresh_model(test_inputs)).squeeze(-1)

    report = assert_reload_equivalence(original_scores, reloaded_scores, threshold)
    assert report.equivalent is True
    assert report.max_abs_diff <= 2e-6
    assert report.decisions_match is True


def test_evaluation_access_logging_and_cohort_audit() -> None:
    """Test detailed audit logging of test-label access (FR-038, SC-003)."""
    store = TestLabelStore.seal({35: {"tx_a": 1, "tx_b": 0}, 36: {"tx_c": 1}})
    assert store.access_log == []

    members = {
        "s003-run-1": {"weights_digest": _digest("1"), "threshold": 0.5},
        "s003-run-2": {"weights_digest": _digest("2"), "threshold": 0.4},
    }

    # Reading before release fails
    with pytest.raises(EvaluationAccessError, match="test labels are sealed"):
        store.read("s003-cohort", "s003-run-1", _digest("1"), 0.5)
    assert store.access_log == []

    # Release cohort
    store.release("s003-cohort", members)

    # Read member 1
    labels = store.read(
        "s003-cohort",
        "s003-run-1",
        _digest("1"),
        0.5,
        accessor="pipeline_runner",
        purpose="evaluation_step",
    )
    assert labels[35]["tx_a"] == 1
    assert len(store.access_log) == 1

    entry = store.access_log[0]
    assert entry["cohort_id"] == "s003-cohort"
    assert entry["run_id"] == "s003-run-1"
    assert entry["accessor"] == "pipeline_runner"
    assert entry["purpose"] == "evaluation_step"
    assert "timestamp" in entry

    # Unauthorized access attempts do not append to access log
    with pytest.raises(EvaluationAccessError, match="not a frozen cohort member"):
        store.read("s003-cohort", "s003-unknown", _digest("1"), 0.5)
    assert len(store.access_log) == 1

    with pytest.raises(EvaluationAccessError, match="weights or threshold differ"):
        store.read("s003-cohort", "s003-run-1", _digest("wrong"), 0.5)
    assert len(store.access_log) == 1

    with pytest.raises(EvaluationAccessError, match="weights or threshold differ"):
        store.read("s003-cohort", "s003-run-1", _digest("1"), 0.75)
    assert len(store.access_log) == 1


def test_post_unblinding_technical_rerun_enforcement_without_reselection() -> None:
    """Test post-unblinding technical rerun enforcement and reselection prohibition (FR-038, FR-042, SC-003)."""
    store = TestLabelStore.seal({35: {"tx1": 1, "tx2": 0}})
    original_weights = _digest("w")
    original_threshold = 0.50
    cohort_id = "s003-cohort-audit"
    orig_run_id = "s003-run-original"
    rerun_run_id = "s003-run-rerun"

    members = {
        orig_run_id: {"weights_digest": original_weights, "threshold": original_threshold},
    }
    store.release(cohort_id, members)

    # 1. Attempting a technical rerun with altered weights (reselection) is blocked
    with pytest.raises(EvaluationAccessError, match="weights or threshold differ"):
        store.read(
            cohort_id,
            rerun_run_id,
            _digest("altered_w"),
            original_threshold,
            technical_rerun_of=orig_run_id,
            technical_failure_justification="Node preempted during inference",
        )

    # 2. Attempting a technical rerun with altered threshold (reselection) is blocked
    with pytest.raises(EvaluationAccessError, match="weights or threshold differ"):
        store.read(
            cohort_id,
            rerun_run_id,
            original_weights,
            0.65,
            technical_rerun_of=orig_run_id,
            technical_failure_justification="Node preempted during inference",
        )

    # 3. Attempting a technical rerun without justification is blocked
    with pytest.raises(EvaluationAccessError, match="requires a technical failure justification"):
        store.read(
            cohort_id,
            rerun_run_id,
            original_weights,
            original_threshold,
            technical_rerun_of=orig_run_id,
            technical_failure_justification="",
        )

    # 4. Attempting a technical rerun with same run_id as original is blocked
    with pytest.raises(EvaluationAccessError, match="distinct run_id"):
        store.read(
            cohort_id,
            orig_run_id,
            original_weights,
            original_threshold,
            technical_rerun_of=orig_run_id,
            technical_failure_justification="Crash recovery",
        )

    # 5. Attempting a technical rerun referencing a non-member is blocked
    with pytest.raises(EvaluationAccessError, match="not a frozen cohort member"):
        store.read(
            cohort_id,
            rerun_run_id,
            original_weights,
            original_threshold,
            technical_rerun_of="s003-run-nonexistent",
            technical_failure_justification="Crash recovery",
        )

    # 6. Valid technical rerun succeeds and logs technical rerun details
    provider = lambda run_id, step, ids: [0.9 if item == "tx1" else 0.1 for item in ids]
    metrics = evaluate_technical_rerun(
        store,
        cohort_id,
        orig_run_id,
        rerun_run_id,
        "Storage node crashed during original evaluation batch",
        provider,
        weights_digest=original_weights,
        threshold=original_threshold,
    )
    assert metrics["pooled"]["f1_illicit"] == 1.0

    # Verify access log entry
    assert len(store.access_log) == 1
    log_entry = store.access_log[0]
    assert log_entry["cohort_id"] == cohort_id
    assert log_entry["run_id"] == rerun_run_id
    assert log_entry["technical_rerun_of"] == orig_run_id
    assert log_entry["technical_failure_justification"] == "Storage node crashed during original evaluation batch"
    assert log_entry["purpose"] == "technical_rerun"

    # 7. Validate technical rerun manifests and validation functions
    validate_technical_rerun(
        original_run_id=orig_run_id,
        original_weights_digest=original_weights,
        original_threshold=original_threshold,
        original_config_digest=_digest("c"),
        original_data_digest=_digest("d"),
        rerun_run_id=rerun_run_id,
        rerun_weights_digest=original_weights,
        rerun_threshold=original_threshold,
        rerun_config_digest=_digest("c"),
        rerun_data_digest=_digest("d"),
        technical_failure_justification="Crash recovery",
    )

    # Validation blocks altered weights
    with pytest.raises(ProvenanceError, match="cannot alter model weights"):
        validate_technical_rerun(
            original_run_id=orig_run_id,
            original_weights_digest=original_weights,
            original_threshold=original_threshold,
            original_config_digest=_digest("c"),
            original_data_digest=_digest("d"),
            rerun_run_id=rerun_run_id,
            rerun_weights_digest=_digest("altered"),
            rerun_threshold=original_threshold,
            rerun_config_digest=_digest("c"),
            rerun_data_digest=_digest("d"),
            technical_failure_justification="Crash recovery",
        )

    # Validation blocks altered threshold
    with pytest.raises(ProvenanceError, match="cannot alter decision threshold"):
        validate_technical_rerun(
            original_run_id=orig_run_id,
            original_weights_digest=original_weights,
            original_threshold=original_threshold,
            original_config_digest=_digest("c"),
            original_data_digest=_digest("d"),
            rerun_run_id=rerun_run_id,
            rerun_weights_digest=original_weights,
            rerun_threshold=0.8,
            rerun_config_digest=_digest("c"),
            rerun_data_digest=_digest("d"),
            technical_failure_justification="Crash recovery",
        )

    # Manifest creation binds technical rerun
    orig_manifest = EvaluationManifest(
        run_id=orig_run_id,
        cohort_id=cohort_id,
        test_labels_digest=_digest("labels"),
        weights_digest=original_weights,
        threshold=original_threshold,
        scores_digest=_digest("scores1"),
    )
    rerun_manifest = create_technical_rerun_manifest(
        orig_manifest,
        rerun_run_id,
        "Storage node crashed during original evaluation batch",
        new_scores_digest=_digest("scores2"),
        weights_digest=original_weights,
        threshold=original_threshold,
    )
    assert rerun_manifest.run_id == rerun_run_id
    assert rerun_manifest.technical_rerun_of == orig_run_id
    assert rerun_manifest.weights_digest == original_weights
    assert rerun_manifest.threshold == original_threshold
    assert rerun_manifest.technical_failure_justification == "Storage node crashed during original evaluation batch"

    # Attempting to create technical rerun manifest with altered weights fails
    with pytest.raises(ProvenanceError, match="cannot alter model weights"):
        create_technical_rerun_manifest(
            orig_manifest,
            rerun_run_id,
            "Crash recovery",
            new_scores_digest=_digest("scores2"),
            weights_digest=_digest("altered"),
            threshold=original_threshold,
        )

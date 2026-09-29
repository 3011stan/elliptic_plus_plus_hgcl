from __future__ import annotations

import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch

from hgcl.studies.s003.provenance import (
    EnvironmentManifest,
    EvaluationManifest,
    ProvenanceError,
    RunCheckpoint,
    RunStateMachine,
    SelectionManifest,
    SourceManifest,
)


def _digest(char: str = "a") -> str:
    return char * 64


def test_manifest_validation_and_serialization() -> None:
    # 1. SourceManifest
    src_man = SourceManifest(
        data_root="/data/raw",
        files={"tx.csv": {"size_bytes": 100, "sha256": _digest("1")}},
        data_digest=_digest("2"),
    )
    assert src_man.data_digest == _digest("2")
    assert src_man.to_dict()["study_id"] == "s003"
    with pytest.raises(ProvenanceError, match="SHA-256"):
        SourceManifest(data_root="/data/raw", files={}, data_digest="not-hex")

    # 2. EnvironmentManifest
    env_man = EnvironmentManifest.capture(device="cpu", code_revision="0" * 40)
    assert env_man.device == "cpu"
    assert "torch" in env_man.dependencies
    assert len(env_man.code_revision) == 40
    with pytest.raises(ProvenanceError, match="40-character"):
        EnvironmentManifest(
            python_version="3.11", platform="darwin", device="cpu",
            dependencies={}, code_revision="short",
        )

    # 3. SelectionManifest
    sel_man = SelectionManifest(
        run_id="s003-run-001",
        threshold=0.45,
        validation_f1_illicit=0.72,
        validation_mcc=0.68,
        weights_digest=_digest("3"),
        config_digest=_digest("4"),
        hyperparameters={"lr": 0.001},
    )
    assert sel_man.run_id == "s003-run-001"
    with pytest.raises(ProvenanceError, match="s003-"):
        SelectionManifest(
            run_id="invalid-id", threshold=0.5, validation_f1_illicit=0.7,
            validation_mcc=0.6, weights_digest=_digest("3"), config_digest=_digest("4"),
            hyperparameters={},
        )

    # 4. EvaluationManifest requires access_count == 1
    eval_man = EvaluationManifest(
        run_id="s003-run-001",
        cohort_id="s003-cohort-001",
        test_labels_digest=_digest("5"),
        weights_digest=_digest("3"),
        threshold=0.45,
        scores_digest=_digest("6"),
        access_count=1,
    )
    assert eval_man.access_count == 1
    with pytest.raises(ProvenanceError, match="access_count"):
        EvaluationManifest(
            run_id="s003-run-001", cohort_id="s003-cohort-001",
            test_labels_digest=_digest("5"), weights_digest=_digest("3"),
            threshold=0.45, scores_digest=_digest("6"), access_count=2,
        )


def test_run_state_machine_valid_lifecycle() -> None:
    sm = RunStateMachine(run_id="s003-run-01")
    assert sm.state == "planned"
    assert sm.is_terminal is False

    # planned -> running
    sm.start()
    assert sm.state == "running"
    assert sm.started_at is not None

    # running -> interrupted -> running
    sm.interrupt("preempted by scheduler")
    assert sm.state == "interrupted"
    assert sm.is_terminal is False

    sm.resume("resuming task")
    assert sm.state == "running"

    # running -> selected
    sm.select("best validation f1")
    assert sm.state == "selected"

    # selected -> evaluating
    sm.start_evaluation("cohort evaluation")
    assert sm.state == "evaluating"

    # evaluating -> completed
    sm.complete("finished")
    assert sm.state == "completed"
    assert sm.is_terminal is True
    assert sm.completed_at is not None
    assert len(sm.transitions) == 7


def test_run_state_machine_terminal_immutability() -> None:
    # 1. Terminal completed cannot transition
    sm1 = RunStateMachine(run_id="s003-run-completed")
    sm1.start()
    sm1.select()
    sm1.start_evaluation()
    sm1.complete()
    with pytest.raises(ProvenanceError, match="terminal state 'completed'"):
        sm1.start()
    with pytest.raises(ProvenanceError, match="terminal state 'completed'"):
        sm1.fail("runtime", "late error")

    # 2. Terminal failed cannot transition
    sm2 = RunStateMachine(run_id="s003-run-failed")
    sm2.start()
    sm2.fail("resource", "out of memory")
    assert sm2.state == "failed"
    assert sm2.is_terminal is True
    assert sm2.failure_kind == "resource"
    with pytest.raises(ProvenanceError, match="terminal state 'failed'"):
        sm2.resume()
    with pytest.raises(ProvenanceError, match="terminal state 'failed'"):
        sm2.complete()

    # 3. Terminal invalid cannot transition
    sm3 = RunStateMachine(run_id="s003-run-invalid")
    sm3.start()
    sm3.invalidate("data", "corrupt feature matrix")
    assert sm3.state == "invalid"
    assert sm3.is_terminal is True
    with pytest.raises(ProvenanceError, match="terminal state 'invalid'"):
        sm3.resume()


def test_run_state_machine_failure_categorization() -> None:
    sm = RunStateMachine(run_id="s003-run-test")
    sm.start()
    with pytest.raises(ProvenanceError, match="invalid failure kind"):
        sm.fail("unrecognized_kind", "error")

    sm.fail("runtime", "numerical instability")
    assert sm.failure_kind == "runtime"
    assert sm.failure_message == "numerical instability"


def test_checkpoint_atomic_save_load_and_rng_reproducibility(tmp_path: Path) -> None:
    # Set seeds and draw a random value
    torch.manual_seed(1234)
    np.random.seed(5678)
    random.seed(9999)

    rng_states = RunCheckpoint.capture_rng_states()
    val_torch = torch.randn(5)
    val_np = np.random.rand(5)
    val_py = random.random()

    ckpt = RunCheckpoint(
        run_id="s003-run-ckpt",
        epoch=50,
        phase="ssl",
        data_digest=_digest("a"),
        config_digest=_digest("b"),
        code_revision="c" * 40,
        dependencies={"torch": "2.6.0"},
        rng_states=rng_states,
        model_state_dict={"weight": torch.tensor([1.0, 2.0, 3.0])},
        optimizer_state_dict=None,
    )

    ckpt_path = tmp_path / "checkpoints" / "checkpoint_epoch_50.pt"
    saved = ckpt.save_atomic(ckpt_path)
    assert saved.is_file()
    assert not list(ckpt_path.parent.glob("*.tmp"))

    loaded = RunCheckpoint.load(saved)
    assert loaded.run_id == "s003-run-ckpt"
    assert loaded.epoch == 50
    assert loaded.phase == "ssl"
    assert torch.equal(loaded.model_state_dict["weight"], torch.tensor([1.0, 2.0, 3.0]))

    # Test compatibility assertions
    loaded.assert_compatible(data_digest=_digest("a"), config_digest=_digest("b"), code_revision="c" * 40)
    with pytest.raises(ProvenanceError, match="data_digest mismatch"):
        loaded.assert_compatible(data_digest=_digest("0"), config_digest=_digest("b"), code_revision="c" * 40)
    with pytest.raises(ProvenanceError, match="config_digest mismatch"):
        loaded.assert_compatible(data_digest=_digest("a"), config_digest=_digest("0"), code_revision="c" * 40)
    with pytest.raises(ProvenanceError, match="code_revision mismatch"):
        loaded.assert_compatible(data_digest=_digest("a"), config_digest=_digest("b"), code_revision="0" * 40)

    # Test RNG state restoration yields exact identical subsequent draws
    RunCheckpoint.restore_rng_states(loaded.rng_states)
    assert torch.equal(torch.randn(5), val_torch)
    assert np.array_equal(np.random.rand(5), val_np)
    assert random.random() == val_py

"""End-to-end result lineage, audit manifests, atomic run transitions, and checkpoints for S003."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


class ProvenanceError(RuntimeError):
    """Raised when run state transitions or lineage invariants are violated."""


VALID_LIFECYCLE_STATES = frozenset(
    {"planned", "running", "selected", "evaluating", "completed", "interrupted", "failed", "invalid"}
)

TERMINAL_STATES = frozenset({"completed", "failed", "invalid"})

VALID_FAILURE_KINDS = frozenset({"data", "resource", "runtime", "preemption", "contract", "fixture"})


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StateTransition:
    """Audit record of a state transition."""

    from_state: str
    to_state: str
    timestamp: str = field(default_factory=_now_iso)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunStateMachine:
    """Strict state machine for S003 execution runs with terminal immutability."""

    run_id: str
    state: str = "planned"
    transitions: list[StateTransition] = field(default_factory=list)
    failure_kind: str | None = None
    failure_message: str | None = None
    started_at: str | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.startswith("s003-"):
            raise ProvenanceError(f"run_id must start with s003-: {self.run_id}")
        if self.state not in VALID_LIFECYCLE_STATES:
            raise ProvenanceError(f"unknown lifecycle state: {self.state}")
        if not self.transitions:
            self.transitions.append(StateTransition(from_state="none", to_state=self.state, reason="initialized"))

    def _transition(self, to_state: str, *, reason: str | None = None) -> None:
        if self.state in TERMINAL_STATES:
            raise ProvenanceError(f"cannot transition from terminal state '{self.state}' to '{to_state}'")
        if to_state not in VALID_LIFECYCLE_STATES:
            raise ProvenanceError(f"invalid target state: {to_state}")

        prev_state = self.state
        self.state = to_state
        self.transitions.append(StateTransition(from_state=prev_state, to_state=to_state, reason=reason))

    def start(self) -> None:
        if self.state in TERMINAL_STATES:
            raise ProvenanceError(f"cannot transition from terminal state '{self.state}'")
        if self.state != "planned":
            raise ProvenanceError(f"cannot start run from state '{self.state}' (must be 'planned')")
        self.started_at = _now_iso()
        self._transition("running", reason="execution started")

    def resume(self, reason: str = "resuming execution") -> None:
        if self.state in TERMINAL_STATES:
            raise ProvenanceError(f"cannot transition from terminal state '{self.state}'")
        if self.state != "interrupted":
            raise ProvenanceError(f"can only resume an interrupted run (current state: '{self.state}')")
        self._transition("running", reason=reason)

    def interrupt(self, reason: str = "execution interrupted") -> None:
        if self.state in TERMINAL_STATES:
            raise ProvenanceError(f"cannot transition from terminal state '{self.state}'")
        if self.state not in {"running", "evaluating"}:
            raise ProvenanceError(f"cannot interrupt run from state '{self.state}'")
        self._transition("interrupted", reason=reason)

    def select(self, reason: str = "checkpoint selected") -> None:
        if self.state in TERMINAL_STATES:
            raise ProvenanceError(f"cannot transition from terminal state '{self.state}'")
        if self.state != "running":
            raise ProvenanceError(f"cannot select run from state '{self.state}' (must be 'running')")
        self._transition("selected", reason=reason)

    def start_evaluation(self, reason: str = "evaluation started") -> None:
        if self.state in TERMINAL_STATES:
            raise ProvenanceError(f"cannot transition from terminal state '{self.state}'")
        if self.state != "selected":
            raise ProvenanceError(f"cannot evaluate run from state '{self.state}' (must be 'selected')")
        self._transition("evaluating", reason=reason)

    def complete(self, reason: str = "run completed") -> None:
        if self.state in TERMINAL_STATES:
            raise ProvenanceError(f"cannot transition from terminal state '{self.state}'")
        if self.state != "evaluating":
            raise ProvenanceError(f"cannot complete run from state '{self.state}' (must be 'evaluating')")
        self.completed_at = _now_iso()
        self._transition("completed", reason=reason)

    def fail(self, kind: str, message: str) -> None:
        if self.state in TERMINAL_STATES:
            raise ProvenanceError(f"cannot transition from terminal state '{self.state}'")
        if kind not in VALID_FAILURE_KINDS:
            raise ProvenanceError(f"invalid failure kind: {kind}")
        self.failure_kind = kind
        self.failure_message = message
        self.completed_at = _now_iso()
        self._transition("failed", reason=f"{kind}: {message}")

    def invalidate(self, kind: str, message: str) -> None:
        if self.state in TERMINAL_STATES:
            raise ProvenanceError(f"cannot transition from terminal state '{self.state}'")
        if kind not in VALID_FAILURE_KINDS:
            raise ProvenanceError(f"invalid failure kind: {kind}")
        self.failure_kind = kind
        self.failure_message = message
        self.completed_at = _now_iso()
        self._transition("invalid", reason=f"{kind}: {message}")

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "is_terminal": self.is_terminal,
            "failure_kind": self.failure_kind,
            "failure_message": self.failure_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "transitions": [t.to_dict() for t in self.transitions],
        }


@dataclass(frozen=True)
class SourceManifest:
    """Inventory and SHA-256 hashes of original source files."""

    data_root: str
    files: Mapping[str, dict[str, Any]]
    data_digest: str
    schema_version: int = 1
    study_id: str = "s003"

    def __post_init__(self) -> None:
        if not _is_sha256(self.data_digest):
            raise ProvenanceError("data_digest must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentManifest:
    """Environment, runtime, dependency versions, and code revision."""

    python_version: str
    platform: str
    device: str
    dependencies: Mapping[str, str]
    code_revision: str
    clean_worktree: bool = True
    schema_version: int = 1
    study_id: str = "s003"

    def __post_init__(self) -> None:
        if len(self.code_revision) != 40:
            raise ProvenanceError("code_revision must be a 40-character commit hash")

    @classmethod
    def capture(
        cls,
        *,
        device: str = "cpu",
        code_revision: str = "0" * 40,
        clean_worktree: bool = True,
    ) -> EnvironmentManifest:
        deps = {
            "torch": getattr(torch, "__version__", "unknown"),
            "numpy": getattr(np, "__version__", "unknown"),
        }
        try:
            import torch_geometric
            deps["torch_geometric"] = torch_geometric.__version__
        except ImportError:
            pass
        try:
            import sklearn
            deps["scikit_learn"] = sklearn.__version__
        except ImportError:
            pass
        try:
            import polars
            deps["polars"] = polars.__version__
        except ImportError:
            pass
        try:
            import pyarrow
            deps["pyarrow"] = pyarrow.__version__
        except ImportError:
            pass

        return cls(
            python_version=sys.version.split()[0],
            platform=sys.platform,
            device=device,
            dependencies=deps,
            code_revision=code_revision,
            clean_worktree=clean_worktree,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionManifest:
    """Internal validation model selection, weights digest, and hyperparameter configuration."""

    run_id: str
    threshold: float
    validation_f1_illicit: float
    validation_mcc: float
    weights_digest: str
    config_digest: str
    hyperparameters: Mapping[str, Any]
    selected_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        if not self.run_id.startswith("s003-"):
            raise ProvenanceError("run_id must start with s003-")
        if not _is_sha256(self.weights_digest) or not _is_sha256(self.config_digest):
            raise ProvenanceError("weights_digest and config_digest must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationManifest:
    """Evaluation manifest recording sealed cohort evaluation and single-access audit."""

    run_id: str
    cohort_id: str
    test_labels_digest: str
    weights_digest: str
    threshold: float
    scores_digest: str
    access_count: int = 1
    evaluated_at: str = field(default_factory=_now_iso)
    technical_rerun_of: str | None = None
    technical_failure_justification: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.startswith("s003-") or not self.cohort_id.startswith("s003-"):
            raise ProvenanceError("run_id and cohort_id must start with s003-")
        if not _is_sha256(self.test_labels_digest) or not _is_sha256(self.weights_digest):
            raise ProvenanceError("digests must be SHA-256")
        if not _is_sha256(self.scores_digest):
            raise ProvenanceError("scores_digest must be SHA-256")
        if self.access_count != 1:
            raise ProvenanceError(f"access_count must be exactly 1, got {self.access_count}")
        if self.technical_rerun_of is not None:
            if not self.technical_rerun_of.startswith("s003-"):
                raise ProvenanceError("technical_rerun_of must start with s003-")
            if self.technical_rerun_of == self.run_id:
                raise ProvenanceError("technical_rerun_of must differ from run_id")
            if not self.technical_failure_justification or not str(self.technical_failure_justification).strip():
                raise ProvenanceError("technical rerun requires a non-empty technical_failure_justification")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_technical_rerun(
    *,
    original_run_id: str,
    original_weights_digest: str,
    original_threshold: float,
    original_config_digest: str,
    original_data_digest: str,
    rerun_run_id: str,
    rerun_weights_digest: str,
    rerun_threshold: float,
    rerun_config_digest: str,
    rerun_data_digest: str,
    technical_failure_justification: str,
) -> None:
    """Validate that a technical rerun does not alter model selection, config, or data."""
    if not original_run_id.startswith("s003-") or not rerun_run_id.startswith("s003-"):
        raise ProvenanceError("run IDs must start with s003-")
    if rerun_run_id == original_run_id:
        raise ProvenanceError("technical rerun must have a distinct run_id from original")
    if not technical_failure_justification or not str(technical_failure_justification).strip():
        raise ProvenanceError("technical rerun requires a technical failure justification")
    if rerun_weights_digest != original_weights_digest:
        raise ProvenanceError("technical rerun cannot alter model weights (no reselection permitted)")
    if abs(rerun_threshold - original_threshold) > 1e-9:
        raise ProvenanceError("technical rerun cannot alter decision threshold (no reselection permitted)")
    if rerun_config_digest != original_config_digest:
        raise ProvenanceError("technical rerun cannot alter configuration")
    if rerun_data_digest != original_data_digest:
        raise ProvenanceError("technical rerun cannot alter dataset")


def create_technical_rerun_manifest(
    original_manifest: EvaluationManifest,
    new_run_id: str,
    technical_failure_justification: str,
    *,
    new_scores_digest: str,
    weights_digest: str,
    threshold: float,
) -> EvaluationManifest:
    """Create an evaluation manifest for a post-unblinding technical rerun binding to frozen selection."""
    if weights_digest != original_manifest.weights_digest:
        raise ProvenanceError("technical rerun cannot alter model weights (no reselection permitted)")
    if abs(threshold - original_manifest.threshold) > 1e-9:
        raise ProvenanceError("technical rerun cannot alter threshold (no reselection permitted)")
    return EvaluationManifest(
        run_id=new_run_id,
        cohort_id=original_manifest.cohort_id,
        test_labels_digest=original_manifest.test_labels_digest,
        weights_digest=original_manifest.weights_digest,
        threshold=original_manifest.threshold,
        scores_digest=new_scores_digest,
        access_count=1,
        technical_rerun_of=original_manifest.run_id,
        technical_failure_justification=technical_failure_justification,
    )


@dataclass(frozen=True)
class ReloadEquivalenceReport:
    """Verification report for numerical and decision reload equivalence on the same platform."""

    equivalent: bool
    max_abs_diff: float
    max_rel_diff: float
    decisions_match: bool
    rtol: float
    atol: float
    sample_count: int
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_reload_equivalence(
    original_scores: torch.Tensor | Sequence[float],
    reloaded_scores: torch.Tensor | Sequence[float],
    threshold: float,
    *,
    rtol: float = 1e-5,
    atol: float = 2e-6,
) -> ReloadEquivalenceReport:
    """Check reload equivalence according to FR-042 (rtol=1e-5, atol=2e-6, identical classes)."""
    if isinstance(original_scores, (list, tuple)):
        orig = torch.tensor(original_scores, dtype=torch.float64)
    else:
        orig = original_scores.detach().to(dtype=torch.float64).cpu()

    if isinstance(reloaded_scores, (list, tuple)):
        reloaded = torch.tensor(reloaded_scores, dtype=torch.float64)
    else:
        reloaded = reloaded_scores.detach().to(dtype=torch.float64).cpu()

    if orig.shape != reloaded.shape:
        raise ProvenanceError(f"shape mismatch for reload equivalence: {orig.shape} vs {reloaded.shape}")

    if orig.numel() == 0:
        return ReloadEquivalenceReport(
            equivalent=True,
            max_abs_diff=0.0,
            max_rel_diff=0.0,
            decisions_match=True,
            rtol=rtol,
            atol=atol,
            sample_count=0,
            details={"within_tolerance": True, "decisions_match": True},
        )

    abs_diff = torch.abs(orig - reloaded)
    max_abs = float(torch.max(abs_diff).item())

    denom = torch.clamp(torch.abs(orig), min=1e-12)
    rel_diff = abs_diff / denom
    max_rel = float(torch.max(rel_diff).item())

    tolerance = atol + rtol * torch.abs(orig)
    within_tolerance = bool(torch.all(abs_diff <= tolerance).item())

    orig_decisions = (orig >= threshold).to(torch.int64)
    reloaded_decisions = (reloaded >= threshold).to(torch.int64)
    decisions_match = bool(torch.equal(orig_decisions, reloaded_decisions))

    equivalent = within_tolerance and decisions_match
    details = {
        "within_tolerance": within_tolerance,
        "decisions_match": decisions_match,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "tolerance_budget_max": float(torch.max(tolerance).item()),
    }

    return ReloadEquivalenceReport(
        equivalent=equivalent,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        decisions_match=decisions_match,
        rtol=rtol,
        atol=atol,
        sample_count=orig.numel(),
        details=details,
    )


def assert_reload_equivalence(
    original_scores: torch.Tensor | Sequence[float],
    reloaded_scores: torch.Tensor | Sequence[float],
    threshold: float,
    *,
    rtol: float = 1e-5,
    atol: float = 2e-6,
) -> ReloadEquivalenceReport:
    """Assert that reloaded scores satisfy FR-042 equivalence tolerances and decisions."""
    report = check_reload_equivalence(original_scores, reloaded_scores, threshold, rtol=rtol, atol=atol)
    if not report.equivalent:
        raise ProvenanceError(
            f"reload equivalence check failed: within_tolerance={report.details['within_tolerance']} "
            f"(max_abs={report.max_abs_diff:.2e}, tol_max={report.details['tolerance_budget_max']:.2e}), "
            f"decisions_match={report.decisions_match}"
        )
    return report


@dataclass
class RunCheckpoint:
    """Reproducible execution checkpoint containing model weights, RNG states, and dependencies."""

    run_id: str
    epoch: int
    phase: str
    data_digest: str
    config_digest: str
    code_revision: str
    dependencies: Mapping[str, str]
    rng_states: dict[str, Any]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None = None
    created_at: str = field(default_factory=_now_iso)
    checkpoint_digest: str = ""

    def __post_init__(self) -> None:
        if not self.run_id.startswith("s003-"):
            raise ProvenanceError("run_id must start with s003-")
        if not _is_sha256(self.data_digest) or not _is_sha256(self.config_digest):
            raise ProvenanceError("data_digest and config_digest must be SHA-256")
        if len(self.code_revision) != 40:
            raise ProvenanceError("code_revision must be a 40-character commit hash")
        if not self.checkpoint_digest:
            canonical = json.dumps(
                {
                    "run_id": self.run_id,
                    "epoch": self.epoch,
                    "phase": self.phase,
                    "data_digest": self.data_digest,
                    "config_digest": self.config_digest,
                    "code_revision": self.code_revision,
                },
                sort_keys=True,
            )
            self.checkpoint_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def capture_rng_states(cls) -> dict[str, Any]:
        """Capture Python, NumPy, and PyTorch RNG states."""
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @classmethod
    def restore_rng_states(cls, states: Mapping[str, Any]) -> None:
        """Restore Python, NumPy, and PyTorch RNG states."""
        if "python" in states and states["python"] is not None:
            random.setstate(states["python"])
        if "numpy" in states and states["numpy"] is not None:
            np.random.set_state(states["numpy"])
        if "torch" in states and states["torch"] is not None:
            torch.set_rng_state(states["torch"])
        if "torch_cuda" in states and states["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])

    def assert_compatible(
        self,
        *,
        data_digest: str,
        config_digest: str,
        code_revision: str,
    ) -> None:
        """Verify checkpoint compatibility before resuming."""
        if self.data_digest != data_digest:
            raise ProvenanceError("checkpoint data_digest mismatch")
        if self.config_digest != config_digest:
            raise ProvenanceError("checkpoint config_digest mismatch")
        if self.code_revision != code_revision:
            raise ProvenanceError("checkpoint code_revision mismatch")

    def save_atomic(self, target_path: str | Path) -> Path:
        """Save checkpoint to disk using atomic rename."""
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target.parent / f".{target.name}.tmp"
        payload = {
            "run_id": self.run_id,
            "epoch": self.epoch,
            "phase": self.phase,
            "data_digest": self.data_digest,
            "config_digest": self.config_digest,
            "code_revision": self.code_revision,
            "dependencies": dict(self.dependencies),
            "rng_states": self.rng_states,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "created_at": self.created_at,
            "checkpoint_digest": self.checkpoint_digest,
        }
        torch.save(payload, temp_file)
        temp_file.replace(target)
        return target

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> RunCheckpoint:
        """Load and deserialize checkpoint from disk."""
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise ProvenanceError(f"checkpoint file not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            run_id=payload["run_id"],
            epoch=payload["epoch"],
            phase=payload["phase"],
            data_digest=payload["data_digest"],
            config_digest=payload["config_digest"],
            code_revision=payload["code_revision"],
            dependencies=payload["dependencies"],
            rng_states=payload["rng_states"],
            model_state_dict=payload["model_state_dict"],
            optimizer_state_dict=payload.get("optimizer_state_dict"),
            created_at=payload.get("created_at", _now_iso()),
            checkpoint_digest=payload.get("checkpoint_digest", ""),
        )


@dataclass(frozen=True)
class LineageReconstruction:
    """Full chain of custody reconstruction for an arbitrary result (SC-006, FR-026)."""

    run_id: str
    cell_key: str
    study_id: str
    method: str
    fraction: float
    seed: int
    variant: str
    data_digest: str
    config_digest: str
    code_revision: str
    state: str
    threshold: float | None
    weights_digest: str | None
    test_labels_digest: str | None
    scores_digest: str | None
    technical_rerun_of: str | None
    environment: Mapping[str, Any]
    metrics: Mapping[str, Any]
    source_artifacts: Sequence[str]
    isolated_from_s002: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reconstruct_result_lineage(
    run_id: str,
    *,
    artifacts_root: str | Path | None = None,
    matrix_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> LineageReconstruction:
    """Reconstruct complete lineage and decisions for an arbitrary run_id without S02 state (SC-006, FR-026)."""
    if not run_id.startswith("s003-"):
        raise ProvenanceError(f"run_id must start with s003-: {run_id}")

    candidate_dirs: list[Path] = []
    if matrix_dir is not None:
        candidate_dirs.append(Path(matrix_dir).resolve())
    if run_dir is not None:
        candidate_dirs.append(Path(run_dir).resolve())
    if artifacts_root is not None:
        root_path = Path(artifacts_root).resolve()
        candidate_dirs.append(root_path / "runs" / run_id)
        if (root_path / "matrices").is_dir():
            candidate_dirs.extend(sorted((root_path / "matrices").iterdir(), key=lambda p: p.name, reverse=True))

    source_artifacts: list[str] = []
    matched_cell: dict[str, Any] | None = None
    matrix_payload: dict[str, Any] | None = None
    eval_payload: dict[str, Any] | None = None
    run_payload: dict[str, Any] | None = None

    for directory in candidate_dirs:
        if not directory.exists():
            continue

        # Check for direct run.json
        run_file = directory / "run.json" if directory.is_dir() else directory
        if run_file.is_file() and run_file.name == "run.json":
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                payload = data.get("payload", data)
                if payload.get("run_id") == run_id:
                    run_payload = payload
                    source_artifacts.append(str(run_file))
            except Exception:
                pass

        # Check for matrix.json
        matrix_file = directory / "matrix.json"
        if matrix_file.is_file():
            try:
                data = json.loads(matrix_file.read_text(encoding="utf-8"))
                cells = data.get("cells", [])
                cell_items = cells.values() if isinstance(cells, dict) else cells
                for item in cell_items:
                    if item.get("run_id") == run_id or item.get("key") == run_id:
                        matched_cell = item
                        matrix_payload = data
                        source_artifacts.append(str(matrix_file))
                        break
            except Exception:
                pass

        # Check for evaluations.json or report.json
        for candidate_name in ("evaluations.json", "report.json"):
            report_file = directory / candidate_name
            if report_file.is_file():
                try:
                    data = json.loads(report_file.read_text(encoding="utf-8"))
                    evals = data.get("evaluations", data.get("payload", {}).get("evaluations", {}))
                    if isinstance(evals, dict) and run_id in evals:
                        eval_payload = evals[run_id]
                        source_artifacts.append(str(report_file))
                except Exception:
                    pass

        if run_payload is not None or matched_cell is not None:
            break

    if run_payload is None and matched_cell is None:
        raise ProvenanceError(f"cannot reconstruct lineage: run '{run_id}' not found in provided paths")

    cell_key = ""
    state = "unknown"
    data_digest = "0" * 64
    config_digest = "0" * 64
    code_revision = "0" * 40
    weights_digest = None
    threshold = None
    technical_rerun_of = None

    if matched_cell is not None:
        cell_key = matched_cell.get("key", "")
        state = matched_cell.get("state", "selected")
        weights_digest = matched_cell.get("weights_digest")
        threshold = matched_cell.get("threshold")
        config_digest = matched_cell.get("config_digest", config_digest)
        if matrix_payload:
            data_digest = matrix_payload.get("data_digest", data_digest)

    if run_payload is not None:
        state = run_payload.get("state", state)
        cell_key = run_payload.get("cell_key", cell_key)
        data_digest = run_payload.get("data_digest", data_digest)
        config_digest = run_payload.get("config_digest", config_digest)
        code_rev_val = run_payload.get("code_revision", code_revision)
        code_revision = code_rev_val.get("commit", "0" * 40) if isinstance(code_rev_val, dict) else str(code_rev_val)
        weights_digest = run_payload.get("weights_digest", weights_digest)
        threshold = run_payload.get("threshold", threshold)
        technical_rerun_of = run_payload.get("technical_rerun_of")

    family = "main"
    method = "unknown"
    variant = "complete"
    fraction = 0.01
    seed = 11

    if cell_key:
        parts = cell_key.split(":")
        if len(parts) == 4:
            family, ident, frac_text, seed_text = parts
            try:
                fraction = float(frac_text)
                seed = int(seed_text)
            except ValueError:
                pass
            if family == "main":
                method = ident
                variant = "complete"
            elif family == "representation":
                method = "s003_txgcl"
                variant = ident
            elif family in {"ablation", "p2"}:
                method = "s003_txgcl"
                variant = ident

    test_labels_digest = None
    scores_digest = None
    metrics: dict[str, Any] = {}

    if eval_payload is not None:
        test_labels_digest = eval_payload.get("test_labels_digest")
        scores_digest = eval_payload.get("scores_digest")
        metrics = eval_payload.get("pooled", eval_payload)

    all_artifact_strings = " ".join(source_artifacts).lower()
    if "s02" in all_artifact_strings or "s002" in all_artifact_strings:
        raise ProvenanceError("study isolation violated: lineage references Study 002 artifact")

    env_manifest = EnvironmentManifest.capture()

    return LineageReconstruction(
        run_id=run_id,
        cell_key=cell_key,
        study_id="s003",
        method=method,
        fraction=fraction,
        seed=seed,
        variant=variant,
        data_digest=data_digest,
        config_digest=config_digest,
        code_revision=code_revision if len(code_revision) == 40 else "0" * 40,
        state=state,
        threshold=threshold,
        weights_digest=weights_digest,
        test_labels_digest=test_labels_digest,
        scores_digest=scores_digest,
        technical_rerun_of=technical_rerun_of,
        environment=env_manifest.to_dict(),
        metrics=metrics,
        source_artifacts=source_artifacts,
        isolated_from_s002=True,
    )

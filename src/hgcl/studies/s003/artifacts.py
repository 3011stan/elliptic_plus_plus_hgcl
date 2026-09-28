"""Immutable S003 artifact storage and lifecycle primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


class ArtifactError(RuntimeError):
    """Raised when an artifact or lifecycle invariant is violated."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


class ArtifactStore:
    """Write-once artifact store rooted below ``artifacts/s003``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _target(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ArtifactError("artifact path must be relative")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ArtifactError("artifact path escapes store root")
        return target

    @staticmethod
    def sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_once(self, relative_path: str | Path, content: bytes) -> Path:
        target = self._target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ArtifactError(f"artifact already exists: {relative_path}")
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if target.exists():
                raise ArtifactError(f"artifact already exists: {relative_path}")
            os.replace(temporary, target)
            temporary = None
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def write_json(self, relative_path: str | Path, payload: Mapping[str, Any]) -> Path:
        encoded = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        return self._write_once(relative_path, encoded)

    def write_bytes(self, relative_path: str | Path, payload: bytes) -> Path:
        return self._write_once(relative_path, payload)

    def envelope(
        self,
        *,
        artifact_type: str,
        config_digest: str,
        data_digest: str,
        code_revision: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not _is_sha256(config_digest) or not _is_sha256(data_digest):
            raise ArtifactError("config_digest and data_digest must be SHA-256")
        return {
            "schema_version": 1,
            "study_id": "s003",
            "artifact_type": artifact_type,
            "config_digest": config_digest,
            "data_digest": data_digest,
            "code_revision": dict(code_revision),
            "payload": dict(payload),
        }

    def file_record(self, path: str | Path) -> dict[str, Any]:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ArtifactError("registered file must exist inside the artifact store")
        return {
            "path": resolved.relative_to(self.root).as_posix(),
            "size_bytes": resolved.stat().st_size,
            "sha256": self.sha256(resolved),
        }


@dataclass(frozen=True)
class EmbeddingCacheKey:
    method_id: str
    variant_id: str
    seed: int
    data_digest: str
    config_digest: str
    code_revision: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.data_digest) or not _is_sha256(self.config_digest):
            raise ArtifactError("embedding cache requires data/config SHA-256 digests")
        if len(self.code_revision) != 40:
            raise ArtifactError("embedding cache requires a commit revision")

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DryRunApproval:
    approval_id: str
    approved_by: str
    data_digest: str
    dry_run_config_digest: str
    lab_config_digest: str
    code_revision: str
    evidence_digest: str
    design_digest: str
    max_projected_duration_seconds: int
    approved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.approval_id.startswith("s003-"):
            raise ArtifactError("approval_id must start with s003-")
        for name in (
            "data_digest",
            "dry_run_config_digest",
            "lab_config_digest",
            "evidence_digest",
            "design_digest",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ArtifactError(f"{name} must be SHA-256")
        if len(self.code_revision) != 40:
            raise ArtifactError("code_revision must identify one commit")
        if self.max_projected_duration_seconds <= 0:
            raise ArtifactError("projected duration must be positive")

    def assert_compatible(
        self,
        *,
        data_digest: str,
        lab_config_digest: str,
        code_revision: str,
        design_digest: str,
    ) -> None:
        expected = {
            "data_digest": data_digest,
            "lab_config_digest": lab_config_digest,
            "code_revision": code_revision,
            "design_digest": design_digest,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ArtifactError(f"approval mismatch: {name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationCell:
    key: str
    state: str
    run_id: str | None = None
    weights_digest: str | None = None
    threshold: float | None = None
    config_digest: str | None = None

    def __post_init__(self) -> None:
        if self.state not in {"selected", "failed", "invalid"}:
            raise ArtifactError(f"non-terminal cohort cell state: {self.state}")
        if self.state == "selected":
            if not self.run_id or not self.run_id.startswith("s003-"):
                raise ArtifactError("selected cell requires an S003 run_id")
            if not self.weights_digest or not _is_sha256(self.weights_digest):
                raise ArtifactError("selected cell requires weights_digest")
            if not self.config_digest or not _is_sha256(self.config_digest):
                raise ArtifactError("selected cell requires config_digest")
            if self.threshold is None:
                raise ArtifactError("selected cell requires a frozen threshold")


@dataclass
class EvaluationCohort:
    cohort_id: str
    design_digest: str
    test_store_digest: str
    cells: tuple[EvaluationCell, ...] = ()
    state: str = "draft"

    def __post_init__(self) -> None:
        if not self.cohort_id.startswith("s003-"):
            raise ArtifactError("cohort_id must start with s003-")
        if not _is_sha256(self.design_digest) or not _is_sha256(self.test_store_digest):
            raise ArtifactError("cohort digests must be SHA-256")
        self.cells = tuple(self.cells)

    def add_cell(self, cell: EvaluationCell) -> None:
        if self.state != "draft":
            raise ArtifactError("cohort is sealed and cannot accept cells")
        if any(existing.key == cell.key for existing in self.cells):
            raise ArtifactError(f"duplicate cohort cell: {cell.key}")
        self.cells = (*self.cells, cell)

    def seal(self) -> None:
        if self.state != "draft":
            raise ArtifactError(f"cannot seal cohort in state {self.state}")
        if len(self.cells) != 205 or len({cell.key for cell in self.cells}) != 205:
            raise ArtifactError("cohort must contain exactly 205 unique cells")
        if not any(cell.state == "selected" for cell in self.cells):
            raise ArtifactError("cohort must contain at least one selected cell")
        self.state = "sealed"

    def release(self) -> None:
        if self.state == "released":
            raise ArtifactError("cohort test labels were already released")
        if self.state != "sealed":
            raise ArtifactError("only a sealed cohort may release test labels")
        self.state = "released"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "design_digest": self.design_digest,
            "test_store_digest": self.test_store_digest,
            "state": self.state,
            "cells": [asdict(cell) for cell in self.cells],
        }

    def persist(self, store: ArtifactStore, relative_path: str | Path) -> Path:
        return store.write_json(relative_path, self.to_dict())

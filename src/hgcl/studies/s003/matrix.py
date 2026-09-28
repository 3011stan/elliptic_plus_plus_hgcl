"""Resumable, test-blind scheduling for the S003 experiment matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Mapping

from .artifacts import ArtifactError, EmbeddingCacheKey, EvaluationCell, EvaluationCohort
from .evaluation import TestLabelStore


class MatrixScheduleError(RuntimeError):
    """Raised when a matrix lifecycle invariant is violated."""


@dataclass
class MatrixCellState:
    key: str
    state: str = "planned"
    run_id: str | None = None
    weights_digest: str | None = None
    threshold: float | None = None
    config_digest: str | None = None
    failure_kind: str | None = None
    failure_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MatrixScheduler:
    """Track one immutable design until every P1 cell can enter the cohort.

    Scheduling never reads test labels. A run may only be resumed after an
    explicit interruption; failed and invalid cells remain terminal and are
    still represented in the final 205-cell accounting.
    """

    ACTIVE_STATES = {"planned", "running", "interrupted"}
    COHORT_STATES = {"selected", "failed", "invalid"}

    def __init__(
        self,
        cells: tuple[str, ...] | list[str],
        *,
        design_digest: str,
        test_labels: TestLabelStore,
        expected_count: int = 205,
    ) -> None:
        keys = tuple(cells)
        if len(keys) != expected_count or len(set(keys)) != expected_count:
            raise MatrixScheduleError(f"matrix must contain exactly {expected_count} unique cells")
        actual = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
        if actual != design_digest:
            raise MatrixScheduleError("matrix design digest mismatch")
        if test_labels.access_log or test_labels.released_cohort is not None:
            raise MatrixScheduleError("matrix scheduling requires an unopened test-label store")
        self.design_digest = design_digest
        self.test_labels = test_labels
        self.expected_count = expected_count
        self.cells = {key: MatrixCellState(key) for key in keys}

    def _cell(self, key: str) -> MatrixCellState:
        try:
            return self.cells[key]
        except KeyError as exc:
            raise MatrixScheduleError(f"unknown matrix cell: {key}") from exc

    def _assert_test_blind(self) -> None:
        if self.test_labels.access_log or self.test_labels.released_cohort is not None:
            raise MatrixScheduleError("test labels were accessed before cohort sealing")

    def start(self, key: str) -> MatrixCellState:
        self._assert_test_blind()
        cell = self._cell(key)
        if cell.state != "planned":
            raise MatrixScheduleError("only a planned cell can start")
        cell.state = "running"
        return cell

    def interrupt(self, key: str, message: str) -> MatrixCellState:
        cell = self._cell(key)
        if cell.state != "running":
            raise MatrixScheduleError("only a running cell can be interrupted")
        cell.state = "interrupted"
        cell.failure_kind = "interrupted"
        cell.failure_message = message
        return cell

    def resume(self, key: str) -> MatrixCellState:
        self._assert_test_blind()
        cell = self._cell(key)
        if cell.state != "interrupted":
            raise MatrixScheduleError("only an interrupted cell can resume")
        cell.state = "running"
        cell.failure_kind = None
        cell.failure_message = None
        return cell

    def select(
        self,
        key: str,
        *,
        run_id: str,
        weights_digest: str,
        threshold: float,
        config_digest: str,
    ) -> MatrixCellState:
        self._assert_test_blind()
        cell = self._cell(key)
        if cell.state != "running":
            raise MatrixScheduleError("only a running cell can be selected")
        # Reuse EvaluationCell validation so scheduling and evaluation agree.
        EvaluationCell(key, "selected", run_id, weights_digest, threshold, config_digest)
        cell.state = "selected"
        cell.run_id = run_id
        cell.weights_digest = weights_digest
        cell.threshold = threshold
        cell.config_digest = config_digest
        return cell

    def terminate(self, key: str, *, state: str, kind: str, message: str) -> MatrixCellState:
        self._assert_test_blind()
        cell = self._cell(key)
        if state not in {"failed", "invalid"}:
            raise MatrixScheduleError("terminal failure state must be failed or invalid")
        if cell.state not in {"planned", "running", "interrupted"}:
            raise MatrixScheduleError("selected or terminal cells are immutable")
        if not kind or not message:
            raise MatrixScheduleError("failure kind and message are required")
        cell.state = state
        cell.failure_kind = kind
        cell.failure_message = message
        return cell

    def pending(self) -> tuple[str, ...]:
        return tuple(key for key, cell in self.cells.items() if cell.state in self.ACTIVE_STATES)

    def seal_cohort(self, cohort_id: str) -> EvaluationCohort:
        self._assert_test_blind()
        if self.expected_count != 205:
            raise MatrixScheduleError("only the complete 205-cell P1 design may be sealed")
        if self.pending():
            raise MatrixScheduleError("all matrix cells must be selected or terminal before sealing")
        cohort = EvaluationCohort(cohort_id, self.design_digest, self.test_labels.digest)
        for cell in self.cells.values():
            cohort.add_cell(
                EvaluationCell(
                    key=cell.key,
                    state=cell.state,
                    run_id=cell.run_id,
                    weights_digest=cell.weights_digest,
                    threshold=cell.threshold,
                    config_digest=cell.config_digest,
                )
            )
        cohort.seal()
        return cohort

    def to_dict(self) -> dict[str, object]:
        return {
            "design_digest": self.design_digest,
            "expected_count": self.expected_count,
            "test_store_digest": self.test_labels.digest,
            "test_access_count": len(self.test_labels.access_log),
            "cells": [cell.to_dict() for cell in self.cells.values()],
        }


def embedding_cache_key_for_cell(
    cell_key: str,
    *,
    data_digest: str,
    pretraining_config_digest: str,
    code_revision: str,
) -> EmbeddingCacheKey:
    """Build a fraction-independent key for reusable frozen embeddings."""
    parts = cell_key.split(":")
    if len(parts) != 4:
        raise MatrixScheduleError(f"invalid canonical cell key: {cell_key}")
    family, variant, _fraction, seed_text = parts
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise MatrixScheduleError(f"invalid seed in cell key: {cell_key}") from exc
    return EmbeddingCacheKey(
        method_id=variant,
        variant_id=family,
        seed=seed,
        data_digest=data_digest,
        config_digest=pretraining_config_digest,
        code_revision=code_revision,
    )

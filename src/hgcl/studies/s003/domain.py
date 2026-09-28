"""Typed domain entities for the S003 experiment lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Hashable, Mapping


class DomainError(ValueError):
    """Raised when a scientific domain invariant is violated."""


def _is_digest(value: str, length: int = 64) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value.lower())


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    files: Mapping[str, str]
    dataset_id: str = "elliptic-plus-plus"

    def __post_init__(self) -> None:
        required = {"txs_features.csv", "txs_classes.csv", "txs_edgelist.csv"}
        if not required.issubset(self.files):
            raise DomainError(f"missing core files: {sorted(required - self.files.keys())}")
        if any(not _is_digest(digest) for digest in self.files.values()):
            raise DomainError("source file digests must be SHA-256")


@dataclass(frozen=True)
class TemporalSnapshot:
    time_step: int
    tx_ids: tuple[Hashable, ...]
    edges: tuple[tuple[Hashable, Hashable], ...]

    def __post_init__(self) -> None:
        if not 1 <= self.time_step <= 49:
            raise DomainError("time_step must be in 1..49")
        if len(set(self.tx_ids)) != len(self.tx_ids):
            raise DomainError("duplicate transaction IDs")
        known = set(self.tx_ids)
        if any(source not in known or target not in known for source, target in self.edges):
            raise DomainError("edge references a transaction outside the snapshot")

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class LabelBudget:
    seed: int
    fraction: float
    fit_ids: frozenset[Hashable]
    validation_ids: frozenset[Hashable]
    refit_ids: frozenset[Hashable]
    fit_class_counts: Mapping[str, int]
    validation_class_counts: Mapping[str, int]
    parent_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.seed not in {11, 23, 37, 53, 71}:
            raise DomainError("unsupported seed")
        if self.fraction not in {0.01, 0.05, 0.10, 1.00}:
            raise DomainError("unsupported label fraction")
        if self.fit_ids & self.validation_ids:
            raise DomainError("fit and validation IDs must be disjoint")
        if self.refit_ids != self.fit_ids | self.validation_ids:
            raise DomainError("refit IDs must equal fit union validation")
        for counts in (self.fit_class_counts, self.validation_class_counts):
            if counts.get("illicit", 0) < 1 or counts.get("licit", 0) < 1:
                raise DomainError("both known classes are required")
        if self.parent_fraction is not None and self.parent_fraction <= self.fraction:
            raise DomainError("parent fraction must be larger")


class RunState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SELECTED = "selected"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    INVALID = "invalid"
    FAILED = "failed"


TERMINAL_STATES = {RunState.COMPLETED, RunState.INVALID, RunState.FAILED}
TRANSITIONS = {
    RunState.PLANNED: {RunState.RUNNING, RunState.INVALID, RunState.FAILED},
    RunState.RUNNING: {RunState.SELECTED, RunState.INTERRUPTED, RunState.INVALID, RunState.FAILED},
    RunState.SELECTED: {RunState.EVALUATING, RunState.INTERRUPTED, RunState.INVALID, RunState.FAILED},
    RunState.EVALUATING: {RunState.COMPLETED, RunState.INTERRUPTED, RunState.FAILED},
}


@dataclass
class ExperimentRun:
    run_id: str
    state: RunState = RunState.PLANNED
    _resume_state: RunState | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.run_id.startswith("s003-"):
            raise DomainError("run_id must start with s003-")

    def transition(self, target: RunState) -> None:
        if self.state in TERMINAL_STATES:
            raise DomainError(f"terminal state is immutable: {self.state.value}")
        if target not in TRANSITIONS.get(self.state, set()):
            raise DomainError(f"invalid transition: {self.state.value} -> {target.value}")
        if target is RunState.INTERRUPTED:
            self._resume_state = self.state
        self.state = target

    def resume(self) -> None:
        if self.state is not RunState.INTERRUPTED or self._resume_state is None:
            raise DomainError("only an interrupted run can resume")
        self.state = self._resume_state
        self._resume_state = None


@dataclass(frozen=True)
class StatisticalComparison:
    differences: tuple[float | None, ...]

    def __post_init__(self) -> None:
        if len(self.differences) != 5:
            raise DomainError("a primary comparison must declare five seed pairs")

    @property
    def inferential(self) -> bool:
        return all(value is not None for value in self.differences)

    @property
    def inferential_fields_are_null(self) -> bool:
        return not self.inferential

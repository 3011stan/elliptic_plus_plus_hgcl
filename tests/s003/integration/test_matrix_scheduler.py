from __future__ import annotations

import pytest

from hgcl.studies.s003.config import ALL_FRACTIONS, ALL_METHODS, ALL_SEEDS
from hgcl.studies.s003.evaluation import TestLabelStore
from hgcl.studies.s003.matrix import (
    MatrixScheduleError,
    MatrixScheduler,
    embedding_cache_key_for_cell,
)
from hgcl.studies.s003.pipeline import canonical_matrix_design


def _scheduler() -> MatrixScheduler:
    design = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS)
    return MatrixScheduler(
        design["cells"],
        design_digest=design["design_digest"],
        test_labels=TestLabelStore.seal({35: {"tx": 1}}),
    )


def test_scheduler_resumes_only_interrupted_cells_and_preserves_failures() -> None:
    scheduler = _scheduler()
    first, second = tuple(scheduler.cells)[:2]
    scheduler.start(first)
    scheduler.interrupt(first, "preemption")
    scheduler.resume(first)
    scheduler.terminate(first, state="failed", kind="resource", message="out of memory")
    scheduler.terminate(second, state="invalid", kind="data", message="invalid split")
    assert scheduler.cells[first].failure_kind == "resource"
    assert scheduler.cells[second].state == "invalid"
    with pytest.raises(MatrixScheduleError, match="only an interrupted"):
        scheduler.resume(second)
    with pytest.raises(MatrixScheduleError, match="immutable"):
        scheduler.terminate(first, state="failed", kind="retry", message="forbidden")


def test_embedding_cache_is_reused_across_label_fractions() -> None:
    common = {"data_digest": "a" * 64, "pretraining_config_digest": "b" * 64, "code_revision": "c" * 40}
    one = embedding_cache_key_for_cell("main:s003_txgcl:0.01:11", **common)
    full = embedding_cache_key_for_cell("main:s003_txgcl:1:11", **common)
    other_seed = embedding_cache_key_for_cell("main:s003_txgcl:1:23", **common)
    assert one.digest == full.digest
    assert one.digest != other_seed.digest


def test_complete_accounting_seals_cohort_without_test_access() -> None:
    scheduler = _scheduler()
    for index, key in enumerate(scheduler.cells):
        if index == 0:
            scheduler.start(key)
            scheduler.select(
                key,
                run_id="s003-selected-0",
                weights_digest="a" * 64,
                threshold=0.42,
                config_digest="b" * 64,
            )
        else:
            scheduler.terminate(key, state="failed", kind="fixture", message="not executed")
    cohort = scheduler.seal_cohort("s003-cohort-p1")
    assert cohort.state == "sealed"
    assert len(cohort.cells) == 205
    assert scheduler.test_labels.access_log == []
    assert scheduler.test_labels.released_cohort is None


def test_incomplete_or_previously_opened_schedule_cannot_seal() -> None:
    scheduler = _scheduler()
    with pytest.raises(MatrixScheduleError, match="all matrix cells"):
        scheduler.seal_cohort("s003-cohort-p1")

    opened = TestLabelStore.seal({35: {"tx": 1}})
    opened.release("s003-old-cohort", {"s003-old": {"weights_digest": "a" * 64, "threshold": 0.5}})
    design = canonical_matrix_design(ALL_METHODS, ALL_FRACTIONS, ALL_SEEDS)
    with pytest.raises(MatrixScheduleError, match="unopened"):
        MatrixScheduler(design["cells"], design_digest=design["design_digest"], test_labels=opened)

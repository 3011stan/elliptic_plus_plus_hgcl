from dataclasses import dataclass
from pathlib import Path
import pytest
import torch

from hgcl.studies.s003.artifacts import EmbeddingCacheKey
from hgcl.studies.s003.baselines import GraphData
from hgcl.studies.s003.config import ALL_FRACTIONS, ALL_METHODS, ALL_SEEDS
from hgcl.studies.s003.evaluation import TestLabelStore
from hgcl.studies.s003.matrix import (
    EmbeddingCacheStore,
    MatrixScheduleError,
    MatrixScheduler,
    embedding_cache_key_for_cell,
    execute_matrix_cell,
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


@dataclass
class _DummyBudget:
    fit_ids: frozenset[str]
    validation_ids: frozenset[str]
    refit_ids: frozenset[str]
    digest: str = "a" * 64


def _synthetic_dataset():
    ids = tuple(f"tx-{i}" for i in range(20))
    features = torch.randn(20, 182)
    labels = torch.tensor([0, 1] * 10)
    edge_index = torch.tensor([list(range(19)), list(range(1, 20))], dtype=torch.long)
    dataset = GraphData(ids, features, labels, edge_index)
    budget = _DummyBudget(
        fit_ids=frozenset(ids[:12]),
        validation_ids=frozenset(ids[12:16]),
        refit_ids=frozenset(ids[:16]),
    )
    return dataset, budget


def test_scheduler_from_dict_and_resumption() -> None:
    scheduler = _scheduler()
    keys = list(scheduler.cells)
    k1, k2, k3 = keys[0], keys[1], keys[2]

    scheduler.start(k1)
    scheduler.interrupt(k1, "preemption")

    scheduler.start(k2)
    scheduler.terminate(k2, state="failed", kind="resource", message="out of memory")

    scheduler.start(k3)
    scheduler.select(
        k3,
        run_id="s003-run-003",
        weights_digest="a" * 64,
        threshold=0.45,
        config_digest="b" * 64,
    )

    data = scheduler.to_dict()
    restored = MatrixScheduler.from_dict(data, test_labels=scheduler.test_labels)

    assert restored.cells[k1].state == "interrupted"
    assert restored.cells[k2].state == "failed"
    assert restored.cells[k2].failure_kind == "resource"
    assert restored.cells[k3].state == "selected"
    assert restored.cells[k3].threshold == 0.45

    # Resume interrupted cell
    restored.resume(k1)
    assert restored.cells[k1].state == "running"

    # Terminal cells cannot be resumed
    with pytest.raises(MatrixScheduleError, match="only an interrupted"):
        restored.resume(k2)
    with pytest.raises(MatrixScheduleError, match="only an interrupted"):
        restored.resume(k3)


def test_embedding_cache_store_disk_and_memory(tmp_path: Path) -> None:
    store = EmbeddingCacheStore(root=tmp_path)
    key1 = EmbeddingCacheKey("s003_txgcl", "complete", 11, "a" * 64, "b" * 64, "c" * 40)
    key2 = EmbeddingCacheKey("s003_txgcl", "complete", 23, "a" * 64, "b" * 64, "c" * 40)

    tensor = torch.randn(10, 128)
    store.put(key1, {"embeddings": tensor})

    assert store.has(key1) is True
    assert store.has(key2) is False

    retrieved = store.get(key1)
    assert retrieved is not None
    assert torch.allclose(retrieved["embeddings"], tensor)
    assert store.hits == 1
    assert store.misses == 0

    missing = store.get(key2)
    assert missing is None
    assert store.misses == 1

    # New store instance reading from disk
    disk_store = EmbeddingCacheStore(root=tmp_path)
    assert disk_store.has(key1) is True
    from_disk = disk_store.get(key1)
    assert from_disk is not None
    assert torch.allclose(from_disk["embeddings"], tensor)
    assert disk_store.hits == 1

    # In-memory clear
    disk_store.clear()
    assert disk_store.hits == 0
    assert disk_store.misses == 0


def test_execute_matrix_cell_fraction_cache_reuse_and_test_blindness(tmp_path: Path) -> None:
    dataset, budget = _synthetic_dataset()
    scheduler = _scheduler()
    cache_store = EmbeddingCacheStore(root=tmp_path)

    # 1. First run of s003_txgcl with fraction 0.01: cache miss
    c01 = "main:s003_txgcl:0.01:11"
    res01 = execute_matrix_cell(
        c01,
        scheduler,
        dataset=dataset,
        budget=budget,
        data_digest="a" * 64,
        pretraining_config_digest="b" * 64,
        embedding_cache=cache_store,
        ssl_epochs=1,
        downstream_epochs=1,
    )
    assert res01.state == "selected"
    assert scheduler.cells[c01].state == "selected"
    assert cache_store.misses == 1
    assert cache_store.hits == 0

    # 2. Second run with fraction 0.05: cache hit, reuses embeddings
    c05 = "main:s003_txgcl:0.05:11"
    res05 = execute_matrix_cell(
        c05,
        scheduler,
        dataset=dataset,
        budget=budget,
        data_digest="a" * 64,
        pretraining_config_digest="b" * 64,
        embedding_cache=cache_store,
        ssl_epochs=1,
        downstream_epochs=1,
    )
    assert res05.state == "selected"
    assert cache_store.hits == 1

    # 3. Third run with h_only: cache hit, reuses embeddings
    c_h = "representation:h_only:0.01:11"
    res_h = execute_matrix_cell(
        c_h,
        scheduler,
        dataset=dataset,
        budget=budget,
        data_digest="a" * 64,
        pretraining_config_digest="b" * 64,
        embedding_cache=cache_store,
        ssl_epochs=1,
        downstream_epochs=1,
    )
    assert res_h.state == "selected"
    assert cache_store.hits == 2

    # Zero pre-evaluation test access invariant
    assert scheduler.test_labels.access_log == []
    assert scheduler.test_labels.released_cohort is None


def test_execute_matrix_cell_failure_and_invalid_handling() -> None:
    dataset, budget = _synthetic_dataset()
    scheduler = _scheduler()

    # Trigger invalid cell via corrupted budget that causes indexing issue
    bad_budget = _DummyBudget(
        fit_ids=frozenset({"nonexistent_tx"}),
        validation_ids=frozenset({"nonexistent_tx_2"}),
        refit_ids=frozenset({"nonexistent_tx"}),
    )
    c_bad = "main:mlp_x:0.01:11"
    res_bad = execute_matrix_cell(
        c_bad,
        scheduler,
        dataset=dataset,
        budget=bad_budget,
        data_digest="a" * 64,
        pretraining_config_digest="b" * 64,
        ssl_epochs=1,
        downstream_epochs=1,
    )
    assert res_bad.state == "invalid"
    assert res_bad.failure_kind == "data"

    # Terminal state is immutable
    with pytest.raises(MatrixScheduleError, match="immutable"):
        scheduler.terminate(c_bad, state="failed", kind="retry", message="illegal retry")

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from hgcl.studies.s003.artifacts import DryRunApproval, compute_evidence_digest
from hgcl.studies.s003.baselines import (
    AdapterContext,
    GCPALAdapter,
    GraphData,
    S003TxGCLAdapter,
    TabularData,
)
from hgcl.studies.s003.data import PreparedSnapshot
from hgcl.studies.s003.evaluation import TestLabelStore
from hgcl.studies.s003.matrix import (
    EmbeddingCacheStore,
    MatrixScheduler,
    embedding_cache_key_for_cell,
    execute_matrix_cell,
)
from hgcl.studies.s003.pipeline import PreparedDataset, approve_dry_run, run_matrix_pipeline


def _make_multi_snapshot_data(num_snapshots: int = 3, nodes_per_snapshot: int = 20) -> tuple[GraphData, tuple[PreparedSnapshot, ...], AdapterContext]:
    snapshots = []
    all_ids = []
    all_x = []
    all_labels = []

    for step in range(1, num_snapshots + 1):
        step_ids = tuple(f"tx-s{step}-{i}" for i in range(nodes_per_snapshot))
        step_x = torch.randn(nodes_per_snapshot, 182)
        step_labels = torch.tensor([0, 1] * (nodes_per_snapshot // 2))
        step_edges = torch.tensor([
            list(range(nodes_per_snapshot - 1)),
            list(range(1, nodes_per_snapshot)),
        ], dtype=torch.long)
        snap = PreparedSnapshot(step, step_ids, step_x, step_edges, step_labels, {}, f"snap-{step}")
        snapshots.append(snap)
        all_ids.extend(step_ids)
        all_x.append(step_x)
        all_labels.append(step_labels)

    combined_ids = tuple(all_ids)
    combined_x = torch.cat(all_x)
    combined_labels = torch.cat(all_labels)
    combined_edges = torch.empty((2, 0), dtype=torch.long)

    dataset = GraphData(combined_ids, combined_x, combined_labels, combined_edges, snapshots=tuple(snapshots))

    fit_count = int(len(combined_ids) * 0.6)
    val_count = int(len(combined_ids) * 0.2)
    fit_ids = frozenset(combined_ids[:fit_count])
    val_ids = frozenset(combined_ids[fit_count : fit_count + val_count])
    refit_ids = fit_ids | val_ids

    context = AdapterContext(
        seed=11,
        fraction=0.01,
        fit_ids=fit_ids,
        validation_ids=val_ids,
        refit_ids=refit_ids,
        budget_digest="a" * 64,
    )
    return dataset, tuple(snapshots), context


def test_s003_txgcl_adapter_iterates_multi_snapshots() -> None:
    dataset, snapshots, context = _make_multi_snapshot_data(num_snapshots=3, nodes_per_snapshot=20)
    adapter = S003TxGCLAdapter(ssl_epochs=2, downstream_epochs=2, representation="h_concat_x")

    fitted = adapter.fit(dataset, context)

    assert fitted.method_id == "s003_txgcl"
    assert fitted.embeddings is not None
    assert fitted.embeddings.shape == (60, 128)
    assert fitted.downstream.estimator.input_dim == 310


@pytest.mark.parametrize("rep,width", [("x_only", 182), ("h_only", 128), ("h_concat_x", 310)])
def test_s003_txgcl_representations_multi_snapshots(rep: str, width: int) -> None:
    dataset, snapshots, context = _make_multi_snapshot_data(num_snapshots=2, nodes_per_snapshot=20)
    adapter = S003TxGCLAdapter(ssl_epochs=1, downstream_epochs=1, representation=rep)

    fitted = adapter.fit(dataset, context)

    assert fitted.embeddings is not None
    assert fitted.embeddings.shape == (40, 128)
    assert fitted.downstream.estimator.input_dim == width


def test_gcpal_adapter_iterates_multi_snapshots() -> None:
    dataset, snapshots, context = _make_multi_snapshot_data(num_snapshots=3, nodes_per_snapshot=20)
    adapter = GCPALAdapter(ssl_epochs=2, downstream_epochs=2, knn_k=5)

    fitted = adapter.fit(dataset, context)

    assert fitted.method_id == "gcpal"
    assert fitted.embeddings is not None
    assert fitted.embeddings.shape == (60, 128)
    assert fitted.downstream.estimator.input_dim == 310


@pytest.mark.parametrize(
    "variant_kwargs",
    [
        {"use_knn": False},
        {"edge_dropout": False},
        {"masking_policy": "random_individual"},
        {"masking_policy": "random_groups"},
        {"reverse_edges": True, "reverse_edges_approved": True},
    ],
)
def test_s003_txgcl_ablations_multi_snapshots(variant_kwargs: dict) -> None:
    dataset, snapshots, context = _make_multi_snapshot_data(num_snapshots=2, nodes_per_snapshot=16)
    adapter = S003TxGCLAdapter(ssl_epochs=1, downstream_epochs=1, **variant_kwargs)

    fitted = adapter.fit(dataset, context)

    assert fitted.embeddings is not None
    assert fitted.embeddings.shape == (32, 128)


def test_execute_matrix_cell_caching_and_reuse(tmp_path: Path) -> None:
    dataset, snapshots, context = _make_multi_snapshot_data(num_snapshots=2, nodes_per_snapshot=20)
    cache_store = EmbeddingCacheStore(tmp_path / "cache")

    keys = ["main:s003_txgcl:0.01:11", "representation:h_only:0.01:11"]
    import hashlib
    design_digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    test_store = TestLabelStore.seal({35: {"tx-test": 1}})
    scheduler = MatrixScheduler(keys, design_digest=design_digest, test_labels=test_store, expected_count=2)

    budget = SimpleNamespace(
        fit_ids=sorted(context.fit_ids),
        validation_ids=sorted(context.validation_ids),
        refit_ids=sorted(context.refit_ids),
        digest=context.budget_digest,
    )

    # 1. Execute main cell: should train SSL, extract embeddings snapshot-by-snapshot, and cache
    state_main = execute_matrix_cell(
        keys[0],
        scheduler,
        dataset=dataset,
        budget=budget,
        data_digest="d" * 64,
        pretraining_config_digest="c" * 64,
        code_revision="r" * 40,
        config_digest="c" * 64,
        embedding_cache=cache_store,
        ssl_epochs=1,
        downstream_epochs=1,
    )
    assert state_main.state == "selected"

    cache_key = embedding_cache_key_for_cell(
        keys[0],
        data_digest="d" * 64,
        pretraining_config_digest="c" * 64,
        code_revision="r" * 40,
    )
    cached_payload = cache_store.get(cache_key)
    assert cached_payload is not None
    assert "embeddings" in cached_payload
    assert cached_payload["embeddings"].shape == (40, 128)

    # 2. Execute representation:h_only cell: should hit cache and succeed without re-training SSL
    state_h_only = execute_matrix_cell(
        keys[1],
        scheduler,
        dataset=dataset,
        budget=budget,
        data_digest="d" * 64,
        pretraining_config_digest="c" * 64,
        code_revision="r" * 40,
        config_digest="c" * 64,
        embedding_cache=cache_store,
        ssl_epochs=1,
        downstream_epochs=1,
    )
    assert state_h_only.state == "selected"


def test_matrix_runner_failure_observability(tmp_path: Path) -> None:
    dataset, snapshots, context = _make_multi_snapshot_data(num_snapshots=1, nodes_per_snapshot=20)
    keys = ["main:s003_txgcl:0.01:11"]
    import hashlib
    design_digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    test_store = TestLabelStore.seal({35: {"tx-test": 1}})
    scheduler = MatrixScheduler(keys, design_digest=design_digest, test_labels=test_store, expected_count=1)

    bad_budget = SimpleNamespace(
        fit_ids=["non-existent-id"],
        validation_ids=["non-existent-id-2"],
        refit_ids=["non-existent-id", "non-existent-id-2"],
        digest="b" * 64,
    )

    state = execute_matrix_cell(
        keys[0],
        scheduler,
        dataset=dataset,
        budget=bad_budget,
        data_digest="d" * 64,
        pretraining_config_digest="c" * 64,
    )
    assert state.state in {"failed", "invalid"}
    assert state.failure_message is not None
    assert state.failure_kind is not None

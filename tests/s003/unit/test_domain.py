from __future__ import annotations

from pathlib import Path

import pytest

from hgcl.studies.s003.domain import (
    DomainError,
    ExperimentRun,
    LabelBudget,
    RunState,
    SourceDataset,
    StatisticalComparison,
    TemporalSnapshot,
)


def test_source_dataset_requires_core_files_and_external_root(tmp_path: Path) -> None:
    files = {
        "txs_features.csv": "a" * 64,
        "txs_classes.csv": "b" * 64,
        "txs_edgelist.csv": "c" * 64,
    }
    source = SourceDataset(root=tmp_path / "data", files=files)
    assert source.dataset_id == "elliptic-plus-plus"
    with pytest.raises(DomainError):
        SourceDataset(root=tmp_path / "data", files={"txs_features.csv": "a" * 64})


def test_temporal_snapshot_rejects_duplicate_ids_and_cross_step_edges() -> None:
    snapshot = TemporalSnapshot(time_step=3, tx_ids=(10, 20), edges=((10, 20),))
    assert snapshot.edge_count == 1
    with pytest.raises(DomainError):
        TemporalSnapshot(time_step=3, tx_ids=(10, 10), edges=())
    with pytest.raises(DomainError):
        TemporalSnapshot(time_step=3, tx_ids=(10,), edges=((10, 99),))


def test_label_budget_requires_disjoint_nested_refit_and_both_classes() -> None:
    budget = LabelBudget(
        seed=11,
        fraction=0.01,
        fit_ids=frozenset({1, 2}),
        validation_ids=frozenset({3, 4}),
        refit_ids=frozenset({1, 2, 3, 4}),
        fit_class_counts={"illicit": 1, "licit": 1},
        validation_class_counts={"illicit": 1, "licit": 1},
    )
    assert budget.refit_ids == budget.fit_ids | budget.validation_ids
    with pytest.raises(DomainError):
        LabelBudget(
            seed=11,
            fraction=0.01,
            fit_ids=frozenset({1}),
            validation_ids=frozenset({1, 2}),
            refit_ids=frozenset({1, 2}),
            fit_class_counts={"illicit": 1, "licit": 0},
            validation_class_counts={"illicit": 1, "licit": 1},
        )


def test_run_state_machine_and_terminal_immutability() -> None:
    run = ExperimentRun(run_id="s003-run-001")
    run.transition(RunState.RUNNING)
    run.transition(RunState.SELECTED)
    run.transition(RunState.EVALUATING)
    run.transition(RunState.COMPLETED)
    with pytest.raises(DomainError):
        run.transition(RunState.RUNNING)

    interrupted = ExperimentRun(run_id="s003-run-002")
    interrupted.transition(RunState.RUNNING)
    interrupted.transition(RunState.INTERRUPTED)
    interrupted.resume()
    assert interrupted.state is RunState.RUNNING


def test_statistical_comparison_requires_five_pairs_for_inference() -> None:
    complete = StatisticalComparison(differences=(0.1, 0.2, -0.1, 0.3, 0.2))
    assert complete.inferential
    incomplete = StatisticalComparison(differences=(0.1, 0.2, None, 0.3, 0.2))
    assert not incomplete.inferential
    assert incomplete.inferential_fields_are_null

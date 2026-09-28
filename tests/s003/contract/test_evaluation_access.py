from __future__ import annotations

import pytest

from hgcl.studies.s003.evaluation import (
    EvaluationAccessError,
    TestLabelStore,
    binary_metrics,
    engineering_partition,
    evaluate_frozen_members,
    pooled_and_snapshot_metrics,
)


def test_test_store_single_release_and_frozen_member_access() -> None:
    store = TestLabelStore.seal({35: {"a": 1}, 36: {"b": 0}})
    assert store.labels is None
    members = {"s003-run": {"weights_digest": "a" * 64, "threshold": 0.5}}
    store.release("s003-cohort", members)
    assert store.read("s003-cohort", "s003-run", "a" * 64, 0.5)[35]["a"] == 1
    with pytest.raises(EvaluationAccessError, match="released"):
        store.release("s003-cohort", members)
    with pytest.raises(EvaluationAccessError):
        store.read("s003-cohort", "s003-run", "b" * 64, 0.5)


def test_shadow_partition_is_exact_and_never_test() -> None:
    fit, shadow = engineering_partition(range(1, 50))
    assert fit == tuple(range(1, 30))
    assert shadow == tuple(range(30, 35))
    assert not set(range(35, 50)) & (set(fit) | set(shadow))


def test_pooled_and_snapshot_metrics_preserve_null_reasons() -> None:
    predictions = {
        35: ([1, 1], [0.9, 0.8]),
        36: ([0, 1], [0.1, 0.7]),
    }
    result = pooled_and_snapshot_metrics(predictions, threshold=0.5)
    assert result["pooled"]["f1_illicit"] == 1.0
    assert result["snapshots"][35]["mcc"] is None
    assert "reason" in result["snapshots"][35]
    assert binary_metrics([0, 1], [0.1, 0.9], 0.5)["pr_auc_illicit"] == 1.0


def test_cohort_evaluation_releases_once_and_resumes_pending_members() -> None:
    store = TestLabelStore.seal({35: {"a": 1, "b": 0}})
    members = {
        "s003-a": {"weights_digest": "a" * 64, "threshold": 0.5},
        "s003-b": {"weights_digest": "b" * 64, "threshold": 0.5},
    }
    provider = lambda run_id, step, ids: [0.9 if item == "a" else 0.1 for item in ids]
    first = evaluate_frozen_members(store, "s003-cohort", members, provider, completed={"s003-a": {"cached": True}})
    assert first["s003-a"] == {"cached": True}
    assert first["s003-b"]["pooled"]["f1_illicit"] == 1.0
    assert store.released_cohort == "s003-cohort"

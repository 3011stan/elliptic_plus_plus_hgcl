"""Sealed-label evaluation and S003 metric computation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping

import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationAccessError(RuntimeError):
    """Raised on unauthorized or repeated test-label access."""


class TestLabelStore:
    __test__ = False

    def __init__(self, sealed_labels: Mapping[int, Mapping[str, int]], digest: str) -> None:
        self._sealed = deepcopy(dict(sealed_labels))
        self.digest = digest
        self._release: tuple[str, dict[str, dict]] | None = None
        self.access_log: list[dict[str, Any]] = []

    @classmethod
    def seal(cls, labels: Mapping[int, Mapping[str, int]]) -> "TestLabelStore":
        canonical = json.dumps(labels, sort_keys=True, separators=(",", ":"))
        return cls(labels, hashlib.sha256(canonical.encode()).hexdigest())

    @property
    def labels(self) -> None:
        return None

    @property
    def released_cohort(self) -> str | None:
        return self._release[0] if self._release is not None else None

    def persistence_payload(self) -> dict[int, dict[str, int]]:
        """Return sealed bytes material for the preparation persistence boundary."""
        return deepcopy(self._sealed)

    def release(self, cohort_id: str, members: Mapping[str, Mapping[str, object]]) -> None:
        if self._release is not None:
            raise EvaluationAccessError("test labels were already released")
        if not cohort_id.startswith("s003-") or not members:
            raise EvaluationAccessError("release requires a sealed S003 cohort")
        self._release = (cohort_id, deepcopy(dict(members)))

    def read(
        self,
        cohort_id: str,
        run_id: str,
        weights_digest: str,
        threshold: float,
        *,
        accessor: str = "evaluator",
        purpose: str = "test_evaluation",
        technical_rerun_of: str | None = None,
        technical_failure_justification: str | None = None,
    ) -> dict[int, dict[str, int]]:
        if self._release is None:
            raise EvaluationAccessError("test labels are sealed")
        frozen_cohort, members = self._release
        if cohort_id != frozen_cohort:
            raise EvaluationAccessError("cohort ID mismatch with released cohort")

        target_run_id = technical_rerun_of if technical_rerun_of is not None else run_id
        member = members.get(target_run_id)
        if member is None:
            raise EvaluationAccessError("run is not a frozen cohort member")
        if member.get("weights_digest") != weights_digest or member.get("threshold") != threshold:
            raise EvaluationAccessError("weights or threshold differ from frozen selection")

        if technical_rerun_of is not None:
            if not run_id.startswith("s003-"):
                raise EvaluationAccessError("run_id must start with s003-")
            if run_id == technical_rerun_of:
                raise EvaluationAccessError("technical rerun must have a distinct run_id from original")
            if not technical_failure_justification or not str(technical_failure_justification).strip():
                raise EvaluationAccessError("technical rerun requires a technical failure justification")

        log_entry: dict[str, Any] = {
            "cohort_id": cohort_id,
            "run_id": run_id,
            "timestamp": _now_iso(),
            "accessor": accessor,
            "purpose": purpose,
        }
        if technical_rerun_of is not None:
            log_entry["technical_rerun_of"] = technical_rerun_of
            log_entry["technical_failure_justification"] = technical_failure_justification

        self.access_log.append(log_entry)
        return deepcopy(self._sealed)


def engineering_partition(steps) -> tuple[tuple[int, ...], tuple[int, ...]]:
    available = set(steps)
    return (
        tuple(step for step in range(1, 30) if step in available),
        tuple(step for step in range(30, 35) if step in available),
    )


def decisions_from_scores(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    return (scores >= threshold).to(torch.int64)


def binary_metrics(labels, scores, threshold: float) -> dict[str, object]:
    known_pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if int(y) in (0, 1)]
    if not known_pairs:
        return {
            "support": {"illicit": 0, "licit": 0},
            "f1_illicit": 0.0,
            "precision_illicit": 0.0,
            "recall_illicit": 0.0,
            "mcc": None,
            "pr_auc_illicit": None,
            "reason": "metric requires known classes",
        }
    eval_labels = [y for y, _ in known_pairs]
    eval_scores = [s for _, s in known_pairs]
    predicted = [int(score >= threshold) for score in eval_scores]
    support = {"illicit": eval_labels.count(1), "licit": eval_labels.count(0)}
    result: dict[str, object] = {
        "support": support,
        "f1_illicit": float(f1_score(eval_labels, predicted, pos_label=1, zero_division=0)),
        "precision_illicit": float(precision_score(eval_labels, predicted, pos_label=1, zero_division=0)),
        "recall_illicit": float(recall_score(eval_labels, predicted, pos_label=1, zero_division=0)),
    }
    if len(set(eval_labels)) < 2:
        result.update({"mcc": None, "pr_auc_illicit": None, "reason": "metric requires both known classes"})
    else:
        result.update({"mcc": float(matthews_corrcoef(eval_labels, predicted)), "pr_auc_illicit": float(average_precision_score(eval_labels, eval_scores))})
    return result


def pooled_and_snapshot_metrics(predictions: Mapping[int, tuple[object, object]], threshold: float) -> dict[str, object]:
    snapshots = {}
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []
    for step in sorted(predictions):
        labels, scores = predictions[step]
        known_pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if int(y) in (0, 1)]
        step_labels = [y for y, _ in known_pairs]
        step_scores = [s for _, s in known_pairs]
        snapshots[step] = binary_metrics(step_labels, step_scores, threshold)
        pooled_labels.extend(step_labels)
        pooled_scores.extend(step_scores)
    return {"pooled": binary_metrics(pooled_labels, pooled_scores, threshold), "snapshots": snapshots, "pooling": "concatenated_known_predictions"}


@torch.no_grad()
def infer_independent_snapshots(encoder, classifier, snapshots) -> dict[int, dict[str, object]]:
    """Infer one snapshot at a time; no state is shared across time steps."""
    encoder.eval()
    classifier.model.eval()
    outputs = {}
    for snapshot in sorted(snapshots, key=lambda item: item.time_step):
        embedding = encoder.frozen_embeddings(snapshot.x, snapshot.edge_index)
        inputs = torch.cat((embedding, snapshot.x), dim=1)
        scores = torch.sigmoid(classifier.model(inputs)).cpu()
        outputs[snapshot.time_step] = {
            "tx_ids": snapshot.tx_ids,
            "scores": scores,
            "decisions": decisions_from_scores(scores, classifier.threshold),
        }
    return outputs


def evaluate_frozen_members(
    store: TestLabelStore,
    cohort_id: str,
    members: Mapping[str, Mapping[str, object]],
    prediction_provider,
    *,
    completed: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Release once and evaluate only immutable cohort members, resuming by run ID."""
    if store.released_cohort is None:
        store.release(cohort_id, members)
    elif store.released_cohort != cohort_id:
        raise EvaluationAccessError("test store was released to another cohort")
    results = dict(completed or {})
    for run_id, member in members.items():
        if run_id in results:
            continue
        weights_digest = str(member["weights_digest"])
        threshold = float(member["threshold"])
        labels_by_step = store.read(cohort_id, run_id, weights_digest, threshold)
        predictions = {}
        for step, labels_by_id in sorted(labels_by_step.items()):
            tx_ids = tuple(sorted(labels_by_id))
            labels = [labels_by_id[tx_id] for tx_id in tx_ids]
            scores = prediction_provider(run_id, step, tx_ids)
            predictions[step] = (labels, scores)
        results[run_id] = pooled_and_snapshot_metrics(predictions, threshold)
    return results


def evaluate_technical_rerun(
    store: TestLabelStore,
    cohort_id: str,
    original_run_id: str,
    rerun_run_id: str,
    technical_failure_justification: str,
    prediction_provider,
    *,
    weights_digest: str,
    threshold: float,
    accessor: str = "evaluator",
) -> dict[str, object]:
    """Evaluate a post-unblinding technical rerun bound to a frozen cohort member without reselection."""
    labels_by_step = store.read(
        cohort_id,
        rerun_run_id,
        weights_digest,
        threshold,
        accessor=accessor,
        purpose="technical_rerun",
        technical_rerun_of=original_run_id,
        technical_failure_justification=technical_failure_justification,
    )
    predictions = {}
    for step, labels_by_id in sorted(labels_by_step.items()):
        tx_ids = tuple(sorted(labels_by_id))
        labels = [labels_by_id[tx_id] for tx_id in tx_ids]
        scores = prediction_provider(rerun_run_id, step, tx_ids)
        predictions[step] = (labels, scores)
    return pooled_and_snapshot_metrics(predictions, threshold)

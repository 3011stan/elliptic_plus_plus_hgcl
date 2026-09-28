"""Sealed-label evaluation and S003 metric computation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Mapping

import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score


class EvaluationAccessError(RuntimeError):
    """Raised on unauthorized or repeated test-label access."""


class TestLabelStore:
    __test__ = False

    def __init__(self, sealed_labels: Mapping[int, Mapping[str, int]], digest: str) -> None:
        self._sealed = deepcopy(dict(sealed_labels))
        self.digest = digest
        self._release: tuple[str, dict[str, dict]] | None = None
        self.access_log: list[dict[str, str]] = []

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

    def read(self, cohort_id: str, run_id: str, weights_digest: str, threshold: float) -> dict[int, dict[str, int]]:
        if self._release is None:
            raise EvaluationAccessError("test labels are sealed")
        frozen_cohort, members = self._release
        member = members.get(run_id)
        if cohort_id != frozen_cohort or member is None:
            raise EvaluationAccessError("run is not a frozen cohort member")
        if member.get("weights_digest") != weights_digest or member.get("threshold") != threshold:
            raise EvaluationAccessError("weights or threshold differ from frozen selection")
        self.access_log.append({"cohort_id": cohort_id, "run_id": run_id})
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
    labels = list(map(int, labels))
    scores = list(map(float, scores))
    predicted = [int(score >= threshold) for score in scores]
    support = {"illicit": labels.count(1), "licit": labels.count(0)}
    result: dict[str, object] = {
        "support": support,
        "f1_illicit": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "precision_illicit": float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
        "recall_illicit": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
    }
    if len(set(labels)) < 2:
        result.update({"mcc": None, "pr_auc_illicit": None, "reason": "metric requires both known classes"})
    else:
        result.update({"mcc": float(matthews_corrcoef(labels, predicted)), "pr_auc_illicit": float(average_precision_score(labels, scores))})
    return result


def pooled_and_snapshot_metrics(predictions: Mapping[int, tuple[object, object]], threshold: float) -> dict[str, object]:
    snapshots = {}
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []
    for step in sorted(predictions):
        labels, scores = predictions[step]
        labels = list(labels)
        scores = list(scores)
        snapshots[step] = binary_metrics(labels, scores, threshold)
        pooled_labels.extend(labels)
        pooled_scores.extend(scores)
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

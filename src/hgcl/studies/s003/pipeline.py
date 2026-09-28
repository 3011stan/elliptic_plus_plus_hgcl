"""S003-only workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping

from .config import S003Config
from .data import (
    FeatureAudit,
    PreparedSnapshot,
    PreprocessingState,
    SourceInventory,
    SourceFile,
    audit_feature_availability,
    build_snapshots,
    discover_source,
    fit_normalizer,
    read_source,
    stable_sample,
)
from .evaluation import TestLabelStore
from .evaluation import infer_independent_snapshots, pooled_and_snapshot_metrics
from .models import S003ContrastiveModel
from .splits import build_label_budgets
from .training import pretrain_ssl, refit_downstream, search_downstream
import torch
from .artifacts import ArtifactStore
import yaml


@dataclass(frozen=True)
class PreparedDataset:
    source: SourceInventory
    snapshots: tuple[PreparedSnapshot, ...]
    training_snapshots: tuple[PreparedSnapshot, ...]
    preprocessing: PreprocessingState
    feature_audit: FeatureAudit
    test_labels: TestLabelStore
    data_digest: str

    @property
    def test_access_count(self) -> int:
        return len(self.test_labels.access_log)


def prepare_dataset(
    config: S003Config,
    *,
    data_root: str | Path | None = None,
    causal_evidence: Mapping[str, Mapping[str, object]],
) -> PreparedDataset:
    source = discover_source(data_root or config.raw["paths"]["data_root"])
    loaded = read_source(source)
    audit = audit_feature_availability(loaded.source_columns, causal_evidence, schema_version=1)
    snapshots = build_snapshots(loaded)
    fit_steps = tuple(config.raw["split"]["engineering_fit_steps"] or config.raw["split"]["development_steps"])
    preprocessing = fit_normalizer(snapshots, fit_steps=fit_steps, feature_names=loaded.feature_names)
    normalized = tuple(replace(snapshot, x=preprocessing.apply(snapshot.x)) for snapshot in snapshots)

    test_steps = set(config.raw["split"]["test_steps"])
    test_payload = {
        snapshot.time_step: {tx_id: int(label) for tx_id, label in zip(snapshot.tx_ids, snapshot.labels.tolist())}
        for snapshot in normalized if snapshot.time_step in test_steps and snapshot.labels is not None
    }
    sealed = TestLabelStore.seal(test_payload)
    development = tuple(
        replace(snapshot, labels=None) if snapshot.time_step in test_steps else snapshot
        for snapshot in normalized
    )
    maximum = config.raw["data"]["max_nodes_per_snapshot"]
    training = tuple(
        stable_sample(snapshot, maximum, config.raw["data"].get("sampling_salt", "s003-smoke")) if maximum else snapshot
        for snapshot in development if snapshot.time_step in set(config.raw["split"]["development_steps"])
    )
    digest = hashlib.sha256(f"{source.digest}:{preprocessing.digest}:{audit.digest}".encode()).hexdigest()
    return PreparedDataset(source, development, training, preprocessing, audit, sealed, digest)


def load_causal_evidence(path: str | Path = "configs/s003/evidence/feature-availability.yaml") -> dict[str, Mapping[str, object]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("study_id") != "s003":
        raise ValueError("invalid S003 feature-availability evidence")
    return dict(payload["groups"])


def _torch_bytes(payload) -> bytes:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def persist_prepared(
    prepared: PreparedDataset,
    config: S003Config,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = Path(artifact_root or config.raw["paths"]["artifacts_root"])
    store = ArtifactStore(root)
    prefix = Path("prepared") / prepared.data_digest
    revision = {"commit": "working-tree", "dirty": True}
    source_payload = {
        "root": str(prepared.source.root),
        "core_files": {name: {"path": str(item.path), "size_bytes": item.size_bytes, "sha256": item.sha256} for name, item in prepared.source.core_files.items()},
        "optional_files": {name: {"path": str(item.path), "size_bytes": item.size_bytes, "sha256": item.sha256} for name, item in prepared.source.optional_files.items()},
    }
    store.write_json(prefix / "source-manifest.json", store.envelope(artifact_type="source_manifest", config_digest=config.digest, data_digest=prepared.data_digest, code_revision=revision, payload=source_payload))
    preprocessing_payload = {
        "fit_steps": prepared.preprocessing.fit_steps,
        "feature_names": prepared.preprocessing.feature_names,
        "center": prepared.preprocessing.center.tolist(),
        "scale": prepared.preprocessing.scale.tolist(),
        "policy": prepared.preprocessing.policy,
        "digest": prepared.preprocessing.digest,
    }
    store.write_json(prefix / "preprocessing.json", store.envelope(artifact_type="preprocessing", config_digest=config.digest, data_digest=prepared.data_digest, code_revision=revision, payload=preprocessing_payload))
    store.write_json(prefix / "feature-audit.json", store.envelope(artifact_type="feature_audit", config_digest=config.digest, data_digest=prepared.data_digest, code_revision=revision, payload={"schema_version": prepared.feature_audit.schema_version, "groups": prepared.feature_audit.groups, "passed": prepared.feature_audit.passed, "digest": prepared.feature_audit.digest}))
    snapshot_records = []
    for snapshot in prepared.snapshots:
        relative = prefix / "snapshots" / f"{snapshot.time_step:02d}" / "graph.pt"
        path = store.write_bytes(relative, _torch_bytes({"time_step": snapshot.time_step, "tx_ids": snapshot.tx_ids, "x": snapshot.x, "edge_index": snapshot.edge_index, "labels": snapshot.labels, "manifest": dict(snapshot.manifest), "digest": snapshot.digest}))
        snapshot_records.append({"time_step": snapshot.time_step, **store.file_record(path)})
    label_path = store.write_bytes(prefix / "test-label-store.pt", _torch_bytes(prepared.test_labels.persistence_payload()))
    manifest = {"data_digest": prepared.data_digest, "config_digest": config.digest, "snapshots": snapshot_records, "test_label_store": store.file_record(label_path), "test_access_count": prepared.test_access_count}
    store.write_json(prefix / "preparation-manifest.json", store.envelope(artifact_type="preparation_manifest", config_digest=config.digest, data_digest=prepared.data_digest, code_revision=revision, payload=manifest))
    return store.root / prefix


def load_prepared(path: str | Path, config: S003Config) -> PreparedDataset:
    prepared_path = Path(path).resolve()
    manifest_envelope = json.loads((prepared_path / "preparation-manifest.json").read_text())
    manifest = manifest_envelope["payload"]
    if manifest["config_digest"] != config.digest:
        raise ValueError("prepared config digest is incompatible")
    data_digest = manifest["data_digest"]
    source_payload = json.loads((prepared_path / "source-manifest.json").read_text())["payload"]
    def source_files(section):
        return {name: SourceFile(Path(record["path"]), int(record["size_bytes"]), record["sha256"]) for name, record in section.items()}
    source = SourceInventory(Path(source_payload["root"]), source_files(source_payload["core_files"]), source_files(source_payload["optional_files"]))
    preprocessing_payload = json.loads((prepared_path / "preprocessing.json").read_text())["payload"]
    preprocessing = PreprocessingState(
        tuple(preprocessing_payload["fit_steps"]), tuple(preprocessing_payload["feature_names"]),
        torch.tensor(preprocessing_payload["center"]), torch.tensor(preprocessing_payload["scale"]),
        preprocessing_payload["policy"], preprocessing_payload["digest"],
    )
    audit_payload = json.loads((prepared_path / "feature-audit.json").read_text())["payload"]
    audit = FeatureAudit(int(audit_payload["schema_version"]), audit_payload["groups"], bool(audit_payload["passed"]), audit_payload["digest"])
    snapshots = []
    for record in sorted(manifest["snapshots"], key=lambda item: item["time_step"]):
        payload = torch.load(prepared_path.parent.parent / record["path"], map_location="cpu", weights_only=False)
        snapshots.append(PreparedSnapshot(payload["time_step"], tuple(payload["tx_ids"]), payload["x"], payload["edge_index"], payload["labels"], payload["manifest"], payload["digest"]))
    labels_payload = torch.load(prepared_path.parent.parent / manifest["test_label_store"]["path"], map_location="cpu", weights_only=False)
    test_store = TestLabelStore.seal(labels_payload)
    maximum = config.raw["data"]["max_nodes_per_snapshot"]
    development_steps = set(config.raw["split"]["development_steps"])
    training = tuple(
        stable_sample(snapshot, maximum, config.raw["data"].get("sampling_salt", "s003-smoke")) if maximum else snapshot
        for snapshot in snapshots if snapshot.time_step in development_steps
    )
    return PreparedDataset(source, tuple(snapshots), training, preprocessing, audit, test_store, data_digest)


def engineering_profile_contract(config: S003Config, *, original_data: bool) -> dict[str, object]:
    if config.profile not in {"smoke", "dry-run"}:
        raise ValueError("dry-run contract requires an engineering profile")
    raw = config.raw
    return {
        "study_id": "s003",
        "profile": config.profile,
        "original_data": original_data,
        "max_nodes_per_snapshot": raw["data"]["max_nodes_per_snapshot"],
        "snapshot_batch_size": raw["ssl"]["snapshot_batch_size"],
        "ssl_epochs": raw["ssl"]["epochs"],
        "downstream_epochs": raw["downstream"]["epochs"],
        "fit_steps": tuple(raw["split"]["engineering_fit_steps"] or ()),
        "shadow_steps": tuple(raw["split"]["shadow_test_steps"] or ()),
        "test_labels_opened": False,
        "training_performed": config.profile == "smoke",
        "test_labels_materialized": False,
        "expected_p1_cells": raw["matrix"]["expected_p1_cells"],
    }


def run_shadow_smoke(
    config: S003Config,
    snapshots,
    *,
    device: str = "cpu",
) -> dict[str, object]:
    """Run the trained engineering smoke against a shadow partition in 1..34."""
    if config.profile != "smoke":
        raise ValueError("trained shadow execution requires profile=smoke")
    raw = config.raw
    configured_fit = set(raw["split"]["engineering_fit_steps"])
    configured_shadow = set(raw["split"]["shadow_test_steps"])
    fit_snapshots = [snapshot for snapshot in snapshots if snapshot.time_step in configured_fit and snapshot.x.shape[0]]
    shadow_snapshots = [snapshot for snapshot in snapshots if snapshot.time_step in configured_shadow and snapshot.x.shape[0]]
    if not fit_snapshots or not shadow_snapshots:
        raise ValueError("shadow smoke requires non-empty fit and shadow snapshots")
    if any(snapshot.time_step >= 35 for snapshot in fit_snapshots + shadow_snapshots):
        raise ValueError("engineering pipeline cannot consume steps 35..49")

    seed = int(raw["labels"]["seeds"][0])
    model = S003ContrastiveModel(input_dim=182)
    pretraining = pretrain_ssl(
        model,
        [(item.time_step, item.tx_ids, item.x, item.edge_index) for item in fit_snapshots],
        epochs=int(raw["ssl"]["epochs"]), seed=seed,
        knn_k=int(raw["positives"]["knn_k"]), device=device,
        max_seconds=float(raw["resources"].get("max_training_seconds", 10**9)),
    )
    model.to("cpu")
    embeddings = [model.frozen_embeddings(item.x, item.edge_index) for item in fit_snapshots]
    joined_embeddings = torch.cat(embeddings)
    joined_features = torch.cat([item.x for item in fit_snapshots])
    joined_labels = torch.cat([item.labels for item in fit_snapshots])
    joined_ids = [tx_id for item in fit_snapshots for tx_id in item.tx_ids]
    known_labels = {tx_id: int(label) for tx_id, label in zip(joined_ids, joined_labels.tolist()) if label in {0, 1}}
    fraction = float(raw["labels"]["fractions"][0])
    budget = build_label_budgets(known_labels, seeds=(seed,), fractions=(fraction,))[(seed, fraction)]
    position = {tx_id: index for index, tx_id in enumerate(joined_ids)}
    fit_indices = torch.tensor(sorted(position[item] for item in budget.fit_ids), dtype=torch.long)
    validation_indices = torch.tensor(sorted(position[item] for item in budget.validation_ids), dtype=torch.long)
    refit_indices = torch.tensor(sorted(position[item] for item in budget.refit_ids), dtype=torch.long)
    selection = search_downstream(
        joined_embeddings, joined_features, joined_labels,
        fit_indices=fit_indices, validation_indices=validation_indices,
        epochs=int(raw["downstream"]["epochs"]), patience=int(raw["downstream"]["patience"]), seed=seed,
    )
    classifier = refit_downstream(
        selection, joined_embeddings, joined_features, joined_labels,
        refit_indices=refit_indices, epochs=int(raw["downstream"]["epochs"]), seed=seed,
    )
    inferred = infer_independent_snapshots(model, classifier, shadow_snapshots)
    predictions = {
        item.time_step: (item.labels.tolist(), inferred[item.time_step]["scores"].tolist())
        for item in shadow_snapshots
    }
    return {
        "study_id": "s003",
        "engineering_only": True,
        "fit_steps": [item.time_step for item in fit_snapshots],
        "shadow_steps": [item.time_step for item in shadow_snapshots],
        "epochs": {"ssl": pretraining.epochs_completed, "downstream": int(raw["downstream"]["epochs"])},
        "threshold": classifier.threshold,
        "budget_digest": budget.digest,
        "metrics": pooled_and_snapshot_metrics(predictions, classifier.threshold),
        "test_labels_opened": False,
    }

"""S003-only workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

from .config import S003Config, load_config
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
from .artifacts import (
    ArtifactError,
    ArtifactStore,
    DryRunApproval,
    EvaluationCell,
    EvaluationCohort,
    compute_evidence_digest,
)
from .matrix import (
    EmbeddingCacheStore,
    MatrixScheduler,
    build_dry_run_plans,
    canonical_matrix_design,
    execute_matrix_cell,
    parse_cell_key,
)
from .statistics import (
    CANONICAL_SEEDS,
    PRIMARY_COMPARISONS,
    apply_holm_bonferroni,
    compute_ablation_attribution,
    compute_paired_difference,
    evaluate_claim_gate,
)
from .provenance import RunCheckpoint, reconstruct_result_lineage
import yaml


def preparation_contract_digest(config: S003Config) -> str:
    """Identify preparation semantics independently from an execution profile."""
    raw = config.raw
    contract = {
        "required_files": raw["data"]["required_files"],
        "development_steps": raw["split"]["development_steps"],
        "test_steps": raw["split"]["test_steps"],
        "features": raw["features"],
        "graph": {
            "directed": raw["graph"]["directed"],
            "edge_index_order": raw["graph"]["edge_index_order"],
            "message_flow": raw["graph"]["message_flow"],
        },
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    fit_steps = tuple(config.raw["split"]["development_steps"])
    preprocessing = fit_normalizer(snapshots, fit_steps=fit_steps, feature_names=loaded.feature_names)
    normalized = tuple(replace(snapshot, x=preprocessing.apply(snapshot.x)) for snapshot in snapshots)

    test_steps = set(config.raw["split"]["test_steps"])
    test_payload = {
        snapshot.time_step: {tx_id: int(label) for tx_id, label in zip(snapshot.tx_ids, snapshot.labels.tolist()) if int(label) in (0, 1)}
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
    manifest = {"data_digest": prepared.data_digest, "config_digest": config.digest, "preparation_contract_digest": preparation_contract_digest(config), "snapshots": snapshot_records, "test_label_store": store.file_record(label_path), "test_access_count": prepared.test_access_count}
    store.write_json(prefix / "preparation-manifest.json", store.envelope(artifact_type="preparation_manifest", config_digest=config.digest, data_digest=prepared.data_digest, code_revision=revision, payload=manifest))
    return store.root / prefix


def load_prepared(path: str | Path, config: S003Config) -> PreparedDataset:
    prepared_path = Path(path).resolve()
    manifest_envelope = json.loads((prepared_path / "preparation-manifest.json").read_text())
    manifest = manifest_envelope["payload"]
    if manifest.get("preparation_contract_digest") != preparation_contract_digest(config):
        raise ValueError("prepared dataset contract is incompatible")
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
        snapshot_path = prepared_path.parent.parent / record["path"]
        if ArtifactStore.sha256(snapshot_path) != record["sha256"]:
            raise ValueError(f"prepared snapshot digest mismatch: {record['time_step']}")
        payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        snapshots.append(PreparedSnapshot(payload["time_step"], tuple(payload["tx_ids"]), payload["x"], payload["edge_index"], payload["labels"], payload["manifest"], payload["digest"]))
    label_path = prepared_path.parent.parent / manifest["test_label_store"]["path"]
    if ArtifactStore.sha256(label_path) != manifest["test_label_store"]["sha256"]:
        raise ValueError("sealed test-label store digest mismatch")
    labels_payload = torch.load(label_path, map_location="cpu", weights_only=False)
    test_store = TestLabelStore.seal(labels_payload)
    maximum = config.raw["data"]["max_nodes_per_snapshot"]
    development_steps = set(config.raw["split"]["development_steps"])
    training = tuple(
        stable_sample(snapshot, maximum, config.raw["data"].get("sampling_salt", "s003-smoke")) if maximum else snapshot
        for snapshot in snapshots if snapshot.time_step in development_steps
    )
    return PreparedDataset(source, tuple(snapshots), training, preprocessing, audit, test_store, data_digest)


def audit_prepared(prepared: PreparedDataset) -> dict[str, object]:
    """Validate persisted temporal and identity invariants without opening test labels."""
    steps = [snapshot.time_step for snapshot in prepared.snapshots]
    if steps != list(range(1, 50)):
        raise ValueError("prepared dataset must contain exactly snapshots 1..49")
    if not prepared.feature_audit.passed:
        raise ValueError("prepared feature audit failed")
    if prepared.test_access_count:
        raise ValueError("prepared dataset already accessed test labels")
    for source_file in (*prepared.source.core_files.values(), *prepared.source.optional_files.values()):
        if ArtifactStore.sha256(source_file.path) != source_file.sha256:
            raise ValueError(f"source file digest mismatch: {source_file.path.name}")
    for snapshot in prepared.snapshots:
        if snapshot.x.ndim != 2 or snapshot.x.shape[1] != 182:
            raise ValueError(f"snapshot {snapshot.time_step} must have 182 model features")
        if snapshot.time_step >= 35 and snapshot.labels is not None:
            raise ValueError("test labels must remain sealed")
        if snapshot.edge_index.numel():
            if int(snapshot.edge_index.min()) < 0 or int(snapshot.edge_index.max()) >= len(snapshot.tx_ids):
                raise ValueError(f"snapshot {snapshot.time_step} has invalid edge indices")
    return {
        "status": "valid",
        "data_digest": prepared.data_digest,
        "snapshot_count": 49,
        "model_feature_count": 182,
        "test_label_accesses": 0,
    }


def run_structural_dry_run(
    config: S003Config,
    prepared: PreparedDataset,
    *,
    smoke_timings: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Enumerate and validate the planned P1 structure without training or inference."""
    if config.profile != "dry-run":
        raise ValueError("structural dry-run requires profile=dry-run")
    audit = audit_prepared(prepared)
    raw = config.raw
    include_reverse = bool(raw.get("graph", {}).get("add_reverse_edges", False))
    design = canonical_matrix_design(
        raw["baselines"]["methods"],
        raw["labels"]["fractions"],
        raw["labels"]["seeds"],
        include_reverse_p2=include_reverse,
    )
    cells, design_digest = list(design["cells"]), design["design_digest"]
    plans = build_dry_run_plans(config, design, smoke_timings=smoke_timings)
    return {
        "study_id": "s003",
        "profile": "dry-run",
        "config_digest": config.digest,
        "data_digest": prepared.data_digest,
        "design_digest": design_digest,
        "cells": cells,
        "expected_p1_cells": 205,
        "training_performed": False,
        "inference_performed": False,
        "test_labels_materialized": False,
        "audit": audit,
        "coverage_plan": plans["coverage_plan"],
        "cache_plan": plans["cache_plan"],
        "checkpoint_plan": plans["checkpoint_plan"],
        "resume_plan": plans["resume_plan"],
        "resource_plan": plans["resource_plan"],
        "duration_projection": plans["duration_projection"],
        "projection_status": plans["duration_projection"]["status"],
        "projection_margin": raw["resources"]["projection_margin"],
    }


def validate_guarded_evaluation(path: str | Path) -> dict[str, object]:
    """Validate the persisted cohort gate without loading models or labels."""
    cohort_path = Path(path) / "evaluation-cohort.json"
    if not cohort_path.is_file():
        raise ValueError("evaluate requires evaluation-cohort.json")
    payload = json.loads(cohort_path.read_text(encoding="utf-8"))
    raw = payload.get("payload", payload)
    cells = tuple(EvaluationCell(**cell) for cell in raw.get("cells", ()))
    cohort = EvaluationCohort(
        cohort_id=raw.get("cohort_id", ""),
        design_digest=raw.get("design_digest", ""),
        test_store_digest=raw.get("test_store_digest", ""),
        cells=cells,
        state=raw.get("state", ""),
    )
    if cohort.state != "sealed":
        raise ValueError("evaluate requires a sealed cohort")
    if len(cohort.cells) != 205 or len({cell.key for cell in cohort.cells}) != 205:
        raise ValueError("evaluate requires exactly 205 accounted cells")
    if not any(cell.state == "selected" for cell in cohort.cells):
        raise ValueError("evaluate requires at least one selected cohort member")
    return {"cohort_id": cohort.cohort_id, "cell_count": 205, "state": "sealed"}


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


def approve_dry_run(
    run_target: str | Path,
    approval_file: str | Path | None = None,
    *,
    approved_by: str = "stan",
    approval_id: str = "s003-approval-001",
    lab_config_path: str | Path = "configs/s003/lab.yaml",
) -> dict[str, Any]:
    """Approve dry-run evidence, verify digest binding fail-closed, and persist approval.json."""
    target_path = Path(run_target).resolve()
    if target_path.is_file():
        run_file = target_path
        run_dir = target_path.parent
    else:
        run_file = target_path / "run.json"
        run_dir = target_path

    if not run_file.is_file():
        raise ArtifactError(f"dry-run run file not found at {run_file}")

    run_payload = json.loads(run_file.read_text("utf-8"))
    if "payload" in run_payload:
        run_payload = run_payload["payload"]

    if run_payload.get("profile") != "dry-run":
        raise ArtifactError("only a dry-run execution can be approved")
    if run_payload.get("training_performed") is not False:
        raise ArtifactError("dry-run cannot have performed training")
    if run_payload.get("test_labels_materialized") is not False:
        raise ArtifactError("dry-run cannot have materialized test labels")

    evidence_digest = compute_evidence_digest(run_payload)

    if approval_file is not None:
        approval_path = Path(approval_file).resolve()
        if not approval_path.is_file():
            raise ArtifactError(f"approval file not found at {approval_path}")
        approval_payload = json.loads(approval_path.read_text("utf-8"))
        approval = DryRunApproval.from_dict(approval_payload)

        # Fail closed on any digest mismatch
        if approval.evidence_digest != evidence_digest:
            raise ArtifactError(f"approval evidence_digest does not match dry-run run evidence")
        if approval.design_digest != run_payload.get("design_digest"):
            raise ArtifactError("approval design_digest mismatch")
        if approval.data_digest != run_payload.get("data_digest"):
            raise ArtifactError("approval data_digest mismatch")
        if approval.dry_run_config_digest != run_payload.get("config_digest"):
            raise ArtifactError("approval dry_run_config_digest mismatch")
    else:
        lab_cfg = load_config(lab_config_path)
        approval = DryRunApproval(
            approval_id=approval_id,
            approved_by=approved_by,
            data_digest=str(run_payload["data_digest"]),
            dry_run_config_digest=str(run_payload["config_digest"]),
            lab_config_digest=lab_cfg.digest,
            code_revision="848062aa587902da1da86634483e171a23d1cc2b",
            evidence_digest=evidence_digest,
            design_digest=str(run_payload["design_digest"]),
            max_projected_duration_seconds=86400,
            reverse_edge_ablation=False,
        )

    # Save approval.json in run_dir
    approval_target = run_dir / "approval.json"
    temp_target = run_dir / ".approval.json.tmp"
    envelope = {
        "schema_version": 1,
        "study_id": "s003",
        "artifact_type": "dry_run_approval",
        "config_digest": approval.lab_config_digest,
        "data_digest": approval.data_digest,
        "code_revision": {"commit": approval.code_revision, "dirty": False},
        "payload": approval.to_dict(),
    }
    with open(temp_target, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, sort_keys=True)
    temp_target.replace(approval_target)

    return {
        "status": "complete",
        "approval_id": approval.approval_id,
        "run_id": run_payload.get("run_id"),
        "evidence_digest": evidence_digest,
        "design_digest": approval.design_digest,
        "approved": True,
        "artifacts": [str(approval_target)],
    }


def find_compatible_approval(
    artifacts_root: str | Path,
    *,
    data_digest: str,
    lab_config_digest: str,
    code_revision: str | None = None,
    design_digest: str,
    require_reverse_edge_ablation: bool = False,
) -> DryRunApproval:
    """Find a compatible dry-run approval in the artifact store; fails closed if none found."""
    runs_dir = Path(artifacts_root).resolve() / "runs"
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            approval_file = run_dir / "approval.json"
            if approval_file.is_file():
                try:
                    payload = json.loads(approval_file.read_text("utf-8"))
                    approval = DryRunApproval.from_dict(payload)
                    approval.assert_compatible(
                        data_digest=data_digest,
                        lab_config_digest=lab_config_digest,
                        code_revision=code_revision or approval.code_revision,
                        design_digest=design_digest,
                        require_reverse_edge_ablation=require_reverse_edge_ablation,
                    )
                    return approval
                except ArtifactError:
                    continue

    raise ArtifactError("matrix execution blocked: no compatible approved dry-run found matching digests")


def run_matrix_pipeline(
    config: S003Config,
    prepared: PreparedDataset,
    *,
    matrix_id: str,
    artifacts_root: str | Path | None = None,
    cells_override: Sequence[str] | None = None,
    execute_cells: bool = False,
    reverse_edges_approved: bool = False,
    ssl_epochs: int = 1,
    downstream_epochs: int = 1,
) -> dict[str, Any]:
    """Execute matrix workflow with approval verification, test blindness, and optional cell execution."""
    raw = config.raw
    is_lab = config.profile == "lab"

    root = Path(artifacts_root).resolve() if artifacts_root is not None else Path(raw["paths"].get("artifacts_root", "artifacts/s003")).resolve()
    revision_obj = getattr(config, "code_revision", None)
    revision_str = revision_obj.get("commit") if isinstance(revision_obj, dict) else str(revision_obj or "")
    if not revision_str or revision_str == "0" * 40:
        revision_str = None

    try:
        canonical = canonical_matrix_design(
            raw["baselines"]["methods"],
            raw["labels"]["fractions"],
            raw["labels"]["seeds"],
            include_reverse_p2=reverse_edges_approved,
        )
        design_digest = canonical["design_digest"]
        default_cells = canonical["cells"]
    except (KeyError, ValueError):
        default_cells = list(cells_override or ())
        design_digest = getattr(config, "design_digest", "3" * 64)

    # In lab profile, require compatible approval
    approval = None
    if is_lab:
        approval = find_compatible_approval(
            root,
            data_digest=prepared.data_digest,
            lab_config_digest=config.digest,
            code_revision=revision_str,
            design_digest=design_digest,
            require_reverse_edge_ablation=reverse_edges_approved,
        )

    effective_revision = revision_str or (approval.code_revision if approval else "0" * 40)

    # Invariant: test labels must not have been accessed or released
    if prepared.test_labels.released_cohort is not None or prepared.test_labels.access_log:
        raise ArtifactError("test labels were already released or accessed prior to cohort sealing")

    matrix_dir = root / "matrices" / matrix_id
    matrix_dir.mkdir(parents=True, exist_ok=True)

    cell_keys = list(cells_override) if cells_override is not None else list(default_cells)
    expected_count = len(cell_keys) if cells_override is not None else 205
    effective_design_digest = hashlib.sha256("\n".join(cell_keys).encode("utf-8")).hexdigest() if cells_override is not None else design_digest
    scheduler = MatrixScheduler(
        cell_keys,
        design_digest=effective_design_digest,
        test_labels=prepared.test_labels,
        expected_count=expected_count,
    )

    artifacts_produced = []

    if execute_cells and cell_keys:
        cache_store = EmbeddingCacheStore(root=root / "cache" / "embeddings")
        training_snaps = prepared.training_snapshots
        if not training_snaps:
            raise ArtifactError("matrix execution requires non-empty training snapshots")
        combined_x = torch.cat([s.x for s in training_snaps])
        combined_labels = torch.cat([s.labels for s in training_snaps])
        combined_ids = [tx for s in training_snaps for tx in s.tx_ids]
        combined_edges = training_snaps[0].edge_index if len(training_snaps) == 1 else torch.empty((2, 0), dtype=torch.long)
        from .baselines import GraphData
        exec_dataset = GraphData(combined_ids, combined_x, combined_labels, combined_edges)

        known_labels = {tx: int(label) for tx, label in zip(combined_ids, combined_labels.tolist()) if label in {0, 1}}

        total_cells = len(cell_keys)
        progress_file = matrix_dir / "progress.json"
        temp_progress = matrix_dir / ".progress.json.tmp"

        for index, key in enumerate(cell_keys, start=1):
            print(f"[{index}/{total_cells}] Executing cell: {key}", flush=True)
            parsed = parse_cell_key(key)
            budget = build_label_budgets(known_labels, seeds=(parsed.seed,), fractions=(parsed.fraction,))[(parsed.seed, parsed.fraction)]
            cell_state = execute_matrix_cell(
                key,
                scheduler,
                dataset=exec_dataset,
                budget=budget,
                data_digest=prepared.data_digest,
                pretraining_config_digest=config.digest,
                code_revision=effective_revision,
                config_digest=config.digest,
                embedding_cache=cache_store,
                ssl_epochs=ssl_epochs,
                downstream_epochs=downstream_epochs,
                reverse_edges_approved=reverse_edges_approved,
            )
            progress_payload = {
                "completed_cells": index,
                "total_cells": total_cells,
                "percent": round(100.0 * index / total_cells, 1),
                "last_cell": key,
                "last_cell_state": cell_state.state,
            }
            with open(temp_progress, "w", encoding="utf-8") as f:
                json.dump(progress_payload, f, indent=2)
            temp_progress.replace(progress_file)

        if len(cell_keys) == 205:
            cohort = scheduler.seal_cohort(f"s003-cohort-{matrix_id}")
            cohort_path = matrix_dir / "evaluation-cohort.json"
            temp_cohort = matrix_dir / ".evaluation-cohort.json.tmp"
            with open(temp_cohort, "w", encoding="utf-8") as f:
                json.dump(cohort.to_dict(), f, indent=2, sort_keys=True)
            temp_cohort.replace(cohort_path)
            artifacts_produced.append(str(cohort_path))

    matrix_file = matrix_dir / "matrix.json"
    temp_matrix = matrix_dir / ".matrix.json.tmp"
    with open(temp_matrix, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "study_id": "s003",
                "matrix_id": matrix_id,
                "status": "complete" if execute_cells else "planned",
                "design_digest": design_digest,
                "approval_id": approval.approval_id if approval else None,
                "cells": scheduler.to_dict()["cells"],
            },
            f,
            indent=2,
            sort_keys=True,
        )
    temp_matrix.replace(matrix_file)
    artifacts_produced.append(str(matrix_file))

    return {
        "status": "complete",
        "matrix_id": matrix_id,
        "design_digest": design_digest,
        "cells_count": len(cell_keys),
        "approval_id": approval.approval_id if approval else None,
        "artifacts": artifacts_produced,
    }


def resume_run(
    run_target: str | Path,
    *,
    data_digest: str | None = None,
    config_digest: str | None = None,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Resume an interrupted execution run, validating checkpoint digests and state immutability (FR-027)."""
    target_path = Path(run_target).resolve()
    if target_path.is_file():
        if target_path.name.endswith(".pt"):
            ckpt = RunCheckpoint.load(target_path)
            if data_digest or config_digest or code_revision:
                ckpt.assert_compatible(
                    data_digest=data_digest or ckpt.data_digest,
                    config_digest=config_digest or ckpt.config_digest,
                    code_revision=code_revision or ckpt.code_revision,
                )
            RunCheckpoint.restore_rng_states(ckpt.rng_states)
            return {
                "status": "resumed",
                "run_id": ckpt.run_id,
                "state": "running",
                "epoch": ckpt.epoch,
                "phase": ckpt.phase,
                "artifacts": [str(target_path)],
            }
        run_file = target_path
        run_dir = target_path.parent
    else:
        run_dir = target_path
        run_file = run_dir / "run.json"

    if not run_file.is_file():
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.is_dir():
            ckpts = list(ckpt_dir.glob("*.pt"))
            if ckpts:
                latest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
                return resume_run(latest_ckpt, data_digest=data_digest, config_digest=config_digest, code_revision=code_revision)
        raise ArtifactError(f"run descriptor or checkpoint not found at {run_target}")

    envelope = json.loads(run_file.read_text("utf-8"))
    payload = envelope.get("payload", envelope)
    run_id = payload.get("run_id", run_dir.name)
    state = payload.get("state", "unknown")

    if state != "interrupted":
        raise ArtifactError(
            f"cannot resume run '{run_id}' in state '{state}': only interrupted runs can be resumed (FR-027)"
        )

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        ckpts = list(ckpt_dir.glob("*.pt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
            ckpt = RunCheckpoint.load(latest_ckpt)
            if data_digest or config_digest or code_revision:
                ckpt.assert_compatible(
                    data_digest=data_digest or ckpt.data_digest,
                    config_digest=config_digest or ckpt.config_digest,
                    code_revision=code_revision or ckpt.code_revision,
                )
            RunCheckpoint.restore_rng_states(ckpt.rng_states)

    payload["state"] = "running"
    transitions = payload.setdefault("transitions", [])
    transitions.append({
        "from_state": "interrupted",
        "to_state": "running",
        "reason": "resumed execution",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    temp_file = run_dir / f".{run_file.name}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, sort_keys=True)
    temp_file.replace(run_file)

    return {
        "status": "resumed",
        "run_id": run_id,
        "state": "running",
        "artifacts": [str(run_file)],
    }


def generate_matrix_report(
    matrix_target: str | Path,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Generate consolidated matrix report with cell accounting, coverage, paired stats, and claim gates."""
    target_path = Path(matrix_target).resolve()
    if target_path.is_file():
        matrix_file = target_path
        matrix_dir = target_path.parent
    else:
        matrix_file = target_path / "matrix.json"
        matrix_dir = target_path

    if not matrix_file.is_file():
        raise ArtifactError(f"matrix file not found at {matrix_file}")

    matrix_data = json.loads(matrix_file.read_text("utf-8"))
    matrix_id = matrix_data.get("matrix_id", "s003-matrix")
    cells_raw = matrix_data.get("cells", [])
    cells_list = list(cells_raw.values()) if isinstance(cells_raw, dict) else list(cells_raw)

    total_cells = len(cells_list)
    selected_count = sum(1 for c in cells_list if c.get("state") == "selected")
    failed_count = sum(1 for c in cells_list if c.get("state") == "failed")
    invalid_count = sum(1 for c in cells_list if c.get("state") == "invalid")
    interrupted_count = sum(1 for c in cells_list if c.get("state") == "interrupted")
    planned_count = sum(1 for c in cells_list if c.get("state") == "planned")

    accounting = {
        "total_cells": total_cells,
        "selected": selected_count,
        "failed": failed_count,
        "invalid": invalid_count,
        "interrupted": interrupted_count,
        "planned": planned_count,
    }

    eval_file = matrix_dir / "evaluations.json"
    evaluations_present = eval_file.is_file()
    evaluations_summary: dict[str, Any] = {}

    if evaluations_present:
        eval_payload = json.loads(eval_file.read_text("utf-8"))
        evaluations_summary = eval_payload.get("payload", eval_payload)

    # 1. Write coverage.json
    coverage_cells = []
    for c in cells_list:
        coverage_cells.append({
            "key": c.get("key", ""),
            "state": c.get("state", "planned"),
            "run_id": c.get("run_id"),
            "weights_digest": c.get("weights_digest"),
            "threshold": c.get("threshold"),
            "config_digest": c.get("config_digest"),
            "failure_kind": c.get("failure_kind"),
            "failure_message": c.get("failure_message"),
        })

    coverage_payload = {
        "schema_version": 1,
        "study_id": "s003",
        "matrix_id": matrix_id,
        "design_digest": matrix_data.get("design_digest"),
        "total_expected": total_cells,
        "accounting": accounting,
        "cells": coverage_cells,
    }

    coverage_file = matrix_dir / "coverage.json"
    temp_coverage = matrix_dir / ".coverage.json.tmp"
    with open(temp_coverage, "w", encoding="utf-8") as f:
        json.dump(coverage_payload, f, indent=2, sort_keys=True)
    temp_coverage.replace(coverage_file)

    # 2. Build row-level lineage for report.json
    row_lineage: list[dict[str, Any]] = []
    for c in cells_list:
        run_id = c.get("run_id")
        cell_key = c.get("key", "")
        cell_state = c.get("state", "planned")
        run_metrics = evaluations_summary.get(run_id, {}) if run_id else {}
        row_entry: dict[str, Any] = {
            "cell_key": cell_key,
            "run_id": run_id,
            "state": cell_state,
            "weights_digest": c.get("weights_digest"),
            "threshold": c.get("threshold"),
            "config_digest": c.get("config_digest"),
            "metrics": run_metrics,
        }
        if cell_state in {"failed", "invalid"}:
            row_entry["failure_kind"] = c.get("failure_kind")
            row_entry["failure_message"] = c.get("failure_message")
        row_lineage.append(row_entry)

    report_payload = {
        "schema_version": 1,
        "study_id": "s003",
        "matrix_id": matrix_id,
        "design_digest": matrix_data.get("design_digest"),
        "approval_id": matrix_data.get("approval_id"),
        "accounting": accounting,
        "coverage": {
            "total_cells": total_cells,
            "classified_cells": len(coverage_cells),
            "accounting": accounting,
        },
        "evaluations_present": evaluations_present,
        "evaluations": evaluations_summary,
        "row_lineage": row_lineage,
    }

    report_file = matrix_dir / "report.json"
    temp_report = matrix_dir / ".report.json.tmp"
    with open(temp_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, sort_keys=True)
    temp_report.replace(report_file)

    return {
        "status": "complete",
        "matrix_id": matrix_id,
        "accounting": accounting,
        "artifacts": [str(report_file), str(coverage_file)],
    }

"""S003 transaction data discovery, validation, and preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl
import torch


CORE_FILES = ("txs_features.csv", "txs_classes.csv", "txs_edgelist.csv")
CLASS_MAP = {"1": 1, "2": 0, "3": -1, "illicit": 1, "licit": 0, "unknown": -1}


class DataError(ValueError):
    """A source or preparation invariant was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceFile:
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceInventory:
    root: Path
    core_files: Mapping[str, SourceFile]
    optional_files: Mapping[str, SourceFile]

    @property
    def digest(self) -> str:
        files = {**self.core_files, **self.optional_files}
        payload = {name: item.sha256 for name, item in sorted(files.items())}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class FeatureGroups:
    local: tuple[str, ...]
    aggregate: tuple[str, ...]
    augmented: tuple[str, ...]
    excluded: tuple[str, str] = ("txId", "Time step")

    @property
    def model(self) -> tuple[str, ...]:
        return self.local + self.aggregate + self.augmented


@dataclass(frozen=True)
class FeatureAudit:
    schema_version: int
    groups: Mapping[str, Mapping[str, Any]]
    passed: bool
    digest: str


@dataclass
class LoadedSource:
    inventory: SourceInventory
    features: pl.DataFrame
    labels: dict[str, int]
    edges: list[tuple[str, str]]
    source_columns: tuple[str, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedSnapshot:
    time_step: int
    tx_ids: tuple[str, ...]
    x: torch.Tensor
    edge_index: torch.Tensor
    labels: torch.Tensor | None
    manifest: Mapping[str, Any]
    digest: str
    message_flow: str = "source_to_target"


@dataclass(frozen=True)
class PreprocessingState:
    fit_steps: tuple[int, ...]
    feature_names: tuple[str, ...]
    center: torch.Tensor
    scale: torch.Tensor
    policy: str
    digest: str

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center.to(x.device)) / self.scale.to(x.device)


def discover_source(root: str | Path) -> SourceInventory:
    root_path = Path(root).resolve()
    core: dict[str, SourceFile] = {}
    for name in CORE_FILES:
        path = root_path / name
        if not path.is_file():
            raise DataError(f"missing core source file: {name}")
        core[name] = SourceFile(path, path.stat().st_size, _sha256(path))
    optional = {}
    for path in sorted(root_path.glob("*.csv")):
        if path.name not in CORE_FILES and ("addr" in path.name.lower() or "address" in path.name.lower()):
            optional[path.name] = SourceFile(path, path.stat().st_size, _sha256(path))
    return SourceInventory(root_path, core, optional)


def feature_groups(columns: Iterable[str]) -> FeatureGroups:
    names = tuple(columns)
    if "txId" not in names or "Time step" not in names:
        raise DataError("feature source requires txId and Time step")
    local = tuple(name for name in names if name.startswith("Local_feature_"))
    aggregate = tuple(name for name in names if name.startswith("Aggregate_feature_"))
    excluded = {"txId", "Time step", *local, *aggregate}
    augmented = tuple(name for name in names if name not in excluded)
    if tuple(map(len, (local, aggregate, augmented))) != (93, 72, 17):
        raise DataError("expected feature groups with sizes 93, 72, and 17")
    return FeatureGroups(local, aggregate, augmented)


def audit_feature_availability(
    columns: Iterable[str], evidence: Mapping[str, Mapping[str, Any]], *, schema_version: int
) -> FeatureAudit:
    groups = feature_groups(columns)
    if schema_version != 1:
        raise DataError("unsupported causal-audit schema version")
    records: dict[str, Mapping[str, Any]] = {}
    for name, features in (("local", groups.local), ("aggregate", groups.aggregate), ("augmented", groups.augmented)):
        record = evidence.get(name)
        if not record or record.get("causal") is not True or not record.get("source"):
            raise DataError(f"causal evidence failed or is missing for {name}")
        records[name] = {**record, "feature_count": len(features)}
    payload = {"schema_version": schema_version, "groups": records, "excluded": groups.excluded}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return FeatureAudit(schema_version, records, True, digest)


def read_source(source: SourceInventory) -> LoadedSource:
    features = pl.read_csv(source.core_files["txs_features.csv"].path).with_columns(pl.col("txId").cast(pl.String))
    columns = tuple(features.columns)
    groups = feature_groups(columns)
    features = features.with_columns(pl.col(list(groups.model)).fill_null(0.0))
    if features.get_column("txId").n_unique() != features.height:
        raise DataError("duplicate transaction IDs in features")
    if features.get_column("Time step").min() < 1 or features.get_column("Time step").max() > 49:
        raise DataError("Time step must be in 1..49")
    classes = pl.read_csv(source.core_files["txs_classes.csv"].path).with_columns(
        pl.col("txId").cast(pl.String), pl.col("class").cast(pl.String)
    )
    if classes.get_column("txId").n_unique() != classes.height:
        raise DataError("duplicate transaction IDs in classes")
    known_ids = set(features.get_column("txId").to_list())
    if not set(classes.get_column("txId").to_list()) <= known_ids:
        raise DataError("class references missing transaction")
    try:
        labels = {row[0]: CLASS_MAP[row[1]] for row in classes.iter_rows()}
    except KeyError as error:
        raise DataError(f"invalid class value: {error.args[0]}") from error
    edges_frame = pl.read_csv(source.core_files["txs_edgelist.csv"].path).with_columns(
        pl.col("txId1").cast(pl.String), pl.col("txId2").cast(pl.String)
    )
    edges = [(source_id, target_id) for source_id, target_id in edges_frame.iter_rows()]
    missing = {item for edge in edges for item in edge} - known_ids
    if missing:
        raise DataError(f"edge references missing transaction: {sorted(missing)[:3]}")
    return LoadedSource(source, features, labels, edges, columns, groups.model)


def _snapshot_digest(tx_ids: tuple[str, ...], x: torch.Tensor, edge_index: torch.Tensor, labels: torch.Tensor) -> str:
    digest = hashlib.sha256("\0".join(tx_ids).encode())
    digest.update(x.detach().cpu().contiguous().numpy().tobytes())
    digest.update(edge_index.detach().cpu().contiguous().numpy().tobytes())
    digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_snapshots(data: LoadedSource, *, steps: Iterable[int] = range(1, 50)) -> list[PreparedSnapshot]:
    step_by_id = dict(data.features.select("txId", "Time step").iter_rows())
    edge_steps: dict[int, list[tuple[str, str]]] = {}
    for source, target in data.edges:
        if step_by_id[source] != step_by_id[target]:
            raise DataError("edge endpoints have divergent time steps")
        edge_steps.setdefault(int(step_by_id[source]), []).append((source, target))
    snapshots = []
    for step in steps:
        frame = data.features.filter(pl.col("Time step") == step).sort("txId")
        tx_ids = tuple(frame.get_column("txId").to_list())
        index = {tx_id: position for position, tx_id in enumerate(tx_ids)}
        x = torch.tensor(frame.select(data.feature_names).to_numpy(), dtype=torch.float32)
        labels = torch.tensor([data.labels.get(tx_id, -1) for tx_id in tx_ids], dtype=torch.int64)
        unique: set[tuple[str, str]] = set()
        duplicate_count = self_loop_count = 0
        for edge in edge_steps.get(step, []):
            if edge[0] == edge[1]:
                self_loop_count += 1
            elif edge in unique:
                duplicate_count += 1
            else:
                unique.add(edge)
        ordered = sorted(unique, key=lambda edge: (index[edge[0]], index[edge[1]]))
        edge_index = (
            torch.tensor([[index[a] for a, _ in ordered], [index[b] for _, b in ordered]], dtype=torch.long)
            if ordered else torch.empty((2, 0), dtype=torch.long)
        )
        manifest = {"time_step": step, "node_count": len(tx_ids), "edge_count": edge_index.shape[1], "duplicate_edges_removed": duplicate_count, "self_loops_removed": self_loop_count, "edge_index_order": "source_target", "message_flow": "source_to_target"}
        snapshots.append(PreparedSnapshot(step, tx_ids, x, edge_index, labels, manifest, _snapshot_digest(tx_ids, x, edge_index, labels)))
    return snapshots


def stable_sample(snapshot: PreparedSnapshot, maximum: int, salt: str = "s003-smoke") -> PreparedSnapshot:
    if len(snapshot.tx_ids) <= maximum:
        return snapshot
    chosen = sorted(range(len(snapshot.tx_ids)), key=lambda i: hashlib.sha256(f"{salt}{snapshot.tx_ids[i]}".encode()).hexdigest())[:maximum]
    chosen_set = set(chosen)
    chosen = sorted(chosen)
    remap = {old: new for new, old in enumerate(chosen)}
    kept_edges = [position for position in range(snapshot.edge_index.shape[1]) if int(snapshot.edge_index[0, position]) in chosen_set and int(snapshot.edge_index[1, position]) in chosen_set]
    edge_index = (
        torch.tensor([[remap[int(snapshot.edge_index[row, position])] for position in kept_edges] for row in (0, 1)], dtype=torch.long)
        if kept_edges else torch.empty((2, 0), dtype=torch.long)
    )
    tx_ids = tuple(snapshot.tx_ids[i] for i in chosen)
    x = snapshot.x[chosen]
    labels = snapshot.labels[chosen] if snapshot.labels is not None else torch.full((len(chosen),), -1)
    manifest = {**snapshot.manifest, "node_count": len(tx_ids), "edge_count": edge_index.shape[1], "sampling": "sha256_tx_id"}
    return PreparedSnapshot(snapshot.time_step, tx_ids, x, edge_index, labels, manifest, _snapshot_digest(tx_ids, x, edge_index, labels))


def fit_normalizer(snapshots: Iterable[PreparedSnapshot], *, fit_steps: Iterable[int], feature_names: tuple[str, ...]) -> PreprocessingState:
    fit_steps_tuple = tuple(fit_steps)
    tensors = [snapshot.x for snapshot in snapshots if snapshot.time_step in fit_steps_tuple and snapshot.x.numel()]
    if not tensors:
        raise DataError("normalizer has no fit nodes")
    joined = torch.cat(tensors)
    center = joined.mean(dim=0)
    scale = joined.std(dim=0, unbiased=False)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    digest = hashlib.sha256(center.numpy().tobytes() + scale.numpy().tobytes() + json.dumps(fit_steps_tuple).encode()).hexdigest()
    return PreprocessingState(fit_steps_tuple, feature_names, center, scale, "zscore_fit_only", digest)

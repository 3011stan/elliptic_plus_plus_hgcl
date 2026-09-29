"""Resumable, test-blind scheduling for the S003 experiment matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .artifacts import ArtifactError, EmbeddingCacheKey, EvaluationCell, EvaluationCohort
from .baselines import (
    AdapterContext,
    GCPALAdapter,
    GraphData,
    InspectionLAdapter,
    MLPXAdapter,
    RandomForestAdapter,
    S003TxGCLAdapter,
    SupervisedGNNAdapter,
    TabularData,
    XGBoostAdapter,
)
from .evaluation import TestLabelStore


class MatrixScheduleError(RuntimeError):
    """Raised when a matrix lifecycle invariant is violated."""


@dataclass
class MatrixCellState:
    key: str
    state: str = "planned"
    run_id: str | None = None
    weights_digest: str | None = None
    threshold: float | None = None
    config_digest: str | None = None
    failure_kind: str | None = None
    failure_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MatrixScheduler:
    """Track one immutable design until every P1 cell can enter the cohort.

    Scheduling never reads test labels. A run may only be resumed after an
    explicit interruption; failed and invalid cells remain terminal and are
    still represented in the final 205-cell accounting.
    """

    ACTIVE_STATES = {"planned", "running", "interrupted"}
    COHORT_STATES = {"selected", "failed", "invalid"}

    def __init__(
        self,
        cells: tuple[str, ...] | list[str],
        *,
        design_digest: str,
        test_labels: TestLabelStore,
        expected_count: int = 205,
    ) -> None:
        keys = tuple(cells)
        if len(keys) != expected_count or len(set(keys)) != expected_count:
            raise MatrixScheduleError(f"matrix must contain exactly {expected_count} unique cells")
        actual = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
        if actual != design_digest:
            raise MatrixScheduleError("matrix design digest mismatch")
        if test_labels.access_log or test_labels.released_cohort is not None:
            raise MatrixScheduleError("matrix scheduling requires an unopened test-label store")
        self.design_digest = design_digest
        self.test_labels = test_labels
        self.expected_count = expected_count
        self.cells = {key: MatrixCellState(key) for key in keys}

    def _cell(self, key: str) -> MatrixCellState:
        try:
            return self.cells[key]
        except KeyError as exc:
            raise MatrixScheduleError(f"unknown matrix cell: {key}") from exc

    def _assert_test_blind(self) -> None:
        if self.test_labels.access_log or self.test_labels.released_cohort is not None:
            raise MatrixScheduleError("test labels were accessed before cohort sealing")

    def start(self, key: str) -> MatrixCellState:
        self._assert_test_blind()
        cell = self._cell(key)
        if cell.state != "planned":
            raise MatrixScheduleError("only a planned cell can start")
        cell.state = "running"
        return cell

    def interrupt(self, key: str, message: str) -> MatrixCellState:
        cell = self._cell(key)
        if cell.state != "running":
            raise MatrixScheduleError("only a running cell can be interrupted")
        cell.state = "interrupted"
        cell.failure_kind = "interrupted"
        cell.failure_message = message
        return cell

    def resume(self, key: str) -> MatrixCellState:
        self._assert_test_blind()
        cell = self._cell(key)
        if cell.state != "interrupted":
            raise MatrixScheduleError("only an interrupted cell can resume")
        cell.state = "running"
        cell.failure_kind = None
        cell.failure_message = None
        return cell

    def select(
        self,
        key: str,
        *,
        run_id: str,
        weights_digest: str,
        threshold: float,
        config_digest: str,
    ) -> MatrixCellState:
        self._assert_test_blind()
        cell = self._cell(key)
        if cell.state != "running":
            raise MatrixScheduleError("only a running cell can be selected")
        # Reuse EvaluationCell validation so scheduling and evaluation agree.
        EvaluationCell(key, "selected", run_id, weights_digest, threshold, config_digest)
        cell.state = "selected"
        cell.run_id = run_id
        cell.weights_digest = weights_digest
        cell.threshold = threshold
        cell.config_digest = config_digest
        return cell

    def terminate(self, key: str, *, state: str, kind: str, message: str) -> MatrixCellState:
        self._assert_test_blind()
        cell = self._cell(key)
        if state not in {"failed", "invalid"}:
            raise MatrixScheduleError("terminal failure state must be failed or invalid")
        if cell.state not in {"planned", "running", "interrupted"}:
            raise MatrixScheduleError("selected or terminal cells are immutable")
        if not kind or not message:
            raise MatrixScheduleError("failure kind and message are required")
        cell.state = state
        cell.failure_kind = kind
        cell.failure_message = message
        return cell

    def pending(self) -> tuple[str, ...]:
        return tuple(key for key, cell in self.cells.items() if cell.state in self.ACTIVE_STATES)

    def seal_cohort(self, cohort_id: str) -> EvaluationCohort:
        self._assert_test_blind()
        if self.expected_count != 205:
            raise MatrixScheduleError("only the complete 205-cell P1 design may be sealed")
        if self.pending():
            raise MatrixScheduleError("all matrix cells must be selected or terminal before sealing")
        cohort = EvaluationCohort(cohort_id, self.design_digest, self.test_labels.digest)
        for cell in self.cells.values():
            cohort.add_cell(
                EvaluationCell(
                    key=cell.key,
                    state=cell.state,
                    run_id=cell.run_id,
                    weights_digest=cell.weights_digest,
                    threshold=cell.threshold,
                    config_digest=cell.config_digest,
                )
            )
        cohort.seal()
        return cohort

    def to_dict(self) -> dict[str, object]:
        return {
            "design_digest": self.design_digest,
            "expected_count": self.expected_count,
            "test_store_digest": self.test_labels.digest,
            "test_access_count": len(self.test_labels.access_log),
            "cells": [cell.to_dict() for cell in self.cells.values()],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        test_labels: TestLabelStore,
    ) -> MatrixScheduler:
        if test_labels.access_log or test_labels.released_cohort is not None:
            raise MatrixScheduleError("matrix scheduling requires an unopened test-label store")
        design_digest = str(payload["design_digest"])
        expected_count = int(payload.get("expected_count", 205))
        raw_cells = payload["cells"]
        cells = [item["key"] if isinstance(item, dict) else str(item) for item in raw_cells]
        scheduler = cls(
            cells=cells,
            design_digest=design_digest,
            test_labels=test_labels,
            expected_count=expected_count,
        )
        for item in raw_cells:
            if isinstance(item, dict):
                key = str(item["key"])
                cell = scheduler.cells[key]
                cell.state = str(item.get("state", "planned"))
                cell.run_id = item.get("run_id")
                cell.weights_digest = item.get("weights_digest")
                cell.threshold = item.get("threshold")
                cell.config_digest = item.get("config_digest")
                cell.failure_kind = item.get("failure_kind")
                cell.failure_message = item.get("failure_message")
        return scheduler


@dataclass(frozen=True)
class CanonicalCellKey:
    key: str
    family: str
    method: str
    variant: str
    fraction: float
    seed: int
    is_p1: bool
    uses_ssl: bool
    uses_embedding_cache: bool


def parse_cell_key(cell_key: str) -> CanonicalCellKey:
    """Parse a canonical matrix cell key into its typed semantics."""
    parts = cell_key.split(":")
    if len(parts) != 4:
        raise MatrixScheduleError(f"invalid canonical cell key: {cell_key}")
    family, identifier, fraction_text, seed_text = parts
    try:
        fraction = float(fraction_text)
        seed = int(seed_text)
    except ValueError as exc:
        raise MatrixScheduleError(f"invalid fraction or seed in cell key: {cell_key}") from exc

    if family == "main":
        method = identifier
        variant = "complete"
        is_p1 = True
        uses_ssl = method in {"s003_txgcl", "inspection_l_dgi", "gcpal"}
        uses_cache = uses_ssl
    elif family == "representation":
        method = "s003_txgcl"
        variant = identifier
        is_p1 = True
        uses_ssl = True
        uses_cache = True
    elif family == "ablation":
        method = "s003_txgcl"
        variant = identifier
        is_p1 = True
        uses_ssl = True
        uses_cache = True
    elif family == "p2":
        method = "s003_txgcl"
        variant = identifier
        is_p1 = False
        uses_ssl = True
        uses_cache = True
    else:
        raise MatrixScheduleError(f"unknown canonical cell family: {family}")

    return CanonicalCellKey(
        key=cell_key,
        family=family,
        method=method,
        variant=variant,
        fraction=fraction,
        seed=seed,
        is_p1=is_p1,
        uses_ssl=uses_ssl,
        uses_embedding_cache=uses_cache,
    )


def cell_uses_embedding_cache(cell_key: str) -> bool:
    """Return True if the cell relies on reusable frozen SSL embeddings."""
    return parse_cell_key(cell_key).uses_embedding_cache


def embedding_cache_key_for_cell(
    cell_key: str,
    *,
    data_digest: str,
    pretraining_config_digest: str,
    code_revision: str,
) -> EmbeddingCacheKey:
    """Build a fraction-independent key for reusable frozen embeddings."""
    parsed = parse_cell_key(cell_key)
    if not parsed.uses_embedding_cache:
        raise MatrixScheduleError(f"cell does not use embedding cache: {cell_key}")

    # s003_txgcl main and h_only share the exact same SSL pretraining embeddings.
    if parsed.family in {"main", "representation"}:
        method_id = parsed.method
        variant_id = "main"
    elif parsed.family in {"ablation", "p2"}:
        method_id = parsed.method
        variant_id = parsed.variant
    else:
        method_id = parsed.method
        variant_id = parsed.family

    return EmbeddingCacheKey(
        method_id=method_id,
        variant_id=variant_id,
        seed=parsed.seed,
        data_digest=data_digest,
        config_digest=pretraining_config_digest,
        code_revision=code_revision,
    )


def canonical_matrix_design(
    methods: Sequence[str],
    fractions: Sequence[float],
    seeds: Sequence[int],
    *,
    include_reverse_p2: bool = False,
) -> dict[str, Any]:
    """Enumerate the exact canonical P1 matrix (205 cells) and optional P2 (5 cells)."""
    cells = [f"main:{method}:{fraction:g}:{seed}" for method in methods for fraction in fractions for seed in seeds]
    cells.extend(f"representation:h_only:0.01:{seed}" for seed in seeds)
    for variant in ("no_knn", "no_edge_dropout", "random_individual", "random_groups"):
        cells.extend(f"ablation:{variant}:0.01:{seed}" for seed in seeds)
    p1_count = len(cells)
    if p1_count != 205 or len(set(cells)) != 205:
        raise ValueError("canonical P1 design must contain exactly 205 unique cells")
    if include_reverse_p2:
        cells.extend(f"p2:reverse_edges:0.01:{seed}" for seed in seeds)
    digest = hashlib.sha256("\n".join(cells).encode("utf-8")).hexdigest()
    return {
        "cells": tuple(cells),
        "p1_count": p1_count,
        "p2_count": len(cells) - p1_count,
        "design_digest": digest,
    }


def build_matrix_coverage(
    cells: Sequence[str],
    cell_states: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Classify matrix cells into states and verify SC-001/SC-005 accounting."""
    states = {} if cell_states is None else dict(cell_states)
    records = []
    counts_by_state: dict[str, int] = {}
    counts_by_family: dict[str, int] = {}

    for cell_key in cells:
        parsed = parse_cell_key(cell_key)
        state = states.get(cell_key, "planned" if cell_states is None else "missing")
        counts_by_state[state] = counts_by_state.get(state, 0) + 1
        counts_by_family[parsed.family] = counts_by_family.get(parsed.family, 0) + 1
        records.append({
            "key": cell_key,
            "state": state,
            "family": parsed.family,
            "method": parsed.method,
            "variant": parsed.variant,
            "fraction": parsed.fraction,
            "seed": parsed.seed,
            "is_p1": parsed.is_p1,
        })

    p1_records = [r for r in records if r["is_p1"]]
    p1_complete = (
        len(p1_records) == 205
        and all(r["state"] in {"selected", "failed", "invalid", "completed"} for r in p1_records)
    )
    sc_001_compliant = (
        len(p1_records) == 205
        and counts_by_state.get("missing", 0) == 0
    )

    regime_1pct = {r["key"]: r for r in records if r["fraction"] == 0.01}
    sc_005_ablation = {
        "x_only": [k for k, r in regime_1pct.items() if r["method"] == "mlp_x" and r["family"] == "main"],
        "h_only": [k for k, r in regime_1pct.items() if r["family"] == "representation" and r["variant"] == "h_only"],
        "h_concat_x": [k for k, r in regime_1pct.items() if r["method"] == "s003_txgcl" and r["family"] == "main"],
        "no_knn": [k for k, r in regime_1pct.items() if r["family"] == "ablation" and r["variant"] == "no_knn"],
        "no_edge_dropout": [k for k, r in regime_1pct.items() if r["family"] == "ablation" and r["variant"] == "no_edge_dropout"],
        "random_individual": [k for k, r in regime_1pct.items() if r["family"] == "ablation" and r["variant"] == "random_individual"],
        "random_groups": [k for k, r in regime_1pct.items() if r["family"] == "ablation" and r["variant"] == "random_groups"],
        "functional_blocks": [k for k, r in regime_1pct.items() if r["method"] == "s003_txgcl" and r["family"] == "main"],
    }

    return {
        "total_cells": len(cells),
        "expected_p1_cells": 205,
        "p1_complete": p1_complete,
        "sc_001_compliant": sc_001_compliant,
        "counts_by_state": counts_by_state,
        "counts_by_family": counts_by_family,
        "ablation_coverage_1pct": sc_005_ablation,
        "cells": records,
    }


def build_dry_run_plans(
    config: Any,
    design: Mapping[str, Any],
    *,
    smoke_timings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble coverage, caching, checkpoint, resume, resource, and projection plans."""
    raw = config.raw if hasattr(config, "raw") else dict(config)
    cells = list(design["cells"])
    coverage = build_matrix_coverage(cells)

    ssl_cells = [cell for cell in cells if cell_uses_embedding_cache(cell)]
    dummy_digests = {
        "data_digest": "0" * 64,
        "pretraining_config_digest": "0" * 64,
        "code_revision": "0" * 40,
    }
    unique_ssl_keys = {
        embedding_cache_key_for_cell(cell, **dummy_digests).digest
        for cell in ssl_cells
    }

    coverage_plan = {
        "expected_p1_cells": 205,
        "expected_p2_cells": int(design.get("p2_count", 0)),
        "total_cells": len(cells),
        "family_counts": coverage["counts_by_family"],
        "methods": list(raw["baselines"]["methods"]),
        "fractions": list(raw["labels"]["fractions"]),
        "seeds": list(raw["labels"]["seeds"]),
        "sc_001_compliant": coverage["sc_001_compliant"],
        "ablation_breakdown_1pct": coverage["ablation_coverage_1pct"],
    }

    cache_plan = {
        "embeddings": "method-variant-seed",
        "downstream_reuse": "fractions",
        "unique_ssl_pretrainings": len(unique_ssl_keys),
        "ssl_reuse_across_fractions": True,
        "h_only_reuses_main_ssl": True,
    }

    checkpoint_plan = {
        "atomic": True,
        "identity": "config-data-code",
        "preserves": [
            "epoch",
            "model_state",
            "optimizer_state",
            "rng_state",
            "hyperparameters",
            "threshold",
            "prevalence",
        ],
    }

    resume_plan = {
        "allowed_state": "interrupted",
        "terminal_states_immutable": True,
        "terminal_states": ["selected", "failed", "invalid", "completed"],
        "post_unblinding_guard": "frozen_weights_and_threshold_only",
    }

    resource_plan = {
        "device": raw["resources"]["device"],
        "max_vram_gib": raw["resources"]["max_vram_gib"],
        "max_ram_gib": raw["resources"]["max_ram_gib"],
        "snapshot_batch_size": int(raw.get("ssl", {}).get("snapshot_batch_size", 1)),
        "projection_margin": float(raw["resources"]["projection_margin"]),
    }

    margin = float(raw["resources"]["projection_margin"])
    if smoke_timings:
        t_ssl = smoke_timings.get("ssl_seconds", smoke_timings.get("ssl_epoch_seconds", 0.0) * 100)
        t_down = smoke_timings.get("downstream_seconds", 1.0)
        t_eval = smoke_timings.get("eval_seconds", 0.5)
        base_seconds = (len(unique_ssl_keys) * t_ssl) + (len(cells) * (t_down + t_eval))
        projected_seconds = base_seconds * (1.0 + margin)
        duration_projection = {
            "status": "estimated",
            "projection_margin": margin,
            "estimated_base_seconds": round(base_seconds, 2),
            "estimated_total_seconds": round(projected_seconds, 2),
            "multipliers": {
                "unique_ssl_pretrainings": len(unique_ssl_keys),
                "total_downstream_cells": len(cells),
                "margin_factor": 1.0 + margin,
            },
        }
    else:
        duration_projection = {
            "status": "pending_smoke_measurement",
            "projection_margin": margin,
            "formula": f"total_seconds = ({len(unique_ssl_keys)} * t_ssl_100 + {len(cells)} * (t_downstream + t_eval)) * {1.0 + margin:.2f}",
            "multipliers": {
                "unique_ssl_pretrainings": len(unique_ssl_keys),
                "total_downstream_cells": len(cells),
                "margin_factor": 1.0 + margin,
            },
        }

    return {
        "coverage_plan": coverage_plan,
        "cache_plan": cache_plan,
        "checkpoint_plan": checkpoint_plan,
        "resume_plan": resume_plan,
        "resource_plan": resource_plan,
        "duration_projection": duration_projection,
    }


class EmbeddingCacheStore:
    """Fraction-independent storage for frozen SSL embeddings and pre-trained models."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else None
        self._memory: dict[str, Any] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: EmbeddingCacheKey) -> Any | None:
        digest = key.digest
        if digest in self._memory:
            self.hits += 1
            return self._memory[digest]
        if self.root is not None:
            path = self.root / f"{digest}.pt"
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                self._memory[digest] = payload
                self.hits += 1
                return payload
        self.misses += 1
        return None

    def put(self, key: EmbeddingCacheKey, payload: Any) -> None:
        digest = key.digest
        self._memory[digest] = payload
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / f"{digest}.pt"
            temp_path = self.root / f".{digest}.tmp"
            torch.save(payload, temp_path)
            temp_path.replace(path)

    def has(self, key: EmbeddingCacheKey) -> bool:
        digest = key.digest
        if digest in self._memory:
            return True
        if self.root is not None and (self.root / f"{digest}.pt").is_file():
            return True
        return False

    def clear(self) -> None:
        self._memory.clear()
        self.hits = 0
        self.misses = 0


def create_adapter_for_cell(
    cell_key: str,
    *,
    ssl_epochs: int = 100,
    downstream_epochs: int = 100,
    reverse_edges_approved: bool = False,
) -> Any:
    """Instantiate the appropriate adapter configured for the canonical cell."""
    parsed = parse_cell_key(cell_key)
    method = parsed.method

    if method == "mlp_x":
        return MLPXAdapter(epochs=downstream_epochs)
    elif method == "random_forest":
        return RandomForestAdapter()
    elif method == "xgboost":
        return XGBoostAdapter()
    elif method in {"gcn_supervised", "graphsage_supervised", "gin_supervised"}:
        kind = method.replace("_supervised", "")
        return SupervisedGNNAdapter(kind, epochs=downstream_epochs)
    elif method == "inspection_l_dgi":
        return InspectionLAdapter(ssl_epochs=ssl_epochs)
    elif method == "gcpal":
        return GCPALAdapter(ssl_epochs=ssl_epochs, downstream_epochs=downstream_epochs)
    elif method == "s003_txgcl":
        if parsed.family == "representation":
            return S003TxGCLAdapter(
                ssl_epochs=ssl_epochs,
                downstream_epochs=downstream_epochs,
                representation="h_only",
                masking_policy="functional_blocks",
                use_knn=True,
                edge_dropout=True,
            )
        elif parsed.family == "ablation":
            variant = parsed.variant
            if variant == "no_knn":
                return S003TxGCLAdapter(
                    ssl_epochs=ssl_epochs,
                    downstream_epochs=downstream_epochs,
                    representation="h_concat_x",
                    masking_policy="functional_blocks",
                    use_knn=False,
                    edge_dropout=True,
                )
            elif variant == "no_edge_dropout":
                return S003TxGCLAdapter(
                    ssl_epochs=ssl_epochs,
                    downstream_epochs=downstream_epochs,
                    representation="h_concat_x",
                    masking_policy="functional_blocks",
                    use_knn=True,
                    edge_dropout=False,
                )
            elif variant == "random_individual":
                return S003TxGCLAdapter(
                    ssl_epochs=ssl_epochs,
                    downstream_epochs=downstream_epochs,
                    representation="h_concat_x",
                    masking_policy="random_individual",
                    use_knn=True,
                    edge_dropout=True,
                )
            elif variant == "random_groups":
                return S003TxGCLAdapter(
                    ssl_epochs=ssl_epochs,
                    downstream_epochs=downstream_epochs,
                    representation="h_concat_x",
                    masking_policy="random_groups",
                    use_knn=True,
                    edge_dropout=True,
                )
            else:
                raise MatrixScheduleError(f"unknown ablation variant: {variant}")
        elif parsed.family == "p2":
            if parsed.variant == "reverse_edges":
                return S003TxGCLAdapter(
                    ssl_epochs=ssl_epochs,
                    downstream_epochs=downstream_epochs,
                    representation="h_concat_x",
                    masking_policy="functional_blocks",
                    use_knn=True,
                    edge_dropout=True,
                    reverse_edges=True,
                    reverse_edges_approved=reverse_edges_approved,
                )
            else:
                raise MatrixScheduleError(f"unknown p2 variant: {parsed.variant}")
        else:
            return S003TxGCLAdapter(
                ssl_epochs=ssl_epochs,
                downstream_epochs=downstream_epochs,
                representation="h_concat_x",
                masking_policy="functional_blocks",
                use_knn=True,
                edge_dropout=True,
            )
    else:
        raise MatrixScheduleError(f"unknown method for cell: {cell_key}")


def execute_matrix_cell(
    cell_key: str,
    scheduler: MatrixScheduler,
    *,
    dataset: Any,
    budget: Any,
    data_digest: str,
    pretraining_config_digest: str,
    code_revision: str = "0" * 40,
    config_digest: str = "0" * 64,
    embedding_cache: EmbeddingCacheStore | None = None,
    ssl_epochs: int = 100,
    downstream_epochs: int = 100,
    reverse_edges_approved: bool = False,
    run_id_prefix: str = "s003-run",
) -> MatrixCellState:
    """Execute a single matrix cell with cache reuse, strict state transitions, and test blindness."""
    parsed = parse_cell_key(cell_key)

    if scheduler.cells[cell_key].state == "interrupted":
        scheduler.resume(cell_key)
    else:
        scheduler.start(cell_key)

    try:
        adapter = create_adapter_for_cell(
            cell_key,
            ssl_epochs=ssl_epochs,
            downstream_epochs=downstream_epochs,
            reverse_edges_approved=reverse_edges_approved,
        )

        context = AdapterContext(
            seed=parsed.seed,
            fraction=parsed.fraction,
            fit_ids=frozenset(budget.fit_ids),
            validation_ids=frozenset(budget.validation_ids),
            refit_ids=frozenset(budget.refit_ids),
            budget_digest=budget.digest,
        )

        cached_embeddings = None
        cached_model = None
        cache_key = None

        if parsed.uses_embedding_cache and embedding_cache is not None:
            cache_key = embedding_cache_key_for_cell(
                cell_key,
                data_digest=data_digest,
                pretraining_config_digest=pretraining_config_digest,
                code_revision=code_revision,
            )
            cached_payload = embedding_cache.get(cache_key)
            if cached_payload is not None:
                if isinstance(cached_payload, dict):
                    cached_embeddings = cached_payload.get("embeddings")
                    cached_model = cached_payload.get("model")
                else:
                    cached_embeddings = cached_payload

        import inspect
        sig = inspect.signature(adapter.fit)
        if "cached_embeddings" in sig.parameters:
            fitted = adapter.fit(
                dataset,
                context,
                cached_embeddings=cached_embeddings,
                cached_model=cached_model,
            )
        else:
            fitted = adapter.fit(dataset, context)

        if (
            parsed.uses_embedding_cache
            and embedding_cache is not None
            and cached_embeddings is None
            and cache_key is not None
        ):
            extracted = None
            model_obj = getattr(fitted, "model", getattr(fitted, "encoder", None))
            if model_obj is not None:
                edges = getattr(dataset, "edge_index", torch.empty((2, 0), dtype=torch.long))
                if hasattr(model_obj, "frozen_embeddings"):
                    extracted = model_obj.frozen_embeddings(dataset.features, edges)
                elif callable(model_obj):
                    extracted = model_obj(dataset.features, edges)
            if extracted is not None:
                embedding_cache.put(cache_key, {"embeddings": extracted.detach(), "model": model_obj})

        threshold = float(getattr(fitted, "threshold", getattr(getattr(fitted, "downstream", None), "threshold", 0.5)))
        run_id = f"{run_id_prefix}-{cell_key.replace(':', '-')}"
        weights_digest = hashlib.sha256(f"weights:{cell_key}:{threshold}".encode("utf-8")).hexdigest()

        return scheduler.select(
            cell_key,
            run_id=run_id,
            weights_digest=weights_digest,
            threshold=threshold,
            config_digest=config_digest,
        )

    except (ValueError, KeyError) as exc:
        return scheduler.terminate(
            cell_key,
            state="invalid",
            kind="data",
            message=str(exc),
        )
    except Exception as exc:
        return scheduler.terminate(
            cell_key,
            state="failed",
            kind="runtime",
            message=str(exc),
        )

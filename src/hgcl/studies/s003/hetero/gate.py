"""Causality, supervision, comparability, and resource gates for the heterogeneous extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import S003Config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HeteroGateError(RuntimeError):
    """Raised on invalid heterogeneous extension configuration or gate invariant violation."""


@dataclass(frozen=True)
class GateCheckResult:
    """Individual gate audit result with evidence."""

    gate_name: str
    passed: bool
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class HeterogeneousExtensionDecision:
    """Consolidated decision recording all 4 gates and include/defer resolution (Section 15, FR-030)."""

    causality_gate: GateCheckResult
    supervision_gate: GateCheckResult
    comparability_gate: GateCheckResult
    resources_gate: GateCheckResult
    decision: str  # "include" or "defer"
    approved_by: str | None = None
    study_id: str = "s003"
    created_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        if self.decision not in {"include", "defer"}:
            raise HeteroGateError(f"invalid decision '{self.decision}': must be 'include' or 'defer'")

        all_passed = (
            self.causality_gate.passed
            and self.supervision_gate.passed
            and self.comparability_gate.passed
            and self.resources_gate.passed
        )

        if self.decision == "include":
            if not all_passed:
                failing = [
                    g.gate_name
                    for g in (self.causality_gate, self.supervision_gate, self.comparability_gate, self.resources_gate)
                    if not g.passed
                ]
                raise HeteroGateError(f"cannot include heterogeneous extension when gates failed: {failing}")
            if not self.approved_by or not str(self.approved_by).strip():
                raise HeteroGateError("decision 'include' requires explicit approved_by authorization")

    @property
    def is_included(self) -> bool:
        return self.decision == "include"

    @property
    def is_deferred(self) -> bool:
        return self.decision == "defer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "decision": self.decision,
            "approved_by": self.approved_by,
            "created_at": self.created_at,
            "gates": {
                "causality": self.causality_gate.to_dict(),
                "supervision": self.supervision_gate.to_dict(),
                "comparability": self.comparability_gate.to_dict(),
                "resources": self.resources_gate.to_dict(),
            },
        }


def check_causality_gate(
    *,
    data_root: str | Path | None = None,
    address_files_present: bool = False,
    features_are_causal_per_snapshot: bool = False,
    evidence: Mapping[str, Any] | None = None,
) -> GateCheckResult:
    """Check causality gate (FR-030, FR-031): address attributes must not incorporate future events."""
    ev = dict(evidence or {})
    if not address_files_present:
        return GateCheckResult(
            gate_name="causality",
            passed=False,
            reason="address evidence or feature files are missing from data root",
            evidence={**ev, "address_files_present": False},
        )
    if not features_are_causal_per_snapshot:
        return GateCheckResult(
            gate_name="causality",
            passed=False,
            reason="address features contain global lifetime aggregations violating causal snapshot isolation",
            evidence={**ev, "features_are_causal_per_snapshot": False},
        )
    return GateCheckResult(
        gate_name="causality",
        passed=True,
        reason="address features are indexed causally per snapshot without future leakage",
        evidence={**ev, "address_files_present": True, "features_are_causal_per_snapshot": True},
    )


def check_supervision_gate(
    *,
    target_node_type: str = "tx",
    supervision_loss: str = "tx_binary_classification",
    wallet_labels_used: bool = False,
    evidence: Mapping[str, Any] | None = None,
) -> GateCheckResult:
    """Check supervision gate (FR-031): target must remain transaction classification."""
    ev = dict(evidence or {})
    if target_node_type != "tx":
        return GateCheckResult(
            gate_name="supervision",
            passed=False,
            reason=f"target node type must be 'tx', got '{target_node_type}'",
            evidence={**ev, "target_node_type": target_node_type},
        )
    if wallet_labels_used:
        return GateCheckResult(
            gate_name="supervision",
            passed=False,
            reason="wallet labels must not be used to supervise the model or leak into loss",
            evidence={**ev, "wallet_labels_used": True},
        )
    return GateCheckResult(
        gate_name="supervision",
        passed=True,
        reason="supervision strictly targets transaction illicit classification without address labels",
        evidence={**ev, "target_node_type": "tx", "wallet_labels_used": False},
    )


def check_comparability_gate(
    *,
    development_steps: Sequence[int] = tuple(range(1, 35)),
    test_steps: Sequence[int] = tuple(range(35, 50)),
    fractions: Sequence[float] = (0.01, 0.05, 0.1, 1.0),
    seeds: Sequence[int] = (11, 23, 37, 53, 71),
    evidence: Mapping[str, Any] | None = None,
) -> GateCheckResult:
    """Check comparability gate (FR-022, FR-030): must use identical splits, budgets, and seeds."""
    ev = dict(evidence or {})
    expected_dev = tuple(range(1, 35))
    expected_test = tuple(range(35, 50))
    expected_fractions = (0.01, 0.05, 0.1, 1.0)
    expected_seeds = (11, 23, 37, 53, 71)

    if tuple(development_steps) != expected_dev or tuple(test_steps) != expected_test:
        return GateCheckResult(
            gate_name="comparability",
            passed=False,
            reason="temporal split steps do not match canonical 1..34 and 35..49 partition",
            evidence={**ev, "development_steps": list(development_steps), "test_steps": list(test_steps)},
        )
    if tuple(sorted(fractions)) != expected_fractions:
        return GateCheckResult(
            gate_name="comparability",
            passed=False,
            reason="label fractions do not match canonical [0.01, 0.05, 0.1, 1.0]",
            evidence={**ev, "fractions": list(fractions)},
        )
    if tuple(seeds) != expected_seeds:
        return GateCheckResult(
            gate_name="comparability",
            passed=False,
            reason="seeds do not match canonical [11, 23, 37, 53, 71]",
            evidence={**ev, "seeds": list(seeds)},
        )
    return GateCheckResult(
        gate_name="comparability",
        passed=True,
        reason="splits, budgets, and seeds strictly match the homogeneous core",
        evidence={**ev, "fractions": list(fractions), "seeds": list(seeds)},
    )


def check_resources_gate(
    *,
    projected_ram_gib: float,
    max_ram_gib: float,
    projected_duration_seconds: float,
    max_duration_seconds: float,
    evidence: Mapping[str, Any] | None = None,
) -> GateCheckResult:
    """Check resources gate (FR-030): memory and duration projections must fit declared budgets."""
    ev = dict(evidence or {})
    if projected_ram_gib > max_ram_gib:
        return GateCheckResult(
            gate_name="resources",
            passed=False,
            reason=f"projected RAM ({projected_ram_gib:.1f} GiB) exceeds limit ({max_ram_gib:.1f} GiB)",
            evidence={**ev, "projected_ram_gib": projected_ram_gib, "max_ram_gib": max_ram_gib},
        )
    if projected_duration_seconds > max_duration_seconds:
        return GateCheckResult(
            gate_name="resources",
            passed=False,
            reason=f"projected duration ({projected_duration_seconds:.0f}s) exceeds limit ({max_duration_seconds:.0f}s)",
            evidence={**ev, "projected_duration_seconds": projected_duration_seconds, "max_duration_seconds": max_duration_seconds},
        )
    return GateCheckResult(
        gate_name="resources",
        passed=True,
        reason="projected RAM and execution duration are within declared limits",
        evidence={**ev, "projected_ram_gib": projected_ram_gib, "projected_duration_seconds": projected_duration_seconds},
    )


def evaluate_heterogeneous_gate(
    config: S003Config | Mapping[str, Any],
    *,
    data_root: str | Path | None = None,
    address_files_present: bool = False,
    features_are_causal_per_snapshot: bool = False,
    projected_ram_gib: float = 64.0,
    projected_duration_seconds: float = 86400.0,
    approved_by: str | None = None,
    output_path: str | Path | None = None,
) -> HeterogeneousExtensionDecision:
    """Evaluate 100% of heterogeneous gates and record reproducible include/defer decision (FR-030–FR-032, SC-008)."""
    raw = config.raw if isinstance(config, S003Config) else config

    # 1. Causality gate
    causality = check_causality_gate(
        data_root=data_root or raw.get("paths", {}).get("data_root"),
        address_files_present=address_files_present,
        features_are_causal_per_snapshot=features_are_causal_per_snapshot,
    )

    # 2. Supervision gate
    supervision = check_supervision_gate(
        target_node_type=raw.get("target_node_type", "tx"),
        supervision_loss=raw.get("task", "tx_binary_classification"),
        wallet_labels_used=bool(raw.get("wallet_labels_used", False)),
    )

    # 3. Comparability gate
    dev_steps = raw.get("split", {}).get("development_steps", tuple(range(1, 35)))
    test_steps = raw.get("split", {}).get("test_steps", tuple(range(35, 50)))
    fractions = raw.get("labels", {}).get("fractions", (0.01, 0.05, 0.1, 1.0))
    seeds = raw.get("labels", {}).get("seeds", (11, 23, 37, 53, 71))
    comparability = check_comparability_gate(
        development_steps=dev_steps,
        test_steps=test_steps,
        fractions=fractions,
        seeds=seeds,
    )

    # 4. Resources gate
    max_ram = float(raw.get("resources", {}).get("max_ram_gib", 16.0))
    max_duration = float(raw.get("resources", {}).get("max_duration_seconds", 36000.0))
    resources = check_resources_gate(
        projected_ram_gib=projected_ram_gib,
        max_ram_gib=max_ram,
        projected_duration_seconds=projected_duration_seconds,
        max_duration_seconds=max_duration,
    )

    all_passed = causality.passed and supervision.passed and comparability.passed and resources.passed
    decision_value = "include" if (all_passed and approved_by) else "defer"

    decision = HeterogeneousExtensionDecision(
        causality_gate=causality,
        supervision_gate=supervision,
        comparability_gate=comparability,
        resources_gate=resources,
        decision=decision_value,
        approved_by=approved_by if decision_value == "include" else None,
    )

    if output_path is not None:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target.parent / f".{target.name}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(decision.to_dict(), f, indent=2, sort_keys=True)
        temp_file.replace(target)

    return decision

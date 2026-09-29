from __future__ import annotations

import pytest

from hgcl.studies.s003.hetero import (
    GateCheckResult,
    HeteroGateError,
    HeterogeneousExtensionDecision,
    check_causality_gate,
    check_comparability_gate,
    check_resources_gate,
    check_supervision_gate,
)


def _passing_gate(name: str) -> GateCheckResult:
    return GateCheckResult(gate_name=name, passed=True, reason="ok", evidence={})


def _failing_gate(name: str, reason: str = "failed") -> GateCheckResult:
    return GateCheckResult(gate_name=name, passed=False, reason=reason, evidence={})


def test_heterogeneous_decision_contract_and_invariants() -> None:
    """Test contract rules of HeterogeneousExtensionDecision (FR-030–FR-032, SC-008)."""
    # 1. Invalid decision value raises HeteroGateError
    with pytest.raises(HeteroGateError, match="invalid decision"):
        HeterogeneousExtensionDecision(
            causality_gate=_passing_gate("causality"),
            supervision_gate=_passing_gate("supervision"),
            comparability_gate=_passing_gate("comparability"),
            resources_gate=_passing_gate("resources"),
            decision="unapproved_execution",
        )

    # 2. Cannot include when any gate fails
    with pytest.raises(HeteroGateError, match="cannot include heterogeneous extension when gates failed"):
        HeterogeneousExtensionDecision(
            causality_gate=_failing_gate("causality", "future leakage"),
            supervision_gate=_passing_gate("supervision"),
            comparability_gate=_passing_gate("comparability"),
            resources_gate=_passing_gate("resources"),
            decision="include",
            approved_by="researcher",
        )

    # 3. Cannot include without explicit approved_by
    with pytest.raises(HeteroGateError, match="requires explicit approved_by authorization"):
        HeterogeneousExtensionDecision(
            causality_gate=_passing_gate("causality"),
            supervision_gate=_passing_gate("supervision"),
            comparability_gate=_passing_gate("comparability"),
            resources_gate=_passing_gate("resources"),
            decision="include",
            approved_by=None,
        )

    with pytest.raises(HeteroGateError, match="requires explicit approved_by authorization"):
        HeterogeneousExtensionDecision(
            causality_gate=_passing_gate("causality"),
            supervision_gate=_passing_gate("supervision"),
            comparability_gate=_passing_gate("comparability"),
            resources_gate=_passing_gate("resources"),
            decision="include",
            approved_by="   ",
        )

    # 4. Valid include when all 4 pass and approved
    incl = HeterogeneousExtensionDecision(
        causality_gate=_passing_gate("causality"),
        supervision_gate=_passing_gate("supervision"),
        comparability_gate=_passing_gate("comparability"),
        resources_gate=_passing_gate("resources"),
        decision="include",
        approved_by="researcher",
    )
    assert incl.is_included is True
    assert incl.is_deferred is False

    # 5. Defer records 100% of gates (SC-008)
    defer = HeterogeneousExtensionDecision(
        causality_gate=_failing_gate("causality", "missing files"),
        supervision_gate=_passing_gate("supervision"),
        comparability_gate=_passing_gate("comparability"),
        resources_gate=_failing_gate("resources", "OOM projection"),
        decision="defer",
    )
    assert defer.is_deferred is True
    assert defer.is_included is False
    data = defer.to_dict()
    assert len(data["gates"]) == 4
    assert set(data["gates"].keys()) == {"causality", "supervision", "comparability", "resources"}
    assert data["gates"]["causality"]["passed"] is False
    assert data["gates"]["resources"]["passed"] is False
    assert data["gates"]["supervision"]["passed"] is True

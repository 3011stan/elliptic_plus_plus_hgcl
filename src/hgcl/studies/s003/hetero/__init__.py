"""Non-blocking heterogeneous-extension gate namespace."""

from .gate import (
    GateCheckResult,
    HeteroGateError,
    HeterogeneousExtensionDecision,
    check_causality_gate,
    check_comparability_gate,
    check_resources_gate,
    check_supervision_gate,
    evaluate_heterogeneous_gate,
)

__all__ = [
    "GateCheckResult",
    "HeteroGateError",
    "HeterogeneousExtensionDecision",
    "check_causality_gate",
    "check_comparability_gate",
    "check_resources_gate",
    "check_supervision_gate",
    "evaluate_heterogeneous_gate",
]

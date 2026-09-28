from __future__ import annotations

import pytest

from hgcl.studies.s003.baselines import (
    AdapterContext,
    BaselineContractError,
    FitPrevalence,
    MethodRegistry,
    default_method_registry,
)


EXPECTED_METHODS = {
    "mlp_x",
    "random_forest",
    "xgboost",
    "gcn_supervised",
    "graphsage_supervised",
    "gin_supervised",
    "inspection_l_dgi",
    "gcpal",
    "s003_txgcl",
}


def test_registry_contains_exactly_the_nine_predeclared_methods() -> None:
    registry = default_method_registry()
    assert set(registry.method_ids) == EXPECTED_METHODS
    assert len(registry.method_ids) == 9
    registry.assert_complete()


def test_context_fixes_fairness_contract_for_every_adapter() -> None:
    context = AdapterContext(
        seed=11,
        fraction=0.01,
        fit_ids=frozenset({"a", "b"}),
        validation_ids=frozenset({"c", "d"}),
        refit_ids=frozenset({"a", "b", "c", "d"}),
        budget_digest="a" * 64,
    )
    assert context.target_node_type == "transaction"
    assert context.edge_index_order == "source_target"
    assert context.message_flow == "source_to_target"
    assert context.feature_count == 182
    assert context.selection_primary == "f1_illicit"
    assert context.selection_tiebreaker == "mcc"
    assert context.evaluation_policy == "pooled_and_per_snapshot"

    with pytest.raises(BaselineContractError, match="refit"):
        AdapterContext(
            seed=11,
            fraction=0.01,
            fit_ids=frozenset({"a"}),
            validation_ids=frozenset({"b"}),
            refit_ids=frozenset({"a"}),
            budget_digest="a" * 64,
        )


def test_prevalence_is_derived_only_from_fit_labels() -> None:
    prevalence = FitPrevalence.from_fit_labels([1, 0, 0, 0])
    assert prevalence.counts == {0: 3, 1: 1}
    assert prevalence.neural_class_weights == {0: 2 / 3, 1: 2.0}
    assert prevalence.rf_class_weight == {0: 2 / 3, 1: 2.0}
    assert prevalence.xgb_scale_pos_weight == 3.0

    with pytest.raises(BaselineContractError, match="both classes"):
        FitPrevalence.from_fit_labels([0, 0])
    with pytest.raises(BaselineContractError, match="known binary"):
        FitPrevalence.from_fit_labels([0, -1, 1])


def test_characteristic_component_gap_is_marked_approximation() -> None:
    registry = default_method_registry()
    exact = registry.assess_fidelity(
        "inspection_l_dgi", {"gin_2x128", "dgi", "random_forest_100"}
    )
    assert exact.status == "exact"
    assert exact.missing_components == ()

    approximation = registry.assess_fidelity(
        "gcpal", {"gin_2x128", "two_stochastic_views", "knn_k10"}
    )
    assert approximation.status == "approximation"
    assert set(approximation.missing_components) == {
        "multi_positive_loss",
        "mlp_2layer_h_concat_x",
    }


def test_registry_rejects_missing_or_extra_methods() -> None:
    complete = default_method_registry()
    definitions = dict(complete.definitions)
    definitions.pop("xgboost")
    with pytest.raises(BaselineContractError, match="registry mismatch"):
        MethodRegistry(definitions).assert_complete()

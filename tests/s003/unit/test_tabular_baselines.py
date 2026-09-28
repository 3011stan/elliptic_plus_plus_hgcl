from __future__ import annotations

import torch

from hgcl.studies.s003.baselines import (
    AdapterContext,
    MLPXAdapter,
    RandomForestAdapter,
    TabularData,
    XGBoostAdapter,
)


def _dataset_and_context() -> tuple[TabularData, AdapterContext]:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(40, 182, generator=generator)
    labels = torch.tensor([0, 1] * 20)
    features[:, 0] = labels.float() * 4 - 2
    ids = tuple(f"tx-{index}" for index in range(40))
    fit = frozenset(ids[:20])
    validation = frozenset(ids[20:30])
    refit = fit | validation
    return (
        TabularData(ids, features, labels),
        AdapterContext(11, 0.01, fit, validation, refit, "a" * 64),
    )


def test_tabular_defaults_match_predeclared_protocol() -> None:
    assert RandomForestAdapter().n_estimators == 300
    assert XGBoostAdapter().n_estimators == 500
    assert MLPXAdapter().epochs == 100


def test_mlp_x_uses_shared_selection_and_fit_only_prevalence() -> None:
    data, context = _dataset_and_context()
    fitted = MLPXAdapter(epochs=2, patience=1).fit(data, context)
    scores = MLPXAdapter().predict_scores(fitted, data)[0]
    assert fitted.method_id == "mlp_x"
    assert fitted.prevalence.counts == {0: 10, 1: 10}
    assert fitted.hyperparameters["fit_pos_weight"] == 1.0
    assert fitted.threshold in {0.3, 0.5, 0.7}
    assert scores.shape == (40,)


def test_random_forest_uses_300_tree_protocol_and_fit_weights() -> None:
    data, context = _dataset_and_context()
    adapter = RandomForestAdapter(n_estimators=10)
    fitted = adapter.fit(data, context)
    scores = adapter.predict_scores(fitted, data)[0]
    assert fitted.hyperparameters["n_estimators"] == 10
    assert fitted.estimator.class_weight == {0: 1.0, 1: 1.0}
    assert scores.shape == (40,)


def test_xgboost_uses_declared_grid_and_fit_only_scale_pos_weight() -> None:
    data, context = _dataset_and_context()
    adapter = XGBoostAdapter(n_estimators=10)
    fitted = adapter.fit(data, context)
    scores = adapter.predict_scores(fitted, data)[0]
    assert fitted.hyperparameters["max_depth"] in {4, 8}
    assert fitted.hyperparameters["learning_rate"] in {0.05, 0.1}
    assert fitted.estimator.get_params()["scale_pos_weight"] == 1.0
    assert scores.shape == (40,)

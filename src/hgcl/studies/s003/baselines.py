"""Fair-comparison contracts and method registry for S003."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from xgboost import XGBClassifier

from .models import DirectedSupervisedGNN
from .models import DirectedGINEncoder, S003ContrastiveModel, symmetric_multi_positive_loss
from .augmentations import make_s003_views, make_two_stochastic_views, maybe_add_reverse_edges
from .positives import build_positive_sets, cosine_knn
from .training import RefitClassifier, refit_downstream, search_downstream


class BaselineContractError(ValueError):
    """Raised when a method would violate the common comparison protocol."""


METHOD_IDS = (
    "mlp_x",
    "random_forest",
    "xgboost",
    "gcn_supervised",
    "graphsage_supervised",
    "gin_supervised",
    "inspection_l_dgi",
    "gcpal",
    "s003_txgcl",
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


@dataclass(frozen=True)
class AdapterContext:
    """Immutable inputs shared by every comparable method adapter."""

    seed: int
    fraction: float
    fit_ids: frozenset[str]
    validation_ids: frozenset[str]
    refit_ids: frozenset[str]
    budget_digest: str
    target_node_type: str = "transaction"
    edge_index_order: str = "source_target"
    message_flow: str = "source_to_target"
    feature_count: int = 182
    selection_primary: str = "f1_illicit"
    selection_tiebreaker: str = "mcc"
    evaluation_policy: str = "pooled_and_per_snapshot"

    def __post_init__(self) -> None:
        if self.seed not in {11, 23, 37, 53, 71}:
            raise BaselineContractError("unsupported seed")
        if self.fraction not in {0.01, 0.05, 0.10, 1.00}:
            raise BaselineContractError("unsupported label fraction")
        if self.fit_ids & self.validation_ids:
            raise BaselineContractError("fit and validation IDs must be disjoint")
        if self.refit_ids != self.fit_ids | self.validation_ids:
            raise BaselineContractError("refit IDs must equal fit union validation")
        if not _is_sha256(self.budget_digest):
            raise BaselineContractError("budget_digest must be SHA-256")
        expected = {
            "target_node_type": "transaction",
            "edge_index_order": "source_target",
            "message_flow": "source_to_target",
            "feature_count": 182,
            "selection_primary": "f1_illicit",
            "selection_tiebreaker": "mcc",
            "evaluation_policy": "pooled_and_per_snapshot",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise BaselineContractError(f"shared adapter invariant mismatch: {name}")


@dataclass(frozen=True)
class FitPrevalence:
    """Class-balancing values computed exclusively from the fit subset."""

    counts: Mapping[int, int]
    neural_class_weights: Mapping[int, float]
    rf_class_weight: Mapping[int, float]
    xgb_scale_pos_weight: float

    @classmethod
    def from_fit_labels(cls, labels: Sequence[int]) -> "FitPrevalence":
        values = [int(label) for label in labels]
        if any(label not in {0, 1} for label in values):
            raise BaselineContractError("prevalence accepts only known binary fit labels")
        counts = {0: values.count(0), 1: values.count(1)}
        if not all(counts.values()):
            raise BaselineContractError("both classes are required in fit prevalence")
        total = len(values)
        weights = {label: total / (2 * count) for label, count in counts.items()}
        return cls(
            counts=counts,
            neural_class_weights=weights,
            rf_class_weight=dict(weights),
            xgb_scale_pos_weight=counts[0] / counts[1],
        )


@runtime_checkable
class MethodAdapter(Protocol):
    """Interface implemented by every trainable method in T046–T049."""

    method_id: str
    characteristic_components: frozenset[str]

    def fit(self, dataset: Any, context: AdapterContext) -> Any: ...

    def predict_scores(self, fitted: Any, snapshots: Any) -> Mapping[int, Any]: ...


@dataclass(frozen=True)
class MethodDefinition:
    method_id: str
    family: str
    required_components: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.method_id not in METHOD_IDS:
            raise BaselineContractError(f"unknown method: {self.method_id}")


@dataclass(frozen=True)
class FidelityAssessment:
    method_id: str
    status: str
    missing_components: tuple[str, ...]


class MethodRegistry:
    """Immutable scientific registry; adapter implementations attach later."""

    def __init__(self, definitions: Mapping[str, MethodDefinition]) -> None:
        self.definitions = dict(definitions)
        if any(key != definition.method_id for key, definition in self.definitions.items()):
            raise BaselineContractError("registry key must equal method_id")

    @property
    def method_ids(self) -> tuple[str, ...]:
        return tuple(self.definitions)

    def assert_complete(self) -> None:
        actual = set(self.definitions)
        expected = set(METHOD_IDS)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise BaselineContractError(f"registry mismatch: missing={missing}, extra={extra}")

    def definition(self, method_id: str) -> MethodDefinition:
        try:
            return self.definitions[method_id]
        except KeyError as error:
            raise BaselineContractError(f"method is not registered: {method_id}") from error

    def assess_fidelity(
        self, method_id: str, implemented_components: set[str] | frozenset[str]
    ) -> FidelityAssessment:
        definition = self.definition(method_id)
        missing = tuple(sorted(definition.required_components - set(implemented_components)))
        return FidelityAssessment(
            method_id=method_id,
            status="approximation" if missing else "exact",
            missing_components=missing,
        )


def default_method_registry() -> MethodRegistry:
    definitions = {
        "mlp_x": MethodDefinition("mlp_x", "tabular_neural", frozenset({"x_only", "mlp"})),
        "random_forest": MethodDefinition("random_forest", "tabular_tree", frozenset({"random_forest_300"})),
        "xgboost": MethodDefinition("xgboost", "tabular_boosting", frozenset({"xgboost_500_max"})),
        "gcn_supervised": MethodDefinition("gcn_supervised", "supervised_gnn", frozenset({"gcn_2x128"})),
        "graphsage_supervised": MethodDefinition("graphsage_supervised", "supervised_gnn", frozenset({"graphsage_2x128"})),
        "gin_supervised": MethodDefinition("gin_supervised", "supervised_gnn", frozenset({"gin_2x128"})),
        "inspection_l_dgi": MethodDefinition(
            "inspection_l_dgi",
            "self_supervised_baseline",
            frozenset({"gin_2x128", "dgi", "random_forest_100"}),
        ),
        "gcpal": MethodDefinition(
            "gcpal",
            "self_supervised_baseline",
            frozenset(
                {
                    "gin_2x128",
                    "two_stochastic_views",
                    "knn_k10",
                    "multi_positive_loss",
                    "mlp_2layer_h_concat_x",
                }
            ),
        ),
        "s003_txgcl": MethodDefinition(
            "s003_txgcl",
            "proposed",
            frozenset({"gin_2x128", "stochastic_view", "functional_block_view", "knn_k10", "multi_positive_loss"}),
        ),
    }
    registry = MethodRegistry(definitions)
    registry.assert_complete()
    return registry


@dataclass(frozen=True)
class TabularData:
    tx_ids: tuple[str, ...]
    features: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != 182:
            raise BaselineContractError("tabular adapters require 182 features")
        if len(self.tx_ids) != self.features.shape[0] or self.labels.shape != (len(self.tx_ids),):
            raise BaselineContractError("tabular data shapes do not align")
        if len(set(self.tx_ids)) != len(self.tx_ids):
            raise BaselineContractError("tabular transaction IDs must be unique")


@dataclass(frozen=True)
class FittedTabularModel:
    method_id: str
    estimator: Any
    threshold: float
    validation_f1_illicit: float
    validation_mcc: float
    hyperparameters: Mapping[str, float | int]
    prevalence: FitPrevalence


def _indices(data: TabularData, identifiers: frozenset[str]) -> np.ndarray:
    positions = {tx_id: index for index, tx_id in enumerate(data.tx_ids)}
    missing = identifiers - positions.keys()
    if missing:
        raise BaselineContractError(f"budget IDs absent from dataset: {sorted(missing)[:3]}")
    return np.asarray(sorted(positions[tx_id] for tx_id in identifiers), dtype=np.int64)


def _selection(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    candidates = []
    for threshold in (0.3, 0.5, 0.7):
        predicted = (scores >= threshold).astype(np.int64)
        candidates.append(
            (
                float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
                float(matthews_corrcoef(labels, predicted)),
                threshold,
            )
        )
    f1, mcc, threshold = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
    return threshold, f1, mcc


def _fit_arrays(
    data: TabularData, context: AdapterContext
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, FitPrevalence]:
    fit = _indices(data, context.fit_ids)
    validation = _indices(data, context.validation_ids)
    refit = _indices(data, context.refit_ids)
    labels = data.labels.detach().cpu().numpy().astype(np.int64)
    if np.any(~np.isin(labels[refit], (0, 1))):
        raise BaselineContractError("supervised adapters accept only known labels in their budget")
    prevalence = FitPrevalence.from_fit_labels(labels[fit].tolist())
    features = data.features.detach().cpu().numpy()
    return features, labels, fit, validation, refit, prevalence


class MLPXAdapter:
    method_id = "mlp_x"
    characteristic_components = frozenset({"x_only", "mlp"})

    def __init__(self, *, epochs: int = 100, patience: int = 10) -> None:
        self.epochs = epochs
        self.patience = patience

    def fit(self, dataset: TabularData, context: AdapterContext) -> FittedTabularModel:
        _, labels, fit, validation, refit, prevalence = _fit_arrays(dataset, context)
        empty_embeddings = torch.empty((len(dataset.tx_ids), 0), dtype=dataset.features.dtype)
        selection = search_downstream(
            empty_embeddings,
            dataset.features,
            dataset.labels,
            fit_indices=torch.as_tensor(fit),
            validation_indices=torch.as_tensor(validation),
            epochs=self.epochs,
            patience=self.patience,
            seed=context.seed,
        )
        fitted = refit_downstream(
            selection,
            empty_embeddings,
            dataset.features,
            dataset.labels,
            refit_indices=torch.as_tensor(refit),
            epochs=self.epochs,
            seed=context.seed,
        )
        return FittedTabularModel(
            self.method_id,
            fitted,
            selection.threshold,
            selection.f1_illicit,
            selection.mcc,
            selection.hyperparameters,
            prevalence,
        )

    def predict_scores(self, fitted: FittedTabularModel, snapshots: TabularData) -> Mapping[int, Any]:
        classifier: RefitClassifier = fitted.estimator
        classifier.model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(classifier.model(snapshots.features)).cpu().numpy()
        return {0: scores}


class RandomForestAdapter:
    method_id = "random_forest"
    characteristic_components = frozenset({"random_forest_300"})

    def __init__(self, *, n_estimators: int = 300) -> None:
        self.n_estimators = n_estimators

    def fit(self, dataset: TabularData, context: AdapterContext) -> FittedTabularModel:
        features, labels, fit, validation, refit, prevalence = _fit_arrays(dataset, context)
        candidates = []
        for min_samples_leaf in (1, 5):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                min_samples_leaf=min_samples_leaf,
                class_weight=dict(prevalence.rf_class_weight),
                random_state=context.seed,
                n_jobs=1,
            )
            model.fit(features[fit], labels[fit])
            threshold, f1, mcc = _selection(model.predict_proba(features[validation])[:, 1], labels[validation])
            candidates.append((f1, mcc, -min_samples_leaf, threshold, min_samples_leaf))
        best = max(candidates, key=lambda item: item[:3])
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=best[4],
            class_weight=dict(prevalence.rf_class_weight),
            random_state=context.seed,
            n_jobs=1,
        )
        model.fit(features[refit], labels[refit])
        return FittedTabularModel(
            self.method_id,
            model,
            best[3],
            best[0],
            best[1],
            {"n_estimators": self.n_estimators, "min_samples_leaf": best[4]},
            prevalence,
        )

    def predict_scores(self, fitted: FittedTabularModel, snapshots: TabularData) -> Mapping[int, Any]:
        features = snapshots.features.detach().cpu().numpy()
        return {0: fitted.estimator.predict_proba(features)[:, 1]}


class XGBoostAdapter:
    method_id = "xgboost"
    characteristic_components = frozenset({"xgboost_500_max"})

    def __init__(self, *, n_estimators: int = 500) -> None:
        self.n_estimators = n_estimators

    def _model(
        self, context: AdapterContext, prevalence: FitPrevalence, *, max_depth: int, learning_rate: float
    ) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=prevalence.xgb_scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=context.seed,
            n_jobs=1,
        )

    def fit(self, dataset: TabularData, context: AdapterContext) -> FittedTabularModel:
        features, labels, fit, validation, refit, prevalence = _fit_arrays(dataset, context)
        candidates = []
        for max_depth in (4, 8):
            for learning_rate in (0.05, 0.1):
                model = self._model(
                    context, prevalence, max_depth=max_depth, learning_rate=learning_rate
                )
                model.fit(features[fit], labels[fit])
                threshold, f1, mcc = _selection(
                    model.predict_proba(features[validation])[:, 1], labels[validation]
                )
                candidates.append((f1, mcc, -max_depth, -learning_rate, threshold, max_depth, learning_rate))
        best = max(candidates, key=lambda item: item[:4])
        model = self._model(context, prevalence, max_depth=best[5], learning_rate=best[6])
        model.fit(features[refit], labels[refit])
        return FittedTabularModel(
            self.method_id,
            model,
            best[4],
            best[0],
            best[1],
            {"n_estimators": self.n_estimators, "max_depth": best[5], "learning_rate": best[6]},
            prevalence,
        )

    def predict_scores(self, fitted: FittedTabularModel, snapshots: TabularData) -> Mapping[int, Any]:
        features = snapshots.features.detach().cpu().numpy()
        return {0: fitted.estimator.predict_proba(features)[:, 1]}


@dataclass(frozen=True)
class GraphData(TabularData):
    edge_index: torch.Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise BaselineContractError("edge_index must have shape [2,E]")
        if self.edge_index.numel() and (
            int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= len(self.tx_ids)
        ):
            raise BaselineContractError("edge_index references an unknown node")


class SupervisedGNNAdapter:
    """Shared supervised training policy for directed GCN, GraphSAGE, and GIN."""

    def __init__(self, kind: str, *, epochs: int = 100, learning_rate: float = 1e-3) -> None:
        names = {"gcn": "gcn_supervised", "graphsage": "graphsage_supervised", "gin": "gin_supervised"}
        if kind not in names:
            raise BaselineContractError(f"unsupported GNN adapter: {kind}")
        self.kind = kind
        self.method_id = names[kind]
        self.characteristic_components = frozenset({f"{kind}_2x128"})
        self.epochs = epochs
        self.learning_rate = learning_rate

    def _train(
        self, data: GraphData, indices: np.ndarray, *, seed: int, pos_weight: float
    ) -> DirectedSupervisedGNN:
        torch.manual_seed(seed)
        model = DirectedSupervisedGNN(self.kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        tensor_indices = torch.as_tensor(indices)
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data.features, data.edge_index)[tensor_indices], data.labels[tensor_indices].float())
            loss.backward()
            optimizer.step()
        return model

    def fit(self, dataset: GraphData, context: AdapterContext) -> FittedTabularModel:
        _, labels, fit, validation, refit, prevalence = _fit_arrays(dataset, context)
        fit_model = self._train(
            dataset, fit, seed=context.seed, pos_weight=prevalence.xgb_scale_pos_weight
        )
        fit_model.eval()
        with torch.no_grad():
            validation_scores = torch.sigmoid(fit_model(dataset.features, dataset.edge_index))[validation].numpy()
        threshold, f1, mcc = _selection(validation_scores, labels[validation])
        model = self._train(
            dataset, refit, seed=context.seed, pos_weight=prevalence.xgb_scale_pos_weight
        )
        return FittedTabularModel(
            self.method_id,
            model,
            threshold,
            f1,
            mcc,
            {"layers": 2, "hidden": 128, "epochs": self.epochs, "learning_rate": self.learning_rate},
            prevalence,
        )

    def predict_scores(self, fitted: FittedTabularModel, snapshots: GraphData) -> Mapping[int, Any]:
        fitted.estimator.eval()
        with torch.no_grad():
            scores = torch.sigmoid(fitted.estimator(snapshots.features, snapshots.edge_index)).numpy()
        return {0: scores}


@dataclass(frozen=True)
class FittedRepresentationModel:
    method_id: str
    encoder: Any
    downstream: FittedTabularModel
    migration_notes: tuple[str, ...]
    fidelity_status: str = "exact"


class InspectionLDGIAdapter:
    method_id = "inspection_l_dgi"
    characteristic_components = frozenset({"gin_2x128", "dgi", "random_forest_100"})

    def __init__(self, *, ssl_epochs: int = 100, rf_estimators: int = 100) -> None:
        self.ssl_epochs = ssl_epochs
        self.rf_estimators = rf_estimators

    def fit(self, dataset: GraphData, context: AdapterContext) -> FittedRepresentationModel:
        torch.manual_seed(context.seed)
        encoder = DirectedGINEncoder(182)
        discriminator = torch.nn.Bilinear(128, 128, 1)
        optimizer = torch.optim.Adam([*encoder.parameters(), *discriminator.parameters()], lr=1e-3)
        for _ in range(self.ssl_epochs):
            optimizer.zero_grad(set_to_none=True)
            positive = encoder(dataset.features, dataset.edge_index)
            permutation = torch.randperm(len(dataset.tx_ids))
            negative = encoder(dataset.features[permutation], dataset.edge_index)
            summary = torch.sigmoid(positive.mean(dim=0)).expand_as(positive)
            positive_logits = discriminator(positive, summary).squeeze(-1)
            negative_logits = discriminator(negative, summary).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
            loss += torch.nn.functional.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
            loss.backward()
            optimizer.step()
        encoder.eval()
        with torch.no_grad():
            embeddings = encoder(dataset.features, dataset.edge_index)
        embedded = TabularData(dataset.tx_ids, torch.cat((embeddings, torch.zeros(len(dataset.tx_ids), 54)), dim=1), dataset.labels)
        downstream = RandomForestAdapter(n_estimators=self.rf_estimators).fit(embedded, context)
        downstream = FittedTabularModel(self.method_id, downstream.estimator, downstream.threshold, downstream.validation_f1_illicit, downstream.validation_mcc, downstream.hyperparameters, downstream.prevalence)
        return FittedRepresentationModel(self.method_id, encoder, downstream, ("Elliptic++ uses 182 features instead of Elliptic's 166",))


class GCPALAdapter:
    method_id = "gcpal"
    characteristic_components = frozenset({"gin_2x128", "two_stochastic_views", "knn_k10", "multi_positive_loss", "mlp_2layer_h_concat_x"})

    def __init__(self, *, ssl_epochs: int = 100, downstream_epochs: int = 100, knn_k: int = 10) -> None:
        self.ssl_epochs = ssl_epochs
        self.downstream_epochs = downstream_epochs
        self.knn_k = knn_k

    def fit(self, dataset: GraphData, context: AdapterContext) -> FittedRepresentationModel:
        torch.manual_seed(context.seed)
        model = S003ContrastiveModel(182)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        positives = build_positive_sets(len(dataset.tx_ids), dataset.edge_index, cosine_knn(dataset.features, dataset.tx_ids, k=self.knn_k))
        for epoch in range(self.ssl_epochs):
            pair = make_two_stochastic_views(dataset.features, dataset.edge_index, seed=context.seed + epoch)
            optimizer.zero_grad(set_to_none=True)
            left, right = pair.views
            z_left, z_right = model(left.x, left.edge_index, right.x, right.edge_index)
            loss = symmetric_multi_positive_loss(z_left, z_right, positives)
            loss.backward()
            optimizer.step()
        embeddings = model.frozen_embeddings(dataset.features, dataset.edge_index)
        _, _, fit, validation, refit, prevalence = _fit_arrays(dataset, context)
        selection = search_downstream(embeddings, dataset.features, dataset.labels, fit_indices=torch.as_tensor(fit), validation_indices=torch.as_tensor(validation), epochs=self.downstream_epochs, patience=10, seed=context.seed)
        classifier = refit_downstream(selection, embeddings, dataset.features, dataset.labels, refit_indices=torch.as_tensor(refit), epochs=self.downstream_epochs, seed=context.seed)
        downstream = FittedTabularModel(self.method_id, classifier, selection.threshold, selection.f1_illicit, selection.mcc, selection.hyperparameters, prevalence)
        return FittedRepresentationModel(self.method_id, model, downstream, ("Elliptic++ uses 182 features instead of Elliptic's 166",))


class S003TxGCLAdapter:
    method_id = "s003_txgcl"
    characteristic_components = frozenset({"gin_2x128", "stochastic_view", "functional_block_view", "knn_k10", "multi_positive_loss"})

    def __init__(self, *, ssl_epochs: int = 100, downstream_epochs: int = 100, representation: str = "h_concat_x", masking_policy: str = "functional_blocks", use_knn: bool = True, edge_dropout: bool = True, reverse_edges: bool = False, reverse_edges_approved: bool = False) -> None:
        if representation not in {"x_only", "h_only", "h_concat_x"}:
            raise BaselineContractError("unknown S003 representation")
        self.ssl_epochs, self.downstream_epochs = ssl_epochs, downstream_epochs
        self.representation, self.masking_policy = representation, masking_policy
        self.use_knn, self.edge_dropout = use_knn, edge_dropout
        self.reverse_edges, self.reverse_edges_approved = reverse_edges, reverse_edges_approved

    def fit(self, dataset: GraphData, context: AdapterContext) -> FittedRepresentationModel:
        edges = maybe_add_reverse_edges(dataset.edge_index, enabled=self.reverse_edges, approved=self.reverse_edges_approved)
        model = S003ContrastiveModel(182)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        knn = cosine_knn(dataset.features, dataset.tx_ids, k=10) if self.use_knn else tuple(() for _ in dataset.tx_ids)
        positives = build_positive_sets(len(dataset.tx_ids), edges, knn)
        for epoch in range(self.ssl_epochs):
            pair = make_s003_views(dataset.features, edges, seed=context.seed + epoch, masking_policy=self.masking_policy, edge_drop=0.1 if self.edge_dropout else 0.0)
            optimizer.zero_grad(set_to_none=True)
            left, right = pair.views
            z_left, z_right = model(left.x, left.edge_index, right.x, right.edge_index)
            loss = symmetric_multi_positive_loss(z_left, z_right, positives)
            loss.backward(); optimizer.step()
        embeddings = model.frozen_embeddings(dataset.features, edges)
        if self.representation == "x_only":
            h, x = torch.empty((len(dataset.tx_ids), 0)), dataset.features
        elif self.representation == "h_only":
            h, x = embeddings, torch.empty((len(dataset.tx_ids), 0))
        else:
            h, x = embeddings, dataset.features
        _, _, fit, validation, refit, prevalence = _fit_arrays(dataset, context)
        selection = search_downstream(h, x, dataset.labels, fit_indices=torch.as_tensor(fit), validation_indices=torch.as_tensor(validation), epochs=self.downstream_epochs, patience=10, seed=context.seed)
        classifier = refit_downstream(selection, h, x, dataset.labels, refit_indices=torch.as_tensor(refit), epochs=self.downstream_epochs, seed=context.seed)
        downstream = FittedTabularModel(self.method_id, classifier, selection.threshold, selection.f1_illicit, selection.mcc, {**selection.hyperparameters, "representation": self.representation, "masking_policy": self.masking_policy, "use_knn": int(self.use_knn), "edge_dropout": int(self.edge_dropout)}, prevalence)
        return FittedRepresentationModel(self.method_id, model, downstream, ())

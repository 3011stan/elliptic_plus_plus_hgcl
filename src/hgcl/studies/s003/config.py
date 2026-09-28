"""Strict S003 configuration contracts and canonicalization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an S003 configuration fails closed."""


TOP_LEVEL_KEYS = {
    "schema_version",
    "study_id",
    "task",
    "method_id",
    "graph_schema",
    "target_node_type",
    "profile",
    "paths",
    "data",
    "split",
    "labels",
    "features",
    "graph",
    "positives",
    "ssl",
    "downstream",
    "baselines",
    "evaluation",
    "statistics",
    "resources",
    "provenance",
    "matrix",
    "approval",
}

SECTION_KEYS = {
    "paths": {"data_root", "artifacts_root"},
    "data": {
        "required_files",
        "max_nodes_per_snapshot",
        "sampling",
        "sampling_salt",
    },
    "split": {
        "development_steps",
        "engineering_fit_steps",
        "shadow_test_steps",
        "test_steps",
    },
    "labels": {"seeds", "fractions", "fit_ratio"},
    "features": {
        "source_non_id_count",
        "model_count",
        "maskable_count",
        "functional_block_sizes",
    },
    "graph": {
        "directed",
        "edge_index_order",
        "message_flow",
        "add_reverse_edges",
    },
    "positives": {"structural_neighbors", "knn_k"},
    "ssl": {
        "encoder",
        "layers",
        "hidden",
        "loss",
        "use_labels",
        "epochs",
        "snapshot_batch_size",
    },
    "downstream": {"encoder_frozen", "representation", "epochs", "patience"},
    "baselines": {"methods", "include_s003_ablations"},
    "evaluation": {"selection", "cohort_release"},
    "statistics": {"test", "correction", "alpha", "effect_size"},
    "resources": {
        "device",
        "max_ram_gib",
        "max_vram_gib",
        "max_training_seconds",
        "projection_margin",
        "reapproval_threshold",
    },
    "provenance": {"require_clean_revision"},
    "matrix": {"expected_p1_cells"},
}

FORBIDDEN_S02_TOKENS = {"native_wallet", "fusion.alphas", "hgcl"}
ALLOWED_OVERRIDE_PREFIXES = ("paths.", "resources.device", "resources.max_")
DEV_STEPS = list(range(1, 35))
FIT_STEPS = list(range(1, 30))
SHADOW_STEPS = list(range(30, 35))
TEST_STEPS = list(range(35, 50))
ALL_SEEDS = [11, 23, 37, 53, 71]
ALL_FRACTIONS = [0.01, 0.05, 0.10, 1.00]
ALL_METHODS = [
    "mlp_x",
    "random_forest",
    "xgboost",
    "gcn_supervised",
    "graphsage_supervised",
    "gin_supervised",
    "inspection_l_dgi",
    "gcpal",
    "s003_txgcl",
]


@dataclass(frozen=True)
class S003Config:
    """Validated configuration plus stable canonical identity."""

    raw: dict[str, Any]
    digest: str
    source: Path

    @property
    def profile(self) -> str:
        return str(self.raw["profile"])

    @property
    def study_id(self) -> str:
        return str(self.raw["study_id"])

    @property
    def method_id(self) -> str:
        return str(self.raw["method_id"])


def canonical_json(raw: Mapping[str, Any]) -> str:
    """Serialize without YAML or insertion-order ambiguity."""
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_digest(raw: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()


def _fail(condition: bool, message: str) -> None:
    if condition:
        raise ConfigError(message)


def _scan_forbidden(value: Any, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _fail(str(key) in FORBIDDEN_S02_TOKENS, f"forbidden S02 key at {location}: {key}")
            _scan_forbidden(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden(child, f"{location}[{index}]")
    elif isinstance(value, str):
        _fail(value in FORBIDDEN_S02_TOKENS, f"forbidden S02 value at {location}: {value}")


def _validate_shape(raw: dict[str, Any]) -> None:
    missing = TOP_LEVEL_KEYS - raw.keys()
    unknown = raw.keys() - TOP_LEVEL_KEYS
    _fail(bool(missing), f"missing top-level keys: {sorted(missing)}")
    _fail(bool(unknown), f"unknown top-level keys: {sorted(unknown)}")
    for section, allowed in SECTION_KEYS.items():
        value = raw.get(section)
        _fail(not isinstance(value, dict), f"{section} must be a mapping")
        section_unknown = value.keys() - allowed
        _fail(bool(section_unknown), f"unknown keys in {section}: {sorted(section_unknown)}")


def _validate_common(raw: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "s003",
        "task": "transaction_classification",
        "method_id": "s003_txgcl",
        "graph_schema": "tx_tx",
        "target_node_type": "transaction",
    }
    for key, value in expected.items():
        _fail(raw[key] != value, f"{key} must be {value!r}")
    _fail(raw["profile"] not in {"smoke", "dry-run", "lab"}, "invalid profile")
    _fail(raw["split"]["development_steps"] != DEV_STEPS, "development_steps must be 1..34")
    _fail(raw["split"]["test_steps"] != TEST_STEPS, "test_steps must be 35..49")
    _fail(raw["labels"]["fit_ratio"] != 0.8, "labels.fit_ratio must be 0.8")
    _fail(raw["features"]["source_non_id_count"] != 183, "source_non_id_count must be 183")
    _fail(raw["features"]["model_count"] != 182, "model_count must be 182")
    _fail(raw["features"]["maskable_count"] != 182, "maskable_count must be 182")
    _fail(raw["features"]["functional_block_sizes"] != [93, 72, 17], "invalid blocks")
    _fail(raw["graph"]["directed"] is not True, "graph must be directed")
    _fail(raw["graph"]["edge_index_order"] != "source_target", "invalid edge order")
    _fail(raw["graph"]["message_flow"] != "source_to_target", "invalid message flow")
    _fail(raw["positives"]["structural_neighbors"] != "successors", "invalid structural positives")
    _fail(raw["positives"]["knn_k"] != 10, "positives.knn_k must be 10")
    _fail(raw["ssl"]["encoder"] != "gin", "ssl.encoder must be gin")
    _fail(raw["ssl"]["layers"] != 2 or raw["ssl"]["hidden"] != 128, "GIN must be 2x128")
    _fail(raw["ssl"]["loss"] != "gcpal_multi_positive", "invalid SSL loss")
    _fail(raw["ssl"]["use_labels"] is not False, "SSL must not use labels")
    _fail(raw["ssl"]["snapshot_batch_size"] != 1, "snapshot batch size must be one")
    _fail(raw["downstream"]["encoder_frozen"] is not True, "encoder must be frozen")
    _fail(raw["downstream"]["representation"] != "h_concat_x", "invalid representation")
    selection = raw["evaluation"].get("selection")
    _fail(
        selection != {"primary": "f1_illicit", "tiebreaker": "mcc"},
        "invalid selection policy",
    )
    _fail(raw["statistics"]["test"] != "paired_t_two_sided", "invalid test")
    _fail(raw["statistics"]["correction"] != "holm", "invalid correction")
    _fail(raw["statistics"]["alpha"] != 0.05, "statistics.alpha must be 0.05")
    _fail(raw["statistics"]["effect_size"] != "cohen_dz", "invalid effect size")
    _fail(raw["matrix"]["expected_p1_cells"] != 205, "expected_p1_cells must be 205")
    _fail(raw["paths"]["artifacts_root"] != "artifacts/s003", "invalid artifacts root")


def _validate_profile(raw: dict[str, Any]) -> None:
    profile = raw["profile"]
    if profile in {"smoke", "dry-run"}:
        _fail(raw["split"]["engineering_fit_steps"] != FIT_STEPS, "invalid engineering fit steps")
        _fail(raw["split"]["shadow_test_steps"] != SHADOW_STEPS, "invalid shadow steps")
        _fail(raw["approval"] is not None, "engineering profiles cannot carry approval")
        _fail(raw["evaluation"]["cohort_release"] != "shadow_only", "invalid shadow release")
    if profile == "smoke":
        _fail(raw["data"]["max_nodes_per_snapshot"] != 256, "smoke cap must be 256")
        _fail(raw["data"].get("sampling") != "sha256_tx_id", "invalid smoke sampling")
        _fail(raw["data"].get("sampling_salt") != "s003-smoke", "invalid smoke salt")
        _fail(raw["labels"]["seeds"] != [11], "smoke uses engineering seed 11")
        _fail(raw["labels"]["fractions"] != [0.01], "smoke uses fraction 1%")
        _fail(raw["ssl"]["epochs"] != 2, "smoke ssl.epochs must be 2")
        _fail(raw["downstream"]["epochs"] != 3, "smoke downstream epochs must be 3")
    elif profile == "dry-run":
        _fail(raw["data"]["max_nodes_per_snapshot"] is not None, "dry-run uses all nodes")
        _fail(raw["labels"]["seeds"] != [11], "dry-run seed must be 11")
        _fail(raw["labels"]["fractions"] != [0.01], "dry-run fraction must be 1%")
        _fail(raw["ssl"]["epochs"] != 10, "dry-run ssl.epochs must be 10")
        _fail(raw["downstream"]["epochs"] != 20, "dry-run downstream epochs must be 20")
        _fail(raw["downstream"]["patience"] != 5, "dry-run patience must be 5")
        _fail(raw["baselines"]["methods"] != ALL_METHODS, "dry-run must represent nine methods")
        _fail(raw["baselines"].get("include_s003_ablations") is not True, "dry-run needs ablations")
    else:
        _fail(not isinstance(raw["ssl"]["epochs"], int) or raw["ssl"]["epochs"] <= 0, "lab ssl.epochs must be a fixed positive integer")
        _fail(raw["labels"]["seeds"] != ALL_SEEDS, "lab seeds mismatch")
        _fail(raw["labels"]["fractions"] != ALL_FRACTIONS, "lab fractions mismatch")
        _fail(raw["split"]["engineering_fit_steps"] is not None, "lab cannot use engineering fit")
        _fail(raw["split"]["shadow_test_steps"] is not None, "lab cannot use shadow test")
        _fail(raw["evaluation"]["cohort_release"] != "single_global", "lab needs global release")
        approval = raw["approval"]
        _fail(not isinstance(approval, dict), "lab approval must be configured")
        _fail(
            set(approval) != {"approval_id", "artifact_path"},
            "lab approval requires approval_id and artifact_path",
        )
        _fail(not str(approval["approval_id"]).startswith("s003-"), "invalid approval_id")


def _apply_overrides(raw: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for dotted, value in overrides.items():
        _fail(
            not any(dotted == prefix or dotted.startswith(prefix) for prefix in ALLOWED_OVERRIDE_PREFIXES),
            f"override not allowed: {dotted}",
        )
        parts = dotted.split(".")
        target: dict[str, Any] = raw
        for part in parts[:-1]:
            _fail(part not in target or not isinstance(target[part], dict), f"invalid override: {dotted}")
            target = target[part]
        _fail(parts[-1] not in target, f"invalid override: {dotted}")
        target[parts[-1]] = value


def load_config(
    path: str | Path, *, overrides: Mapping[str, Any] | None = None
) -> S003Config:
    """Load a strict YAML configuration and compute its stable digest."""
    source = Path(path).resolve()
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load config: {source}") from exc
    _fail(not isinstance(loaded, dict), "config root must be a mapping")
    raw = deepcopy(loaded)
    if overrides:
        _apply_overrides(raw, overrides)
    _scan_forbidden(raw)
    _validate_shape(raw)
    _validate_common(raw)
    _validate_profile(raw)
    return S003Config(raw=raw, digest=config_digest(raw), source=source)


def validate_approval_compatibility(
    config: S003Config,
    approval: Mapping[str, Any],
    *,
    data_digest: str,
    code_revision: str,
    design_digest: str,
) -> None:
    """Reject approval evidence that is not bound to the requested lab run."""
    expected = {
        "lab_config_digest": config.digest,
        "data_digest": data_digest,
        "code_revision": code_revision,
        "design_digest": design_digest,
    }
    for key, value in expected.items():
        _fail(approval.get(key) != value, f"approval {key} mismatch")

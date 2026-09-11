"""Versioned, strict experiment configuration, independent of working directory."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import yaml


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def defaults(profile: str) -> dict:
    if profile not in ("smoke", "smoke-gpu", "lab"):
        raise ValueError("profile must be smoke, smoke-gpu or lab")
    smoke = profile != "lab"
    gpu = profile != "smoke"
    return {
        "schema": 1, "profile": profile,
        "paths": {"project_root": "..", "data_root": "elliptic-plus-plus/raw", "artifacts_root": "artifacts"},
        "split": {"train": [1, 2] if smoke else list(range(1, 29)), "validation": [29] if smoke else list(range(29, 35)), "test": [35] if smoke else list(range(35, 50))},
        "labels": {"seeds": [11] if smoke else [11, 23, 37, 53, 71], "fractions": [1.0] if smoke else [.01, .05, .1, 1.0]},
        "features": {"regimes": ["principal"] if smoke else ["principal", "native_wallet"]},
        "encoder": {"hidden": 32 if smoke else 128, "layers": 2},
        "ssl": {"epochs": 2 if smoke else 100, "lr": .001, "weight_decay": .00001, "temperature": .2, "projection_dim": 32 if smoke else 128, "feature_mask": .1, "edge_dropout": .1},
        "supervised": {"epochs": 3 if smoke else 100, "patience": 3 if smoke else 10, "lrs": [.001] if smoke else [.001, .0003], "weight_decay": .00001, "optimizer": "Adam"},
        "rf": {"trees": 50 if smoke else 300, "min_leaf": [1] if smoke else [1, 5], "max_features": "sqrt", "class_weight": "balanced", "jobs": 4},
        "fusion": {"alphas": [i / 20 for i in range(21)]},
        "evaluation": {"primary": "f1_illicit", "methods": ["rf", "graph_supervised", "hgcl", "fusion"]},
        "resources": {"device": "cuda" if gpu else "cpu", "recut_transactions": 256 if smoke else None, "anchor_batch": 32 if smoke else 128, "workers": 0, "loading": "neighbor" if gpu else "full", "fanouts": [10, 5], "timeout_seconds": 600 if smoke else None, "rss_gib": 24 if gpu else 8, "gpu_gib": 4.5 if gpu else None, "max_nodes": 50000, "max_edges": 200000, "inference_nodes": 4096, "inference_edges": 100000},
        "provenance": {"schema": 1, "source_manifest": "docs/data/source-manifest.json"},
    }


def _shape(value, reference, path="config"):
    if isinstance(reference, dict):
        if not isinstance(value, dict) or set(value) != set(reference):
            raise ValueError(f"{path}: missing or unknown keys")
        for key in reference:
            _shape(value[key], reference[key], f"{path}.{key}")
    elif isinstance(reference, list):
        if not isinstance(value, list) or len(value) != len(reference):
            raise ValueError(f"{path}: invalid list")
        for i, item in enumerate(reference):
            _shape(value[i], item, f"{path}[{i}]")
    elif type(value) is not type(reference):
        raise ValueError(f"{path}: expected {type(reference).__name__}")


@dataclass(frozen=True)
class Config:
    canonical_json: str

    @property
    def values(self) -> dict:
        return json.loads(self.canonical_json)

    @property
    def hash(self) -> str:
        return digest(self.values)

    @property
    def root(self) -> Path:
        return Path(self.values["paths"]["project_root"])

    @property
    def data_root(self) -> Path:
        return Path(self.values["paths"]["data_root"])

    @property
    def artifacts_root(self) -> Path:
        return Path(self.values["paths"]["artifacts_root"])


def load_config(path: Path, *, data_root=None, artifacts_root=None, device=None) -> Config:
    path = path.resolve()
    try:
        value = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    reference = defaults(value.get("profile"))
    _shape(value, reference)
    for section in reference:
        if section != "paths" and value[section] != reference[section]:
            raise ValueError(f"{section}: differs from accepted {value['profile']} protocol; requires explicit revision")
    root = (path.parent / value["paths"]["project_root"]).resolve()
    paths = value["paths"]
    paths["project_root"] = str(root)
    for key, override, env in (("data_root", data_root, "ELLIPTIC_DATA_ROOT"), ("artifacts_root", artifacts_root, "HGCL_ARTIFACTS_ROOT")):
        chosen = override if override is not None else os.environ.get(env, paths[key])
        paths[key] = str((root / chosen).resolve())
    if device is not None:
        if device not in ("cpu", "cuda") or (value["profile"] == "smoke" and device != "cpu"):
            raise ValueError("Unsupported device override")
        value["resources"]["device"] = device
    data, artifacts = Path(paths["data_root"]), Path(paths["artifacts_root"])
    if artifacts == data or data in artifacts.parents or artifacts in data.parents:
        raise ValueError("Data and artifacts roots must not overlap")
    return Config(json.dumps(value, sort_keys=True, allow_nan=False))

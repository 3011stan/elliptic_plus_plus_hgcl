from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[3]
S003_SOURCE = ROOT / "src" / "hgcl" / "studies" / "s003"
S003_CONFIG = ROOT / "configs" / "s003"


def test_s003_does_not_import_historical_implementation() -> None:
    forbidden = {
        "hgcl.cli",
        "hgcl.config",
        "hgcl.data",
        "hgcl.environment",
        "hgcl.evaluation",
        "hgcl.matrix",
        "hgcl.models",
        "hgcl.pipeline",
        "hgcl.provenance",
        "hgcl.training",
    }
    for path in S003_SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            name == blocked or name.startswith(f"{blocked}.")
            for name in imports
            for blocked in forbidden
        ), f"historical import in {path}: {imports}"


def test_s003_configs_reject_historical_keys_and_roots() -> None:
    forbidden_keys = {"native_wallet", "fusion.alphas"}
    for path in S003_CONFIG.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        assert not forbidden_keys.intersection(text)
        assert raw["study_id"] == "s003"
        assert raw["paths"]["artifacts_root"] == "artifacts/s003"
        assert raw["method_id"] != "hgcl"


def test_frozen_s02_paths_are_unchanged_from_closing_tag() -> None:
    frozen = [
        "specs/001-hgcl-experiment",
        "configs/lab.yaml",
        "docs/validation",
    ]
    result = subprocess.run(
        ["git", "diff", "--name-only", "s02-001-final", "--", *frozen],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""

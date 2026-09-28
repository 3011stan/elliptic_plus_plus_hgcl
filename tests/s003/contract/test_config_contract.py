from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from hgcl.studies.s003.config import (
    ConfigError,
    load_config,
    validate_approval_compatibility,
)


CONFIG_ROOT = Path(__file__).parents[3] / "configs" / "s003"


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _raw(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["smoke.yaml", "dry-run.yaml"])
def test_engineering_profiles_are_strict_and_canonical(name: str) -> None:
    config = load_config(CONFIG_ROOT / name)
    assert config.study_id == "s003"
    assert config.method_id == "s003_txgcl"
    assert len(config.digest) == 64
    assert config.raw["features"]["model_count"] == 182
    assert config.raw["graph"]["message_flow"] == "source_to_target"
    assert config.raw["positives"]["structural_neighbors"] == "successors"
    assert config.raw["matrix"]["expected_p1_cells"] == 205


def test_exact_smoke_and_dry_run_limits() -> None:
    smoke = load_config(CONFIG_ROOT / "smoke.yaml").raw
    assert smoke["data"]["max_nodes_per_snapshot"] == 256
    assert smoke["ssl"]["epochs"] == 2
    assert smoke["downstream"]["epochs"] == 3
    assert smoke["ssl"]["snapshot_batch_size"] == 1

    dry = load_config(CONFIG_ROOT / "dry-run.yaml").raw
    assert dry["data"]["max_nodes_per_snapshot"] is None
    assert dry["labels"] == {"seeds": [11, 23, 37, 53, 71], "fractions": [0.01, 0.05, 0.10, 1.00], "fit_ratio": 0.8}
    assert dry["ssl"]["epochs"] == 0
    assert dry["downstream"]["epochs"] == 0
    assert dry["downstream"]["patience"] == 0
    assert dry["split"]["engineering_fit_steps"] is None
    assert dry["split"]["shadow_test_steps"] is None
    assert dry["evaluation"]["cohort_release"] == "none"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("features", "model_count"), 183),
        (("graph", "message_flow"), "target_to_source"),
        (("positives", "structural_neighbors"), "both"),
        (("ssl", "use_labels"), True),
        (("matrix", "expected_p1_cells"), 204),
    ],
)
def test_scientific_invariant_mutations_fail(
    tmp_path: Path, path: tuple[str, str], value: object
) -> None:
    raw = _raw("smoke.yaml")
    raw[path[0]][path[1]] = value
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, raw))


def test_unknown_and_s02_keys_fail(tmp_path: Path) -> None:
    raw = _raw("smoke.yaml")
    raw["unexpected"] = True
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, raw))

    for forbidden in ("native_wallet", "fusion.alphas", "hgcl"):
        candidate = deepcopy(_raw("smoke.yaml"))
        candidate["baselines"][forbidden] = True
        with pytest.raises(ConfigError):
            load_config(_write_config(tmp_path, candidate))


def test_lab_template_is_fail_closed_until_frozen(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="approval"):
        load_config(CONFIG_ROOT / "lab.template.yaml")

    raw = _raw("lab.template.yaml")
    assert raw["ssl"]["epochs"] == 100
    raw["approval"] = {
        "approval_id": "s003-approval-001",
        "artifact_path": "matrices/s003-matrix-001/approval.json",
    }
    config = load_config(_write_config(tmp_path, raw))
    approval = {
        "lab_config_digest": config.digest,
        "data_digest": "a" * 64,
        "code_revision": "b" * 40,
        "design_digest": "c" * 64,
    }
    validate_approval_compatibility(
        config,
        approval,
        data_digest="a" * 64,
        code_revision="b" * 40,
        design_digest="c" * 64,
    )
    approval["design_digest"] = "d" * 64
    with pytest.raises(ConfigError, match="design_digest"):
        validate_approval_compatibility(
            config,
            approval,
            data_digest="a" * 64,
            code_revision="b" * 40,
            design_digest="c" * 64,
        )


def test_lab_rejects_any_ssl_epoch_count_other_than_100(tmp_path: Path) -> None:
    raw = _raw("lab.template.yaml")
    raw["ssl"]["epochs"] = 99
    raw["approval"] = {"approval_id": "s003-approval-001", "artifact_path": "approval.json"}
    with pytest.raises(ConfigError, match="exactly 100"):
        load_config(_write_config(tmp_path, raw))

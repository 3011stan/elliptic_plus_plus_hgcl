from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml

from hgcl.studies.s003.cli import run
import hgcl.studies.s003.cli as cli_module
from hgcl.studies.s003.hetero import (
    check_causality_gate,
    check_comparability_gate,
    check_resources_gate,
    check_supervision_gate,
    evaluate_heterogeneous_gate,
)


def test_causality_gate_evaluations() -> None:
    """Test causality gate on missing files and non-causal global features (FR-030, FR-031)."""
    # 1. Missing address files
    res_missing = check_causality_gate(address_files_present=False)
    assert res_missing.passed is False
    assert "missing" in res_missing.reason

    # 2. Present address files but global lifetime aggregations
    res_global = check_causality_gate(
        address_files_present=True,
        features_are_causal_per_snapshot=False,
    )
    assert res_global.passed is False
    assert "global lifetime aggregations" in res_global.reason

    # 3. Present and causal
    res_causal = check_causality_gate(
        address_files_present=True,
        features_are_causal_per_snapshot=True,
    )
    assert res_causal.passed is True


def test_supervision_gate_evaluations() -> None:
    """Test supervision gate ensuring target remains tx classification (FR-031)."""
    # 1. Target altered to wallet classification
    res_wallet_target = check_supervision_gate(target_node_type="wallet")
    assert res_wallet_target.passed is False
    assert "target node type must be 'tx'" in res_wallet_target.reason

    # 2. Wallet labels used to supervise
    res_wallet_loss = check_supervision_gate(target_node_type="tx", wallet_labels_used=True)
    assert res_wallet_loss.passed is False
    assert "wallet labels must not be used" in res_wallet_loss.reason

    # 3. Valid tx target without wallet labels
    res_valid = check_supervision_gate(target_node_type="tx", wallet_labels_used=False)
    assert res_valid.passed is True


def test_comparability_gate_evaluations() -> None:
    """Test comparability gate enforcing identical splits, budgets, and seeds (FR-022, FR-030)."""
    # 1. Modified test steps
    res_split = check_comparability_gate(test_steps=tuple(range(30, 50)))
    assert res_split.passed is False
    assert "temporal split steps do not match" in res_split.reason

    # 2. Modified budgets
    res_budgets = check_comparability_gate(fractions=(0.02, 0.05, 0.1, 1.0))
    assert res_budgets.passed is False
    assert "label fractions do not match" in res_budgets.reason

    # 3. Modified seeds
    res_seeds = check_comparability_gate(seeds=(1, 2, 3, 4, 5))
    assert res_seeds.passed is False
    assert "seeds do not match" in res_seeds.reason

    # 4. Canonical matching parameters
    res_canonical = check_comparability_gate()
    assert res_canonical.passed is True


def test_resources_gate_evaluations() -> None:
    """Test resources gate projecting memory and duration limits (FR-030)."""
    # 1. Projected RAM exceeds allocation
    res_ram = check_resources_gate(
        projected_ram_gib=64.0,
        max_ram_gib=16.0,
        projected_duration_seconds=3600.0,
        max_duration_seconds=36000.0,
    )
    assert res_ram.passed is False
    assert "projected RAM" in res_ram.reason

    # 2. Projected duration exceeds allocation
    res_dur = check_resources_gate(
        projected_ram_gib=8.0,
        max_ram_gib=16.0,
        projected_duration_seconds=50000.0,
        max_duration_seconds=36000.0,
    )
    assert res_dur.passed is False
    assert "projected duration" in res_dur.reason

    # 3. Within resource limits
    res_ok = check_resources_gate(
        projected_ram_gib=8.0,
        max_ram_gib=16.0,
        projected_duration_seconds=12000.0,
        max_duration_seconds=36000.0,
    )
    assert res_ok.passed is True


def test_evaluate_heterogeneous_gate_nonblocking_and_persistence(tmp_path: Path) -> None:
    """Test non-blocking evaluation of all gates and decision persistence (FR-030–FR-032, SC-008)."""
    config_dict = {
        "study_id": "s003",
        "target_node_type": "tx",
        "task": "tx_binary_classification",
        "split": {
            "development_steps": list(range(1, 35)),
            "test_steps": list(range(35, 50)),
        },
        "labels": {
            "fractions": [0.01, 0.05, 0.1, 1.0],
            "seeds": [11, 23, 37, 53, 71],
        },
        "resources": {
            "max_ram_gib": 16.0,
            "max_duration_seconds": 36000.0,
        },
        "paths": {
            "data_root": "/fake/data",
            "artifacts_root": str(tmp_path / "artifacts"),
        },
    }

    # Case 1: Non-causal features -> Gate produces defer without raising exception
    output_path = tmp_path / "hetero_decision.json"
    decision_defer = evaluate_heterogeneous_gate(
        config_dict,
        address_files_present=True,
        features_are_causal_per_snapshot=False,  # fails causality
        projected_ram_gib=8.0,
        projected_duration_seconds=1000.0,
        output_path=output_path,
    )
    assert decision_defer.is_deferred is True
    assert decision_defer.decision == "defer"
    assert decision_defer.causality_gate.passed is False
    assert decision_defer.supervision_gate.passed is True
    assert decision_defer.comparability_gate.passed is True
    assert decision_defer.resources_gate.passed is True
    assert output_path.is_file()

    saved_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_data["decision"] == "defer"
    assert saved_data["gates"]["causality"]["passed"] is False

    # Case 2: All 4 gates pass and approval given -> Decision is include
    decision_include = evaluate_heterogeneous_gate(
        config_dict,
        address_files_present=True,
        features_are_causal_per_snapshot=True,
        projected_ram_gib=8.0,
        projected_duration_seconds=1000.0,
        approved_by="researcher_lead",
    )
    assert decision_include.is_included is True
    assert decision_include.decision == "include"
    assert decision_include.approved_by == "researcher_lead"


def test_hetero_gate_cli_execution(tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test hgcl-s003 hetero-gate CLI command execution and JSON output."""
    source = Path("configs/s003/dry-run.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["paths"]["data_root"] = str(tmp_path / "data")
    raw["paths"]["artifacts_root"] = "artifacts/s003"
    cfg_path = tmp_path / "dry-run.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts" / "s003").mkdir(parents=True, exist_ok=True)

    prep_path = tmp_path / "prepared"
    prep_path.mkdir(parents=True, exist_ok=True)

    fake_prepared = SimpleNamespace(
        data_digest="a" * 64,
        source=SimpleNamespace(optional_files={}),
    )
    monkeypatch.setattr(cli_module, "load_prepared", lambda *_args, **_kwargs: fake_prepared)

    exit_code = run(["hetero-gate", "--config", str(cfg_path), "--prepared", str(prep_path)])
    assert exit_code == 0

    out = json.loads(capsys.readouterr().out)
    assert out["study_id"] == "s003"
    assert out["command"] == "hetero-gate"
    assert out["status"] == "deferred"
    assert out["decision"] == "defer"
    assert out["gates"]["causality"] is False  # address files missing in mock
    assert len(out["artifacts"]) == 1
    assert Path(out["artifacts"][0]).is_file()

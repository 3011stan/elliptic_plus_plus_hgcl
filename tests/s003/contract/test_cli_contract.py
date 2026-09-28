from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import hgcl.studies.s003.cli as cli_module

from hgcl.studies.s003.cli import CommandError, build_parser, run


COMMANDS = {
    "doctor",
    "prepare",
    "audit",
    "smoke",
    "dry-run",
    "approve-dry-run",
    "matrix",
    "resume",
    "evaluate",
    "report",
    "hetero-gate",
}


def test_parser_exposes_only_s003_commands_without_historical_defaults() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"  # noqa: SLF001
    )
    assert set(subparsers.choices) == COMMANDS
    assert "configs/lab.yaml" not in parser.format_help()
    assert "artifacts/" not in parser.format_help()


def test_success_is_json_and_requires_prefixed_identifier(capsys: pytest.CaptureFixture) -> None:
    def handler(args: object) -> dict:
        return {"status": "ready", "artifacts": []}

    code = run(
        ["smoke", "--config", "config.yaml", "--run-id", "s003-smoke-001"],
        handlers={"smoke": handler},
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["study_id"] == "s003"
    assert payload["command"] == "smoke"

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["smoke", "--config", "config.yaml", "--run-id", "legacy-run"]
        )


def test_structural_dry_run_requires_prepared_dataset() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["dry-run", "--config", "config.yaml", "--run-id", "s003-dry-001"]
        )


def test_structural_dry_run_writes_no_training_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path("configs/s003/dry-run.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    config_path = tmp_path / "dry-run.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    monkeypatch.chdir(tmp_path)
    fake_prepared = SimpleNamespace(data_digest="a" * 64)
    monkeypatch.setattr(cli_module, "load_prepared", lambda *_args, **_kwargs: fake_prepared)
    monkeypatch.setattr(
        cli_module,
        "run_structural_dry_run",
        lambda *_args, **_kwargs: {
            "study_id": "s003",
            "profile": "dry-run",
            "config_digest": "b" * 64,
            "data_digest": "a" * 64,
            "design_digest": "c" * 64,
            "cells": [f"cell-{index}" for index in range(205)],
            "expected_p1_cells": 205,
            "training_performed": False,
            "inference_performed": False,
            "test_labels_materialized": False,
        },
    )

    code = run(
        [
            "dry-run",
            "--config",
            str(config_path),
            "--prepared",
            str(prepared),
            "--run-id",
            "s003-dry-structural",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["training_performed"] is False
    assert payload["test_labels_materialized"] is False
    assert payload["expected_p1_cells"] == 205
    run_root = tmp_path / "artifacts" / "s003" / "runs" / "s003-dry-structural"
    assert (run_root / "run.json").is_file()
    assert (run_root / "design.json").is_file()
    assert not (run_root / "checkpoints").exists()
    assert not (run_root / "predictions").exists()
    assert not (run_root / "metrics").exists()


@pytest.mark.parametrize("exit_code", [2, 3, 4])
def test_structured_errors_use_declared_exit_codes(
    exit_code: int, capsys: pytest.CaptureFixture
) -> None:
    def handler(args: object) -> dict:
        raise CommandError("blocked", exit_code=exit_code, category="test")

    assert run(["doctor", "--config", "bad.yaml"], handlers={"doctor": handler}) == exit_code
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "study_id": "s003",
        "command": "doctor",
        "status": "error",
        "error": {"category": "test", "message": "blocked"},
    }


def test_approve_dry_run_is_injectable_and_fail_closed() -> None:
    with pytest.raises(CommandError, match="handler"):
        run(
            [
                "approve-dry-run",
                "--run",
                "artifacts/s003/runs/s003-dry-001",
                "--approval-file",
                "approval-input.json",
            ],
            handlers={},
            raise_errors=True,
        )

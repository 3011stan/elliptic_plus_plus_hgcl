from __future__ import annotations

import json

import pytest

from hgcl.studies.s003.cli import CommandError, build_parser, run


COMMANDS = {
    "doctor",
    "prepare",
    "audit",
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
        ["dry-run", "--config", "config.yaml", "--run-id", "s003-dry-001"],
        handlers={"dry-run": handler},
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["study_id"] == "s003"
    assert payload["command"] == "dry-run"

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["dry-run", "--config", "config.yaml", "--run-id", "legacy-run"]
        )


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

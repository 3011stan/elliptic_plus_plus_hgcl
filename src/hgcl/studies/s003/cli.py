"""Dedicated command-line surface for S003-TxGCL."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import sys
from typing import Any


class CommandError(RuntimeError):
    """Expected, structured command failure."""

    def __init__(self, message: str, *, exit_code: int = 4, category: str = "command") -> None:
        super().__init__(message)
        if exit_code not in {2, 3, 4}:
            raise ValueError("declared command exit codes are 2, 3, and 4")
        self.exit_code = exit_code
        self.category = category


def _s003_id(value: str) -> str:
    if not value.startswith("s003-"):
        raise argparse.ArgumentTypeError("identifier must start with s003-")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hgcl-s003")
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("doctor", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--config", required=True)
    audit.add_argument("--prepared", required=True)

    dry_run = commands.add_parser("dry-run")
    dry_run.add_argument("--config", required=True)
    dry_run.add_argument("--prepared")
    dry_run.add_argument("--run-id", required=True, type=_s003_id)

    approve = commands.add_parser("approve-dry-run")
    approve.add_argument("--run", required=True)
    approve.add_argument("--approval-file", required=True)

    matrix = commands.add_parser("matrix")
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--prepared", required=True)
    matrix.add_argument("--matrix-id", required=True, type=_s003_id)

    resume = commands.add_parser("resume")
    resume.add_argument("--run", required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)

    report = commands.add_parser("report")
    report.add_argument("--matrix", required=True)

    hetero = commands.add_parser("hetero-gate")
    hetero.add_argument("--config", required=True)
    hetero.add_argument("--prepared", required=True)
    return parser


Handler = Callable[[argparse.Namespace], Mapping[str, Any]]


def run(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[str, Handler] | None = None,
    raise_errors: bool = False,
) -> int:
    args = build_parser().parse_args(argv)
    available = handlers or {}
    try:
        handler = available.get(args.command)
        if handler is None:
            raise CommandError(f"handler is not configured for {args.command}")
        result = dict(handler(args))
        payload = {
            "study_id": "s003",
            "command": args.command,
            "status": result.pop("status", "ok"),
            "artifacts": result.pop("artifacts", []),
            **result,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0
    except CommandError as error:
        if raise_errors:
            raise
        payload = {
            "study_id": "s003",
            "command": args.command,
            "status": "error",
            "error": {"category": error.category, "message": str(error)},
        }
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        return 130


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)

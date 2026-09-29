"""Dedicated command-line surface for S003-TxGCL."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

from .artifacts import ArtifactError, ArtifactStore
from .config import ConfigError, load_config
from .data import DataError, discover_source
from .environment import EnvironmentError, RequiredEnvironment, doctor
from .pipeline import (
    audit_prepared,
    approve_dry_run,
    generate_matrix_report,
    load_causal_evidence,
    load_prepared,
    persist_prepared,
    prepare_dataset,
    run_matrix_pipeline,
    run_structural_dry_run,
    run_shadow_smoke,
    resume_run,
    validate_guarded_evaluation,
)


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

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--prepared")
    smoke.add_argument("--run-id", required=True, type=_s003_id)

    dry_run = commands.add_parser("dry-run")
    dry_run.add_argument("--config", required=True)
    dry_run.add_argument("--prepared", required=True)
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


EXACT_DEPENDENCIES = {
    "torch": "2.6.0",
    "torch-geometric": "2.6.1",
    "scikit-learn": "1.6.1",
    "numpy": "2.4.6",
    "polars": "1.44.2",
    "pyarrow": "21.0.0",
    "PyYAML": "6.0.3",
    "scipy": "1.17.1",
    "xgboost": "3.2.0",
}


def _translate_errors(action: Callable[[], Mapping[str, Any]]) -> Mapping[str, Any]:
    try:
        return action()
    except (ConfigError, DataError, ValueError) as error:
        raise CommandError(str(error), exit_code=2, category="input") from error
    except EnvironmentError as error:
        raise CommandError(str(error), exit_code=3, category="environment") from error
    except (ArtifactError, FileExistsError) as error:
        raise CommandError(str(error), exit_code=4, category="artifact") from error


def default_handlers() -> dict[str, Handler]:
    def doctor_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            config = load_config(args.config)
            raw = config.raw
            discover_source(raw["paths"]["data_root"])
            required = RequiredEnvironment(
                profile=config.profile,
                device=raw["resources"]["device"],
                exact_dependencies=EXACT_DEPENDENCIES,
                max_ram_gib=float(raw["resources"].get("max_ram_gib", 0)),
                max_vram_gib=float(raw["resources"].get("max_vram_gib", 0)),
                min_free_disk_gib=1.0,
                data_root=Path(raw["paths"]["data_root"]),
                artifacts_root=Path(raw["paths"]["artifacts_root"]),
            )
            report = doctor(required)
            return {"status": "ready" if report.ready else "blocked", "violations": list(report.violations), "artifacts": []}
        return _translate_errors(action)

    def prepare_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            config = load_config(args.config)
            prepared = prepare_dataset(config, causal_evidence=load_causal_evidence())
            path = persist_prepared(prepared, config)
            return {"status": "prepared", "data_digest": prepared.data_digest, "artifacts": [str(path)]}
        return _translate_errors(action)

    def audit_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            config = load_config(args.config)
            prepared = load_prepared(args.prepared, config)
            return {**audit_prepared(prepared), "artifacts": []}
        return _translate_errors(action)

    def smoke_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            config = load_config(args.config)
            if config.profile != "smoke":
                raise ValueError("smoke command requires profile=smoke")
            prepared = load_prepared(args.prepared, config) if args.prepared else prepare_dataset(config, causal_evidence=load_causal_evidence())
            result = run_shadow_smoke(config, prepared.training_snapshots, device=config.raw["resources"]["device"])
            store = ArtifactStore(config.raw["paths"]["artifacts_root"])
            relative = Path("runs") / args.run_id / "run.json"
            envelope = store.envelope(
                artifact_type="smoke_run",
                config_digest=config.digest,
                data_digest=prepared.data_digest,
                code_revision={"commit": "working-tree", "dirty": True},
                payload=result,
            )
            path = store.write_json(relative, envelope)
            return {"status": "complete", "engineering_only": True, "data_digest": prepared.data_digest, "artifacts": [str(path)]}
        return _translate_errors(action)

    def dry_run_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            config = load_config(args.config)
            if config.profile != "dry-run":
                raise ValueError("dry-run command requires profile=dry-run")
            prepared = load_prepared(args.prepared, config)
            result = run_structural_dry_run(config, prepared)
            store = ArtifactStore(config.raw["paths"]["artifacts_root"])
            prefix = Path("runs") / args.run_id
            revision = {"commit": "working-tree", "dirty": True}
            run_payload = {key: value for key, value in result.items() if key != "cells"}
            run_path = store.write_json(
                prefix / "run.json",
                store.envelope(
                    artifact_type="structural_dry_run",
                    config_digest=config.digest,
                    data_digest=prepared.data_digest,
                    code_revision=revision,
                    payload=run_payload,
                ),
            )
            design_path = store.write_json(
                prefix / "design.json",
                store.envelope(
                    artifact_type="matrix_design",
                    config_digest=config.digest,
                    data_digest=prepared.data_digest,
                    code_revision=revision,
                    payload={"design_digest": result["design_digest"], "cells": result["cells"]},
                ),
            )
            return {"status": "complete", **run_payload, "artifacts": [str(run_path), str(design_path)]}
        return _translate_errors(action)

    def evaluate_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            gate = validate_guarded_evaluation(args.run)
            raise ArtifactError(
                f"cohort {gate['cohort_id']} is sealed, but model loading remains blocked until the matrix registry is implemented"
            )
        return _translate_errors(action)

    def approve_dry_run_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            return approve_dry_run(args.run, args.approval_file)
        return _translate_errors(action)

    def matrix_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            config = load_config(args.config)
            prepared = load_prepared(args.prepared, config)
            return run_matrix_pipeline(
                config,
                prepared,
                matrix_id=args.matrix_id,
                artifacts_root=config.raw["paths"].get("artifacts_root", "artifacts/s003"),
                execute_cells=True,
            )
        return _translate_errors(action)

    def resume_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            return resume_run(args.run)
        return _translate_errors(action)

    def report_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            return generate_matrix_report(args.matrix)
        return _translate_errors(action)

    def hetero_gate_handler(args: argparse.Namespace) -> Mapping[str, Any]:
        def action() -> Mapping[str, Any]:
            config = load_config(args.config)
            prepared = load_prepared(args.prepared, config)
            from .hetero import evaluate_heterogeneous_gate
            artifacts_root = Path(config.raw["paths"].get("artifacts_root", "artifacts/s003")).resolve()
            output_file = artifacts_root / "hetero" / "hetero_decision.json"
            decision = evaluate_heterogeneous_gate(
                config,
                data_root=config.raw["paths"]["data_root"],
                address_files_present="wallets_features" in getattr(prepared.source, "optional_files", {}),
                features_are_causal_per_snapshot=False,
                output_path=output_file,
            )
            return {
                "status": "deferred" if decision.is_deferred else "included",
                "decision": decision.decision,
                "approved_by": decision.approved_by,
                "gates": {
                    "causality": decision.causality_gate.passed,
                    "supervision": decision.supervision_gate.passed,
                    "comparability": decision.comparability_gate.passed,
                    "resources": decision.resources_gate.passed,
                },
                "artifacts": [str(output_file)],
            }
        return _translate_errors(action)

    return {
        "doctor": doctor_handler,
        "prepare": prepare_handler,
        "audit": audit_handler,
        "smoke": smoke_handler,
        "dry-run": dry_run_handler,
        "approve-dry-run": approve_dry_run_handler,
        "matrix": matrix_handler,
        "resume": resume_handler,
        "evaluate": evaluate_handler,
        "report": report_handler,
        "hetero-gate": hetero_gate_handler,
    }


def run(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[str, Handler] | None = None,
    raise_errors: bool = False,
) -> int:
    args = build_parser().parse_args(argv)
    available = default_handlers() if handlers is None else handlers
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

# Milestone 1 execution checkpoint

Date: 2026-09-11. Scope authorized: documentation consolidation and T001–T012.

## Completed T001

Created pyproject.toml, src/hgcl/__init__.py, src/hgcl/cli.py and tests/conftest.py.
`PYTHONPATH=src .venv/bin/python -m hgcl.cli --help` and `--version` exited 0;
version 0.1.0. Python AST checks passed for all new source/test files; pyproject parsed
with tomllib. Dataset/artifacts/virtual-environment ignores verified with git check-ignore.
No training/preparation commands are advertised before implementation. Pytest tests have
not run: pytest is not installed in the isolated environment.

## Partial T002 and external blocker

`uv venv --python /opt/homebrew/bin/python3.11 --no-python-downloads .venv` succeeded.
Interpreter: CPython 3.11.15; prefix: project-local .venv. System Python unchanged.
An earlier managed-Python download failed by DNS; the existing interpreter resolved
that part. Dependency resolution remains blocked:

```sh
UV_CACHE_DIR=/private/tmp/hgcl-uv-cache uv pip compile pyproject.toml --extra dev --python .venv/bin/python --generate-hashes -o requirements/mac-cpu.lock
```

Both normal and approved escalated execution failed with exit 2 after retries:
`error sending request` to `https://pypi.org/simple/torch-geometric/` (first attempt)
and `https://pypi.org/simple/pyarrow/` (escalated attempt), `dns error`,
`failed to lookup address information: nodename nor servname provided, or not known`.
This was not an automatic approval rejection. The inspected uv cache contains tooling
packages rather than the required training stack; no compatible offline lock was found.

No dependency lock generated; no torch/PyG operation probe passed; no training packages
installed; no original-data preparation or model fitting executed. T002 remains unchecked.
Do not use a different scientific stack or mark dependent tasks complete to bypass this.

## Resume

Restore package-index connectivity and rerun the resolution command above. Install the
locked stack and editable package, execute the CPU graph operation probe, record actual
versions/hashes, then complete T002 and proceed T003–T012. Lab lock remains provisional
until hardware verification in T030. No laboratory access is needed for this milestone.

## SDD checks

Requirements checklist: 16 checked, 0 unchecked; markers unchanged. No extension hooks
file exists before or after execution. Constitution remains a draft; user authorization
covers this milestone. No new scientific choice is required for the blocked installation.

## Resumed verification and pause under D034

Manual Mac installation resolved the historical DNS blocker. Agent rechecked the lock
and installed versions; uv pip check passed for 42 packages. A bipartite GIN CPU
forward/backward probe passed. The original user's probe file was preserved.

Execution is paused on the T002/T030 laboratory-lock/backend task-boundary inconsistency
recorded in tasks.md. No task completion was inferred from the standalone probe.
No extension hooks exist. AGENTS.md permanently records the user's mandatory stop-and-consult rule.

## Completion after D035

Milestone 1 completed. Reusable CPU doctor passes. CLI commands `doctor`, `prepare`,
`validate` are implemented; no training commands advertised yet. Contract suite:
19 passed; original-data integration: 1 passed, independently preparing twice.

Latest preparation: `0bd73a7e18c09748baa0f0462f0620548a48577d9a059996c019f0ba376e50a2`.
Payload hash: `17fceeeab97c666834b4ace2994f85e5f44d97390b2040bc8c7daac864e8d1c8`.
Preparation times: 5.187 s and 3.669 s.
Peak RSS: 1.291 GiB.
The test recut includes 1,024 transactions, 4,399 address–step pairs and 4,930 physical
relations over steps 1,2,29,35. Feature vectors: address 123, transaction 32.
Each source hash was checked before and after preparation; historical evidence preserved.

Tests cover duplicates/roles, missing versus zero, malformed input, train-only preprocessing,
exact native allowlist and masks, nested label budgets/no backfill, unknown masking,
recurrence, configuration paths, atomic status transitions and source identity.
Stored artifact validation checks hashes, graph dimensions/indices, node identities,
canonical endpoint coverage, label budgets, occurrence masks and recurrence metadata.

Limits: full 49-step lab-profile preparation and native complement have not been exercised
end-to-end. Native transforms are fixture-tested. No model trained; no claim about the
600-second training target. CUDA/driver/lab lock belongs to T030.
Machine-readable evidence: milestone-1-results.json.

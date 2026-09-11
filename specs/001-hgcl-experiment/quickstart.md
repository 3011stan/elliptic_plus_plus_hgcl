# Quickstart validation guide

Preparation, CPU smoke, matrix orchestration, report and resume commands are implemented
and verified on the Mac. Actual CUDA environment and scientific matrix execution remain pending.
Run from `/Users/stan/Projects/masters-degree/hgcl-elliptic`.

## Mac environment and data — verified

The existing `.venv` uses Python 3.11.15. No activation is required.

```sh
.venv/bin/python -m hgcl.cli doctor --config configs/smoke.yaml --json
.venv/bin/python -m pytest tests/contract -q
.venv/bin/python -m hgcl.cli prepare --config configs/smoke.yaml --json
```

Copy the `prepared` path printed by the last command:

```sh
.venv/bin/python -m hgcl.cli validate --prepared PREPARED_PATH --config configs/smoke.yaml --json
```

The smoke data profile keeps 256 transactions per step at 1,2,29,35, selected without
labels. Preparation verifies all nine original hashes and the full principal endpoint
coverage before recutting. Raw originals remain unchanged. Prepared snapshots contain
no targets; targets and masks are stored separately. Same-code/config repeats independently
rebuild and compare hashes; the first completed destination is never overwritten.

Run the integration acceptance (two original-data preparations):

```sh
HGCL_RUN_ORIGINAL_TESTS=1 .venv/bin/python -m pytest tests/integration/test_preparation.py -q -s
```

Expected: 19 contract tests and 1 integration test pass. Preparation measurements and
hashes are recorded in docs/validation/milestone-1-results.json. They do not measure training.

For a clean Mac environment, using the already generated lock:

```sh
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements/mac-cpu.lock
uv pip install --python .venv/bin/python --no-deps --editable .
uv pip check --python .venv/bin/python
```

Do not recreate the existing environment unnecessarily. Laboratory lock generation and
CUDA/sampler validation are entirely T030. Full 49-step lab-profile preparation is
implemented but has not yet been exercised end-to-end; native transforms are fixture-tested.

## Complete real-data smoke (US2) — verified

```sh
.venv/bin/python -m hgcl.cli smoke --config configs/smoke.yaml --prepared artifacts/prepared/PREPARED_HASH --run-id smoke-003
```

Runs smoke-001 and smoke-002 already completed. Use a new run ID; existing runs are never overwritten.
Verified: SSL, two graph fits, RF, fusion/threshold selection, freeze, evaluate and reload.
Finite losses, updated encoder/head, aligned evaluation keys and fitting+selection <=600 s.
Preparation/final evaluation measured separately. Output says engineering smoke; no claim
of scientific superiority. Failure retains reasons; revised config uses a new run ID.

## Matrix orchestration (US3) — implemented; laboratory execution pending

The following enumeration is available now, without training or reading dataset labels:

```sh
.venv/bin/python -m hgcl.cli matrix --config configs/lab.yaml --dry-run
```

Without --prepared, this only checks the design, not prepared data integrity.
T030 must first implement/verify CUDA doctor and generate the lab lock on real hardware.
After T030, install the validated lab lock in a separate Python 3.11 environment on the lab machine,
configure its local data root, then:

```sh
python -m hgcl.cli doctor --config configs/lab.yaml --json
python -m hgcl.cli prepare --config configs/lab.yaml --json
python -m hgcl.cli matrix --config configs/lab.yaml --prepared artifacts/prepared/LAB_HASH --matrix-id s02-001 --dry-run
python -m hgcl.cli matrix --config configs/lab.yaml --prepared artifacts/prepared/LAB_HASH --matrix-id s02-001
python -m hgcl.cli report --matrix artifacts/matrices/s02-001
```

Dry run enumerates 80+20 keys without test-label access. Run only after smoke/device checks.
Final report contains paired differences, five-seed mean/sample SD, strata, support and
explicit missing runs. Dry-run success is not completion of scientific execution.

## Resume — implemented

```sh
python -m hgcl.cli resume --run artifacts/runs/INTERRUPTED_RUN_ID
```

Integration check deliberately interrupts a small pre-test fit. Compatible epoch/RNG
restore must agree with uninterrupted CPU execution. Changed hashes and completed
post-test refitting are rejected.

Repeat the same matrix command and matrix ID to reuse completed groups and resume compatible
pre-test groups. Code, data, config and environment-lock changes are rejected. Run-level
resume archives prior status/failure/provenance in attempts/. A frozen run can proceed to
evaluation; a test-scored/evaluated run cannot be refitted. Interrupted post-scoring runs
remain incomplete for inspection. No automatic retuning from test metrics occurs.

Reports write report.json and report.md. Missing evaluations return exit 4, with explicit
available/defined seed counts; mean/sample SD use available defined values and remain
marked incomplete when fewer than five exist. Native-wallet results carry input-policy
and global-label limitations. Candidate configurations (150 planned) are distinct from
100 evaluation keys and ten SSL caches; they do not count repeated interrupted epochs.

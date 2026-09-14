# CLI contract

Implemented `hgcl` / `python -m hgcl.cli` entry point through T029. CUDA doctor remains T030.

| Command | Inputs | Result |
| --- | --- | --- |
| `doctor --config PATH` | profile/device | environment operation probe JSON, no model training |
| `prepare --config PATH` | original data root | verified derived data/preparation hash |
| `validate --prepared PATH --config PATH` | prepared manifest | schema/graph/mask counts and errors |
| `fit --config PATH --prepared PATH --run-id ID` | validated data | fitted branches and frozen validation selection, no test metrics |
| `evaluate --run PATH` | frozen run | saved predictions, then test target join/metrics |
| `smoke --config PATH --prepared PATH --run-id ID` | smoke only | fit/freeze/evaluate/reload and 600-second acceptance |
| `matrix --config PATH --prepared PATH --matrix-id ID --dry-run` | lab profile | 100-key DAG, no training/test-label access |
| `matrix --config PATH --prepared PATH --matrix-id ID` | validated matrix | serial cached fit/evaluate with statuses |
| `resume --run PATH` | compatible partial checkpoint | epoch-boundary restore/replay |
| `report --matrix PATH` | run results | mean/SD, paired differences, strata, missing-run list |

Global overrides: `--data-root`, `--artifacts-root`, `--device cpu|cuda`, `--json`.
Precedence CLI > ELLIPTIC_DATA_ROOT/HGCL_ARTIFACTS_ROOT > YAML. YAML declares project_root
relative to its own location; resolve relative paths there, never from arbitrary cwd.
Progress to stderr; final JSON to stdout. Locks/driver installation are not CLI side effects.
Exit 0 completed stage; 2 invalid data/config; 3 execution/resource/time failure;
4 incompatible resume or incomplete report. Record failure state when a run exists.
No default overwrites, uploads, lab access or post-test refitting.

For matrix --dry-run, --prepared and --matrix-id are optional. Omitting --prepared
enumerates the design only; the output explicitly says data_integrity_checked=false.
Actual matrix execution requires both arguments and a validated laboratory lock.

## Laboratory operator wrapper — D049

`scripts/lab/run.py audit` resolves the prepared path from `lab-prepare.json`,
requires current doctor/smoke/prepare receipts, performs full semantic validation
and writes compact `lab-audit.json`. It omits address selections and computes no
test metrics. `PASS` is validation status, not researcher acceptance.

`scripts/lab/run.py dry-run` additionally requires a compatible audit receipt,
checks preparation/payload identities and writes `lab-dry-run.json`. It returns
zero for `planned` without executing any matrix group. Both stages leave human
acceptance pending. Matrix invocation remains a separate, explicitly authorized
operator action under `../t031-execution-protocol.md`. Existing direct CLI remains
available; no models, label budgets, dependencies or scientific settings changed.

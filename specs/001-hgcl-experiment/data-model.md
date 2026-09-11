# Data model

Schema version 1. Source IDs are strings; tensor indices int64; aggregations float64;
model features float32. Targets are stored separately from graph inputs.

| Entity | Key | Fields / invariant |
| --- | --- | --- |
| Source | filename | hash, size, header, audit version; immutable |
| Transaction | tx_id | one step, 15 named raw numeric values and missing flags |
| Address occurrence | address, step | one canonical row, duplicate count, principal/native descriptors |
| Relation | step, role, address, tx_id | unique per physical role, canonical edge ID links reverse |
| Snapshot | step, regime | ID maps, address vectors 123/110, transaction vectors 32, edges; no y |
| Global target | address | class 1/2/3, explicit binary mapping, unknown masked |
| Label budget | partition, fraction, seed | selected unique addresses, counts, occurrence IDs, hash |
| Preprocessor | regime, fit-ID hash | train-only medians/scales, feature order and version |
| Recurrence | address | seen-in-train/development; actual label exposure is run-specific |
| Recut | preparation hash | selected original IDs, steps, caps, label-free hash rule |
| Checkpoint | run, stage, epoch | states/RNG, dependency hashes, atomic status |
| Frozen selection | run | models, preprocessing, alpha/threshold, validation-mask hash |
| Prediction | run, address, step | finite score [0,1], class, selection hash; unique key |
| Evaluation | regime, fraction, seed, method, stratum | metrics, support, validity reasons, prediction/target hashes |
| Matrix | matrix_id | 100 expected keys, cache DAG, statuses, retry history |

## Files and state

`artifacts/prepared/<hash>/`: canonical Parquet, raw derived per-step vectors, ID maps,
adjacency arrays and manifest. Hash includes source/schema/recut. Fitted preprocessing
is separately hashed and never uses validation/test fit IDs.
`artifacts/runs/<id>/`: resolved-config.yaml, provenance.json, budget.json, status.json,
timings.json, checkpoints/, selection.json, predictions.parquet, metrics.json,
validation-search.parquet. Matrices store DAG manifest, statuses, aggregate and paired reports.

States: planned → validated → fitting → frozen → scored → evaluated → complete.
Any active stage may fail/interruption with reason and last valid checkpoint. Resume only
compatible pre-test fitting. Scientific input changes require new run ID. Write temporary
siblings and atomically rename; success manifest last. Never overwrite a completed run.

Blank input is missing; real zero is not. Invalid numerics have source/key diagnostics.
Undefined metrics are null plus reason/support, not NaN. Evaluation joins targets only
after saved predictions. Training interfaces accept immutable explicit permitted masks.

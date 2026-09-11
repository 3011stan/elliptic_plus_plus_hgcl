# Tasks: H-GCL experiment and late fusion

Input: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts](contracts/cli.md), [quickstart](quickstart.md).
T001–T029 are complete. T030–T033 remain pending. T026–T029 validation:
55 regression tests passed, two optional tests skipped; original-data smoke passed separately. The original-data CPU smoke
passed; detailed evidence is in ../../docs/validation/smoke.md and smoke-results.json.
Evidence and measured
limits: ../../docs/validation/milestone-1-results.json. Laboratory environment work
belongs entirely to T030 (D035). Existing historical audits were preserved.
Tests are required by the specification for data, temporal integrity, model state and smoke.
Paths below are repository-relative. [P] indicates independent files once stated prerequisites pass.

## Phase 1 — Setup

- [X] T001 Create the `src/hgcl/` package, CLI entry point and pytest layout in `pyproject.toml` and `tests/conftest.py`; preserve original data/audit scripts and ignore derived artifacts. (FR-001, FR-011, FR-018)
- [X] T002 Establish the isolated Mac Python 3.11 environment and requirements/mac-cpu.lock with transitive versions/hashes; implement/probe reusable CPU doctor in src/hgcl/environment.py and record results in docs/environments.md. Laboratory locking, CUDA and sampler verification belong entirely to T030 (D035). Depends T001. (FR-018)

## Phase 2 — Foundation

- [X] T003 Implement typed schema, path precedence, fixed protocol validation and resolved hashes in `src/hgcl/config.py`; materialize `configs/smoke.yaml` and `configs/lab.yaml` from the configuration contract, rejecting unknown/incompatible settings. Depends T002. (FR-010, FR-011, FR-016, FR-018)
- [X] T004 [P] Implement immutable source/run provenance, atomic status writes and failure states in `src/hgcl/provenance.py`, including unborn Git/source-tree hashes and checkpoints' dependency identities. Depends T002. (FR-001, FR-012, FR-019, SC-004)

## Phase 3 — US1: Verified data and inputs (P1)

Goal: prepared data and masks can be reproduced without a model.
Independent acceptance: verified originals, canonical pairs/edges, contractual vectors,
train-only preprocessing and identical repeat-preparation IDs/hashes.

- [X] T005 [US1] Add fixtures and contract tests in `tests/contract/test_data_contract.py` for duplicate occurrence/edge invariance, both address roles, blank versus zero, conflicting IDs and future-row perturbation. Depends T003,T004. (FR-002, FR-003, FR-005, FR-021, FR-024, SC-001, SC-002, SC-009)
- [X] T006 [US1] Implement projected CSV ingestion and validation in `src/hgcl/data/ingest.py` and `src/hgcl/data/schema.py`; reuse existing audit/report identities, verify all nine hashes and referenced endpoints, write derived Parquet and isolate target store. Depends T005. (FR-001, FR-002, FR-003, FR-015, FR-022, SC-001)
- [X] T007 [P] [US1] Implement exact allowlists, principal address summaries and native-wallet complement in `src/hgcl/data/features.py`, retaining context-volume semantics and documented size-unit limitation. Depends T006. (FR-005, FR-020, FR-024)
- [X] T008 [US1] Implement missing flags, train-only median/log/scaling fit and immutable transform in `src/hgcl/data/preprocessing.py`; persist fit IDs and feature order, reject entirely missing training columns. Depends T007. (FR-005, FR-006, FR-024, SC-006)
- [X] T009 [US1] Implement canonical snapshot/ID maps, typed physical/reverse edges and deterministic label-free smoke recut in `src/hgcl/data/snapshots.py`; no classes or persistent identity embeddings in graph inputs. Depends T008. (FR-003, FR-007, FR-021, FR-022, SC-002, SC-009)
- [X] T010 [P] [US1] Implement split masks, nested unique-address budgets and recurrence/exposure flags in `src/hgcl/data/splits.py`; validate both-class availability, floor counts, shared masks and selected-label overlap without validation backfill. Depends T006. (FR-006, FR-014, FR-015, FR-016, SC-006)
- [X] T011 [US1] Wire `prepare` and `validate` in `src/hgcl/cli.py` and `src/hgcl/pipeline.py`; emit prepared manifests, diagnostics and measured preparation duration; fail before model fitting. Depends T009,T010. (FR-002, FR-007, FR-010, FR-020)
- [X] T012 [US1] Run original-data preparation twice and compare artifact identities in `tests/integration/test_preparation.py`; reference INV-001 and `docs/data/transaction-input-check.json` without overwriting evidence or rerunning its historical investigation unnecessarily. Depends T011. (FR-023, SC-001, SC-002, SC-008)

## Phase 4 — US2: Real-data end-to-end smoke (P1)

Goal: complete SSL, both graph fits, RF, fusion, frozen evaluation and reload on the Mac.
Independent acceptance: same evaluation keys, finite losses, actual encoder/head updates,
immutable reusable SSL checkpoint and training+selection <=600 seconds on original-data recut.

- [X] T013 [US2] Add graph/state tests in `tests/unit/test_graph_learning.py` for typed anchor alignment, paired reverse-edge dropout, no class-driven augmentation, identical control architecture and encoder/head gradients. Depends T012. (FR-008, FR-009, FR-021, FR-022, SC-009)
- [X] T014 [US2] Implement shared type-projected relation-sum encoder and affine head in `src/hgcl/models/encoder.py` and `src/hgcl/models/heads.py`; one root contribution per destination, two synchronous layers, LayerNorm. Depends T013. (FR-008, FR-009, FR-013, FR-021)
- [X] T015 [US2] Implement numeric masking, physical/reverse dropout and typed normalized contrastive loss in `src/hgcl/models/augmentations.py` and `src/hgcl/models/contrastive.py`; positives retain original typed IDs, negatives only same-snapshot/type distinct anchors. Depends T014. (FR-008, FR-022)
- [X] T016 [US2] Implement full-recut CPU loading and bounded PyG neighbor loading behind one interface in `src/hgcl/data/sampling.py`; preserve canonical IDs/edge pairing, anchor-only losses and deterministic tails with node/edge guards. Depends T015. (FR-007, FR-008, FR-011, FR-021)
- [X] T017 [P] [US2] Implement atomic epoch-boundary checkpoints with RNG, optimizer, sampler and dependency hashes in `src/hgcl/training/checkpoint.py`; reject mismatched resumes and mutation of cached SSL weights. Depends T013,T004. (FR-009, FR-012, FR-019, SC-007)
- [X] T018 [US2] Implement train-only fixed-epoch SSL and per-regime/seed cache in `src/hgcl/training/pretrain.py`, with separate type-anchor batches, finite-loss checks, logged updates and timing. Depends T016,T017. (FR-006, FR-008, FR-009, FR-012)
- [X] T019 [US2] Implement exact chunked layer-wise inference in `src/hgcl/training/inference.py` and compare with full-snapshot outputs in `tests/unit/test_inference.py`; keep eval state and all-neighbor sums independent of chunk size within tolerance. Depends T018. (FR-006, FR-009, FR-021)
- [X] T020 [US2] Implement paired SSL-initialized/from-scratch supervised fits in `src/hgcl/training/supervised.py`, identical head stream/search budget, permitted labels/class weights and AP checkpoint selection using T019 inference; discard contrastive head. Depends T019. (FR-006, FR-008, FR-013, FR-014, FR-016)
- [X] T021 [P] [US2] Implement RF candidate fits and stable illicit-score mapping in `src/hgcl/models/tabular.py` with shared inputs/masks, fixed candidate grid and persisted model selection. Depends T012. (FR-008, FR-013, FR-014, FR-024)
- [X] T022 [US2] Implement AP selection, F1 thresholds, scalar fusion, deterministic ties and metric null/support policy in `src/hgcl/evaluation/selection.py` and `src/hgcl/evaluation/metrics.py`; test threshold endpoints and score/class alignment in `tests/unit/test_selection.py`. Depends T020,T021. (FR-013, FR-015, FR-016, FR-017, SC-005)
- [X] T023 [US2] Wire fit/freeze/evaluate/smoke lifecycle in `src/hgcl/pipeline.py` and `src/hgcl/cli.py`; save predictions before target joins, enforce total fitting+selection timeout and expose phase-specific timing/errors. Depends T022. (FR-008, FR-009, FR-010, FR-011, FR-012, FR-019, SC-003, SC-004)
- [X] T024 [US2] Add integration checks in `tests/integration/test_smoke.py` and `tests/integration/test_resume.py` for graph gradient changes, reload, immutable SSL cache, interrupted replay and original-data end-to-end execution. Depends T023. (FR-009, FR-012, FR-019, SC-003, SC-007)
- [X] T025 [US2] Execute and measure the Mac smoke, recording `artifacts/runs/smoke-001/timings.json` and `docs/validation/smoke.md`; retain failed attempts, explicitly revise only resource profile if needed, and require <=600 seconds before declaring success. Depends T024. (FR-007, FR-010, FR-020, SC-003, SC-004)

## Phase 5 — US3: Controlled scientific matrix (P2)

Goal: execute/resume the accepted matrix and report paired results without test-guided fitting.
Independent acceptance: dry-run exactly 100 keys, matched masks/prediction populations,
freeze enforcement, complete or explicitly incomplete five-seed/recurrence reports.

- [X] T026 [US3] Add partition-access and matrix tests in `tests/contract/test_evaluation_access.py`; trap test-label access during fit/dry-run, forbid validation-label backfill and mismatched evaluation keys, exercise seen/unseen and undefined-stratum metrics. Depends T025. (FR-006, FR-013, FR-014, FR-015, FR-016, FR-022, SC-005, SC-006)
- [X] T027 [US3] Implement the 80+20 evaluation DAG and serial cache-aware runner in `src/hgcl/matrix.py`; reuse 10 SSL caches and branch scores, separate candidate-fit counts, reject post-test retuning and preserve incomplete statuses. Depends T026. (FR-011, FR-012, FR-013, FR-016, FR-019, FR-024)
- [X] T028 [US3] Expose compatible `resume` and matrix restart in `src/hgcl/cli.py` using `src/hgcl/training/checkpoint.py`; reject scientific/source/data changes and already evaluated refits; retain attempt history. Depends T027. (FR-012, FR-019, SC-007)
- [X] T029 [US3] Implement overall/seen/unseen support, prior-label exposure, paired RQ1/RQ2 differences and five-seed mean/sample SD reports in `src/hgcl/evaluation/reporting.py`; label native temporal limitations and incomplete groups; allow non-improvement. Depends T028. (FR-004, FR-013, FR-016, FR-020, FR-024, SC-004, SC-005)
- [ ] T030 [US3] Implement `doctor` in `src/hgcl/environment.py`, then run it on the lab machine when accessible; verify CUDA/sampler/backward, memory/free space and data hashes, generate, install and validate the complete transitive/hash lock `requirements/lab-cuda.lock` and record `docs/validation/lab-preflight.md`. CPU doctor's basic checks are implemented in T002; CUDA acceptance depends on real hardware. Depends T029. (FR-018, SC-001)
- [ ] T031 [US3] Run matrix dry-run and then the laboratory experiment via `configs/lab.yaml`, retaining `artifacts/matrices/s02-001/` and a portable summary in `docs/validation/experiment.md`; do not mark complete for dry-run alone or without all required results. Depends T030. (FR-012, FR-013, FR-016, FR-019, FR-020, SC-004, SC-005)

## Phase 6 — Handoff and cross-cutting verification

- [ ] T032 Update `README.md`, `docs/environments.md` and `specs/001-hgcl-experiment/quickstart.md` with verified commands, actual locks and operational limitations; preserve evidence and distinguish planned/observed timings. Depends T025,T030. (FR-018, FR-020)
- [ ] T033 Re-run only affected required checks and verify saved artifacts against all FR/SC in `docs/validation/acceptance.md`; include source snapshot, data identity, 600-second smoke, full matrix completeness and pending external conditions honestly. Depends T031,T032. (FR-012, FR-020, SC-001, SC-002, SC-003, SC-004, SC-005, SC-006, SC-007, SC-008, SC-009)

## Dependencies and parallel opportunities

Setup T001→T002; foundation T003/T004; US1 T005–T012; US2 T013–T025;
US3 T026–T031; handoff T032–T033. This pipeline has real story dependencies; do not
claim all stories can be implemented independently at once. Each story has its own
acceptance checkpoint once its prerequisites exist.

Examples: US1 T007 (features) and T010 (budgets) after T006; US2 T014 (encoder) and
T017 (checkpoint) after T013; T021 (RF) can proceed independently after T012. US3
report fixtures can be prepared alongside matrix implementation after T026, but T029
integration waits for T028. [P] is a task property, not an instruction to spawn agents.

## Implementation strategy and scope

First deliver US1 as the data MVP, then US2 as the user's runnable engineering milestone.
Do not postpone smoke to the end of laboratory experimentation. Lab access is not needed
for T001–T029 or Mac smoke; T030/T031 require the actual machine or an operator-provided
probe. Never declare them completed from documentation. No paid services, dataset uploads,
driver changes or model training occurred during task generation.

## Requirements coverage

| Requirement | Tasks |
| --- | --- |
| FR-001 | T001, T004, T006 |
| FR-002 | T005, T006, T011 |
| FR-003 | T005, T006, T009 |
| FR-004 | T029 |
| FR-005 | T005, T007, T008 |
| FR-006 | T008, T010, T018, T019, T020, T026 |
| FR-007 | T009, T011, T016, T025 |
| FR-008 | T013, T014, T015, T016, T018, T020, T021, T023 |
| FR-009 | T013, T014, T017, T018, T019, T023, T024 |
| FR-010 | T003, T011, T023, T025 |
| FR-011 | T001, T003, T016, T023, T027 |
| FR-012 | T004, T017, T018, T023, T024, T027, T028, T031, T033 |
| FR-013 | T014, T020, T021, T022, T026, T027, T029, T031 |
| FR-014 | T010, T020, T021, T026 |
| FR-015 | T006, T010, T022, T026 |
| FR-016 | T003, T010, T020, T022, T026, T027, T029, T031 |
| FR-017 | T022 |
| FR-018 | T001, T002, T003, T030, T032 |
| FR-019 | T004, T017, T023, T024, T027, T028, T031 |
| FR-020 | T007, T011, T025, T029, T031, T032, T033 |
| FR-021 | T005, T009, T013, T014, T016, T019 |
| FR-022 | T006, T009, T013, T015, T026 |
| FR-023 | T012 |
| FR-024 | T005, T007, T008, T021, T027, T029 |
| SC-001 | T005, T006, T012, T030, T033 |
| SC-002 | T005, T009, T012, T033 |
| SC-003 | T023, T024, T025, T033 |
| SC-004 | T004, T023, T025, T029, T031, T033 |
| SC-005 | T022, T026, T029, T031, T033 |
| SC-006 | T008, T010, T026, T033 |
| SC-007 | T017, T024, T028, T033 |
| SC-008 | T012, T033 |
| SC-009 | T005, T009, T013, T033 |

## Execution checkpoint — 2026-09-11

T001: package scaffold, pyproject, CLI help/version and pytest layout created. CLI verified
with isolated Python 3.11.15. No preparation/training command is exposed prematurely.
T002: `.venv` created from the existing Homebrew Python 3.11.15. Dependency resolution
failed on PyPI DNS both normally and with approved escalation; no lock or operation
probe is claimed. See ../../docs/validation/milestone-1.md. T003 onward awaits T002.

## Execution paused under D034 — 2026-09-11

User installed the Mac stack and produced artifacts/environment/mac-cpu-probe.json.
Agent verified installed versions, uv pip check (42 compatible packages) and CPU
bipartite GIN forward/backward successfully. requirements/mac-cpu.lock exists with
version/hash pins. The prior Mac DNS installation blocker is resolved.

T002 still requires the reusable CPU doctor implementation and requirements/lab-cuda.lock.
The laboratory OS/driver are unconfirmed (docs/environments.md); T030 also owns laboratory
validation/final locking. plan.md additionally says T002 must exercise the sampling
backend, while the implementation strategy says laboratory access is unnecessary before
T030. This inconsistent task boundary is not silently changed. Under D034, execution
is paused to ask whether laboratory lock/backend work should move entirely to T030,
leaving T002 responsible for the validated Mac environment and reusable CPU doctor.
No T003–T012 implementation or dataset preparation has occurred in this resumed turn.

## Milestone 1 accepted by execution — 2026-09-11

D035 resolved the prior task-boundary blocker. T002 CPU doctor passed; T003–T004
configuration/provenance tests passed. T005–T010 data/feature/mask contracts passed.
T011 doctor/prepare/validate commands are implemented. T012 passed two independent
original-data smoke preparations with identical preparation and payload hashes.
19 contract tests plus 1 original-data integration test passed. Native inputs tested
with fixtures; full laboratory-profile preparation/training has not been executed.
Detailed reports: ../../docs/validation/milestone-1.md and milestone-1-results.json.
No new impediment or scientific scope change occurred. T013 onward remains pending.

## Milestone 2 pause under D034

D036 authorizes T013–T025. T013–T015 graph-learning tests passed (4): gradients,
matched head initialization, paired edge dropout, feature-mask preservation, contrastive
alignment and permutation equivariance. T016 loader/anchor implementation is partial;
T017 checkpoint implementation is partial and its epoch replay/dependency rejection test
passed. The subsequent suite returned 5 passed, 1 failed:
`tests/unit/test_checkpoint.py::test_tail_has_no_fabricated_or_lost_anchors`.
For 65 anchors and batch size 32, expected lengths [32,33], observed [33,32].
The list assignment in anchor_batches evaluates chunks.pop() before its destination
index, so the tail is merged into the wrong batch. Execution stopped without applying
a fix under AGENTS.md / D034. Researcher approval to correct and resume is pending.
No SSL/supervised original-data training, original-data mutation or scope change occurred.

## Resumed under D037; new pause in T022

Authorized T016 tail correction applied: explicitly pop tail, then append to the last
remaining batch. 16 graph/checkpoint/tail tests passed, including ten boundary cases.
T016 pairing/guard fixture test passed; real optional laboratory backend remains T030.
T017 atomic checkpoint replay, immutable-write rejection and dependency guards passed.
T018 SSL, T019 chunked inference, T020 supervised fitting, T021 RF and T022 selection
have been authored. Inference matches full-graph output for three chunk settings.
End-to-end pretraining/fitting has not yet been tested or run; these tasks remain pending.

Latest unit suite: 21 passed, 1 failed in
`test_threshold_agrees_with_bruteforce_and_endpoints`. For y=[0,1], scores=[1,1],
the test expects threshold 1; implementation returns 0. Both thresholds predict all
positive, yielding F1=2/3 and equal distance to 0.5. The accepted final tie-break selects
the lower threshold, 0. Thus this assertion contradicts the agreed contract.
Execution paused under D034/D037 to request correcting this test expectation and adding
an explicit tie-break regression check. No test fix applied; no original-data training.

## Milestone 2 completed after D038

Threshold test expectation corrected as authorized; accepted tie-break unchanged.
23 unit tests passed, then actual SSL interruption/replay integration passed (24 tests).
T018/T019 verified SSL and exact chunked inference; T020–T022 integrated matched graph
fits, RF and validation selection. T023 fit/freeze/evaluate/smoke CLI is implemented.
T024 tests validate actual parameter changes, immutable SSL, replay and model reload.
T025 original-data smoke-001 passed in 2.176 seconds fitting/selection, under 600 seconds;
final evaluation/reload 0.101 s, preparation 5.859 s, measured separately.
Final regression: 43 passed, 2 opt-in original-data tests skipped; original smoke passed
separately. No later failure or scope change occurred. Full matrix/lab T026–T033 pending.

## T026 pause under D034 after D039 authorization

Added tests/contract/test_evaluation_access.py. Three checks passed: fitting API rejects
test partition requests; prediction populations/duplicate keys are rejected; undefined
stratum metrics are explicit. The fit entry-point isolation test failed: runner.fit calls
pipeline.validate, which reads targets/global.parquet including test address labels.
This is validation access, not demonstrated optimization/selection use, but violates the
strict fit access contract. No original-data training was run. Proposed split between
structural integrity and label-semantic validation awaits researcher consultation.
Evidence: ../../docs/validation/matrix-readiness.md. No correction applied yet.


## T026–T029 subsequent pause under D034

D040 access correction passed. Twelve focused tests passed; matrix CLI dry-run
enumerated 100 keys. Full regression: 54 passed, 2 skipped, 1 failed because the new
matrix fixture used a relative artifacts path and collided with its previous run.
No further correction applied pending consultation. See docs/validation/matrix-readiness.md.
Tasks remain unchecked until repeatable regression and original-data smoke verification.


## T026–T029 completed under D041

Fixture paths corrected and 25 identified simulated runs plus their matrix removed;
smoke-001 preserved. Repeated isolated/regression tests passed (55 passed, 2 skipped).
Original smoke-002 passed separately, fitting/selection 2.226 s. Matrix dry-run: 100
keys, 25 groups, 10 SSL caches, 150 supervised/RF candidate configurations. No scientific
matrix training performed. Evidence: ../../docs/validation/matrix-readiness.md and
matrix-results.json. No extension hooks configured. T030–T033 remain unchecked.

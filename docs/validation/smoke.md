# Original-data smoke acceptance

Status: PASS. Run: smoke-001. Scope: T013–T025, Mac CPU.

## Verified behavior

- SSL pretraining: 2 epochs, 128 updates, encoder parameters changed.
- Graph control and H-GCL fine-tuning: same encoder/head architecture and head
  initialization, shared permitted labels/class weights; both encoder and head changed.
- RF, validation-only thresholds and scalar fusion completed for the same address keys.
- SSL checkpoint hash remained unchanged after supervised fitting.
- Models/selection frozen before test inference; predictions saved before test target join.
- Reloaded model predictions reproduced within declared numerical tolerance.
- Epoch-boundary SSL interruption/resumption reproduced uninterrupted weights exactly
  on the integration fixture. Incompatible checkpoint dependencies are rejected.

## Measured timing

Training and selection: **2.176 seconds**, accepted limit 600 seconds.
SSL: 1.770 s; supervised control: 0.127 s;
H-GCL fine-tuning: 0.132 s; RF: 0.065 s;
selection/persistence: 0.079 s.
Final test evaluation/reload: 0.101 seconds, outside fitting time.
Original-data preparation: 5.859 seconds, measured separately; preparation peak RSS
0.981 GiB. Training memory was guarded at 8 GiB; an exact training peak was not logged.
PyTorch CPU threads recorded: 4.

## Data and checks

1,024 original transactions; 4,399 address–step pairs; 4,930 physical relations.
Steps 1,2 train; 29 validation; 35 test. Each retained step has 256 transactions selected
without labels. Each address vector has 123 dimensions; transaction vector 32.
These are engineering settings, not the 100-evaluation scientific matrix.

Regression: 43 tests passed; 2 original-data opt-in tests skipped in the regression
command. The original-data smoke test was separately enabled and passed (17.54 s wall
clock, including import/preparation/test assertions). The real-data preparation repeat
acceptance was already passed in milestone 1. No claim of scientific model superiority.

Artifacts remain under artifacts/runs/smoke-001. Portable evidence and artifact hashes:
[smoke-results.json](smoke-results.json). Do not rerun with the same run ID.

## Implementation choices and remaining scope

The SSL update averages address-anchor and transaction-anchor losses. If their batch
counts differ, the shorter type list cycles so each update retains equal type weighting;
no fake anchors or extra labels are introduced. Batches/IDs are paired across both views.
This operational choice is logged in ssl-report.json.

CPU full-snapshot smoke verified. The neighbor adapter's pairing/guard fixture is tested;
actual laboratory sampler/CUDA operation and dependency locking belong entirely to T030
under D035. T026–T033 cover the full scientific matrix, multi-run resume orchestration,
stratified/paired reporting, lab execution and final acceptance. They are not completed.

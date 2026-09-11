# Implementation Plan: H-GCL and late fusion on Elliptic++

**Feature**: `001-hgcl-experiment` | **Git branch**: `main` (unborn) | **Date**: 2026-09-07
**Spec**: [spec.md](spec.md) | **Status**: milestone 1 completed on 2026-09-11; see tasks.md and ../../docs/status.md.
Spec Kit resolves the feature ID through `.specify/feature.json`; it did not create a Git branch.

## Summary

Implement a local Python CLI for immutable-data verification, temporal snapshot preparation,
SSL, matched supervised graph classifiers, RF, validation-only fusion and frozen evaluation.
Deliver an original-data smoke on the Mac before the lab matrix: 80 principal and 20
native-wallet evaluations. Reuse branch outputs for fusion and label-independent SSL
within each regime/seed; 100 evaluations do not mean 100 separate SSL pretrainings.

Normative inputs: [input-contract.md](input-contract.md); scientific policy:
[training-design.md](training-design.md); interfaces: [CLI](contracts/cli.md),
[configuration](contracts/run-config.md), [data model](data-model.md).

## Technical Context

- Language: Python 3.11 in `.venv`, leaving installed Python 3.14.6 unchanged.
- Initial core pins: torch 2.6.0, torch-geometric 2.6.1, scikit-learn 1.6.1. NumPy,
  Polars, PyArrow, PyYAML and pytest are resolved and locked with their transitive
  dependencies in T002. These are conservative starting versions, not a latest-version claim.
- Storage: projected Parquet, NumPy arrays, JSON manifests and atomic model checkpoints;
  no database server, tracking service or cloud dependency.
- Platform: macOS arm64 CPU smoke; lab CUDA only after local operation probe. MPS is not
  part of initial acceptance. Lab OS/driver/wheel compatibility remains a measured gate.
- Project: one research library and `hgcl` CLI under `src/hgcl/`.
- Scope: 203,769 transactions, 822,942 addresses, 920,691 address–time pairs, 49 snapshots.
- Targets: smoke fitting plus selection <=600 seconds; preparation/final evaluation separate.
  Guard RSS at 8 GiB on Mac, 24 GiB in lab, and allocated GPU memory at 4.5 GiB.
- Verification: pytest contracts, feature/label isolation, real-data preparation, smoke,
  exact inference equivalence, checkpoint replay and matrix dry run.

## Constitution Check

Constitution v0.1.0 is a draft, used as the operational review baseline without implicit
ratification. User authorization covers planning; its draft status does not impose an
additional permission gate. Review before research and after design:

| Principle | Before | After |
| --- | --- | --- |
| User decisions precede dependent tasks | Pass, D001–D030 | Pass, scientific choices preserved |
| Temporal and label boundaries | Pass | Pass, train/validation masks and freeze boundary |
| Original data and provenance | Pass | Pass, derived-only writes and hashes |
| Bounded real-data smoke | Pass | Pass, explicit failure/time acceptance |
| Controlled comparison | Pass | Pass, same masks, paired seeds, matched graph control |

No principle violations. Lab speed/driver checks are execution gates, not assumed facts.

## Project Structure

```text
specs/001-hgcl-experiment/
  plan.md, research.md, data-model.md, quickstart.md, tasks.md
  spec.md, input-contract.md, training-design.md
  contracts/cli.md, contracts/run-config.md
src/hgcl/
  cli.py, config.py, environment.py, provenance.py
  data/ingest.py, schema.py, features.py, preprocessing.py
  data/splits.py, sampling.py, snapshots.py
  models/encoder.py, augmentations.py, contrastive.py, heads.py, tabular.py
  training/pretrain.py, supervised.py, checkpoint.py, inference.py
  evaluation/selection.py, metrics.py, reporting.py
  pipeline.py, matrix.py
configs/smoke.yaml, configs/lab.yaml
requirements/mac-cpu.lock, requirements/lab-cuda.lock
scripts/                         # existing audit scripts preserved
tests/unit/, tests/contract/, tests/integration/
artifacts/prepared/, artifacts/runs/, artifacts/matrices/, artifacts/environment/
```

Source/config/test paths are planned deliverables. Reuse existing audit logic after
compatibility checks; do not rewrite INV-001 evidence or mark old checks as new work.

## Phase 0 — Research decisions

Preserve research.md and append technical decisions with official sources. The explicit
`size` input retains the source numeric unit: no byte/vbyte conversion or fee-per-byte
feature. The unverified extraction unit is a limitation, not a missing computational input.

CPU smoke uses full-batch message passing on its small recut and anchor minibatches for
contrastive loss, avoiding a compiled sampler dependency on Mac. Lab uses PyG NeighborLoader
with an official matching optional sampling backend. T030 must generate the laboratory lock and exercise that backend,
not merely import it (D035). This is a configurable loading strategy within the same pipeline.
No custom production neighbor sampler or package compilation is the default path.

## Phase 1 — Preparation and label budgets

Project columns while reading CSVs; validate original hashes, schemas, keys and references;
store canonical Parquet by step. Build physical and computational reverse relationships,
then derive features from original, unaugmented snapshot relations. Global labels live
in a separate store and never enter graph/feature builders.

Fit preprocessing only on training snapshots. Select unique labeled addresses separately
within train and validation, with class-stratified deterministic nested samples: for each
class c select floor(p*N_c) by SHA-256 rank of `seed|partition|class|address`. Use all at
100%; a zero class count invalidates the configuration instead of inflating its budget.
Report nominal/realized fractions and overlap of selected addresses across partitions.
Sampler access to benchmark labels for stratification is not model access. Apply masks
to every selected address occurrence in that partition, once per pair. Validation-only
labels cannot backfill training masks. Reuse masks across methods and the native 100% arm.

## Phase 2 — Models, bounded batches and inference

Two synchronous encoder layers, separate type projections, 128 lab hidden dimensions.
Combine relation-specific transformed neighbor sums with one destination root contribution
per layer (not one root per relation), then a type-specific MLP. Use LayerNorm so inference
batch composition does not change normalization statistics. Do not claim a homogeneous
GIN expressivity theorem for this adaptation.

Lab sampling: two incoming hops, per-relation fanouts 10 then 5; 128 loss anchors per type.
SSL separately processes an address-anchor and transaction-anchor batch and averages
losses into one update. Views share the base sampled graph and IDs, with distinct masking
and paired edge-drop randomness. Same-type distinct anchors are negatives; context nodes
are not extra targets. Merge the last <2-anchor batch with its predecessor or explicitly
skip it if no such merge exists. Do not fabricate labels or contrastive negatives.

Keep feature/adjacency arrays on CPU and transfer one batch. Cap sampled nodes/edges at
50,000/200,000. OOM/guard violations fail explicitly; revised resource settings are a new
resolved run and must match across graph methods. Never silently drop difficult nodes.

Inference uses exact layer-wise sums in destination and edge chunks, with CPU intermediate
activations. Complete all incoming sums for a destination before root/nonlinearity. Use
all neighbors for validation/test; no stochastic inference resampling. Verify numerical
agreement with full-snapshot inference on a fixture. Eval mode and preprocessing stay frozen.

SSL is label-independent, fixed-epoch, cached per regime/seed. Fine-tuning copies it;
fractions never continue from another fraction's supervised checkpoint. The paired control
uses identical architecture, head initialization stream, supervision and selection budget.
Record SSL and supervised compute separately; total compute is not equal across methods.

## Phase 3 — Selection, evaluation and persistence

Select checkpoints/candidates on permitted validation AP; choose standalone thresholds
and scalar fusion alpha/threshold on validation F1. Freeze model/preprocessing/config/mask
and selection hashes. Test scoring requires frozen state; save unique keyed predictions
before joining test targets. Never retune from overall or recurrence-stratum test metrics.

Report pooled address–time F1 illicit, precision, recall, AP, MCC, confusion/support counts;
also seen/unseen relative to development 1–34 and prior-label-use flags. Empty or undefined
metrics are null with reason. Precision with no predicted positives is 0; no-positive
population F1/recall, single-class AP and zero-denominator MCC are null. Main populations
must contain both classes. Aggregate mean/sample SD (ddof=1) and paired seed differences;
incomplete five-seed groups remain incomplete, never masquerade as full results.

Atomic epoch-boundary checkpoints contain model/optimizer/RNG, sampler state, best selection,
config/data/feature/mask/source hashes. Resume replays an unfinished epoch and rejects
incompatible inputs or refitting completed test-evaluated runs. Record Git revision when
available plus source-tree digest and dirty/untracked-source manifest; an unborn repository
has no commit ID. Scientific runs need a recorded immutable source snapshot.

## Phase 4 — Smoke and laboratory acceptance

Smoke recuts original transactions without labels, preserves incident relations and derives
features with the same formulas. Four principal methods, one seed, 100% of recut labels;
this is not a scientific scarce-label run. Preflight tests native inputs separately.
Count time from SSL start through all fitting/checkpoint/fusion selection, including its
validation inference. Preparation and final frozen test inference/evaluation are separate.
A >600-second attempt is a failed smoke, even if it eventually finishes.

Laboratory dry run enumerates exactly 100 evaluation keys without test-label access.
Default serial GPU scheduling, 10 cached SSL pretrainings (two regimes × five seeds),
shared branch scores for fusion. Candidate-fit counts are separate from evaluation count.
Doctor on the lab machine verifies OS/driver, free space, float32 forward/backward and
sampler. This planning session does not access the lab or install its drivers.

## Delivery gates

US1: reproducible prepared data and no feature/label mixing. US2: original-data smoke
with updated encoder/head, finite losses, aligned outputs, reload and time acceptance.
US3: 100-key matrix, resume, five-seed reports and recurrence/complement labeling.
Dry-run success is not scientific completion. See tasks.md for FR/SC coverage.

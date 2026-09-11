# Research record — temporal availability of Elliptic++ features

**ID**: INV-001  
**Created**: 2026-09-07  
**Status**: Investigation completed; accepted policy detailed in [input-contract.md](input-contract.md).  
**Traceability**: [spec.md](spec.md), FR-002, FR-005, FR-020, FR-023, SC-008;
[decisions](../../docs/decisions.md), D008, D018, D020.

**Results**: [INV-001 evidence and recommendations](../../docs/data/inv-001/README.md),
[full-scan artifact](../../docs/data/inv-001/report.json),
[independent sample verification](../../docs/data/inv-001/verification.json).
Empirical source: `s02-2026-inv001-temporal-audit`, BibTeX `s02inv001temporalaudit2026`.

The three initial concepts were examined: active-step count is the complete observed
step count for all addresses; transaction totals reproduce full dataset incident counts
for all recurrent addresses; last-block cutoff remains indeterminate. See the evidence
table for population counts, the one transaction-total exception and interpretation limits.

## Place in the SDD workflow

INV-001 was evidence gathering during clarification. It is complete and preserved below;
the input contract, plan and tasks were subsequently completed. Implementation status
is maintained in ../../docs/status.md, not in this historical investigation.
The feature remains `001-hgcl-experiment`; no separate feature or training pipeline is needed.

Sequence: accepted requirements → INV-001 evidence → researcher discussion and input-policy
decision → update spec/decisions/checklist → implementation plan → tasks → consistency
analysis → implementation → original-data smoke test → laboratory experiment.

The later planning phase must retain and extend this record, rather than overwrite its
evidence or treat unanswered research questions as settled. Architectural research can
continue there; questions that change the scientific task return to clarification.

## Question and decision to support

For an address observed at step `t`, which published features can be demonstrated to use
only information available by the end of that step? A row's step identifies an occurrence;
it does not itself establish the history used to calculate the row's features.

The decision is whether to use native attributes with explicit temporal limitations,
use verified/recomputed step-restricted attributes in both branches, or evaluate both
settings with one designated as primary. The researcher selected option A on 2026-09-07: snapshot-derived principal inputs in both branches plus a limited native-feature complementary setting.
Removing native wallet attributes only from H-GCL does not establish temporal validity
for fusion if another branch or its downstream classifier still consumes them.

## Existing evidence and its limits

- Local observations: [initial audit](../../docs/data/README.md),
  [machine-readable report](../../docs/data/initial-audit.json),
  [source manifest](../../docs/data/source-manifest.json). Non-time wallet attributes
  are invariant per address across occurrences; labels are global per address; raw
  address–time pairs repeat. These observations do not establish a calculation cutoff
  for every feature. This investigation uses the same original files.
- Source S1, Table 3, Section 3.2 and Appendix A.1: address observations accompany
  transactions, and feature extraction queries the public blockchain. The table distinguishes
  blockchain transaction totals from dataset transaction counts. The consulted text does
  not establish a per-observation cutoff for every address feature.
- Source S2, Sections 4.1–4.2: Inspection-L trains on 34 transaction graphs and applies
  the trained encoder to the remaining 15. This supports the temporal inspiration, not
  direct numerical comparability with wallet classification.
- Source S3, Section 5.1: GCPAL is related SSL work. The exact 1–34/35–49 split and identical
  snapshot handling have not been established from the consulted text; do not attribute
  that protocol without a specific source locator.

**Initial hypotheses and their resolution**: future activity is now evidenced for the
active-step count and full-dataset transaction-count equivalence. The suggestion that
all native features use wider blockchain history is not established; the observed totals
instead match dataset counts almost universally. Invariance alone remains insufficient
proof, and no claim about all 55 attributes or an exact extraction algorithm is accepted.

## Bounded investigation

Initial feature concepts: number of active time steps, total transactions, and last active
block. Map these concepts to exact CSV column names and definitions before testing them.

| Work item | Status | Method and required output |
| --- | --- | --- |
| INV-001-A | Complete | Exact columns and source definitions mapped; discrepancy between blockchain-total wording and observed dataset-count equality documented. |
| INV-001-B | Complete | 13 recurrent addresses selected by a fixed hash/lexical rule, without labels; 34 address–step examples and source locators persisted. |
| INV-001-C | Complete | Full scan of four CSVs, distinct incident transaction counts and prefix comparisons; one total-count exception documented; selected examples verified independently with SQLite. |
| INV-001-D | Complete with documented limits | Paper, official listings and tutorial content inspected. Extraction implementation and verified block-to-step mapping not located in consulted material; last-block cutoff marked indeterminate. |
| INV-001-E | Complete; option A accepted | Evidence and implications delivered; policy selected in D018. Input contract and complementary run matrix were subsequently fixed in D029–D031. |

This is a discovery checklist, not `tasks.md`. Its items produce evidence, not a training
implementation. Original CSVs remain unchanged. Supporting analysis and verification
scripts are now available under `scripts/`; no training pipeline was implemented.

Dataset-derived counts describe the observed sample, not necessarily the address's complete
blockchain activity. A mismatch between those counts and a blockchain-total feature is not
by itself proof of future leakage. Auditing temporal metadata across steps is permitted;
using test labels, outcome associations or test model scores to choose features is not.

## Evidence table and acceptance criteria

For each initial feature, report:

| Field | Required content |
| --- | --- |
| Identity and meaning | Exact column, unit, definition and source locator. |
| Reproducibility | Dataset hashes via manifest, address IDs, steps, selection rule, method and supporting output. |
| Comparison | Published value, observed/recomputed quantity, observation cutoff and scope of history. |
| Conclusion | Future information demonstrated; availability by cutoff demonstrated; or indeterminate. State whether the result applies to an example, a subset, or the extraction procedure. |
| Limitation and action | Alternative explanations, missing evidence and implications for use/reconstruction/exclusion. |

- [x] All three feature concepts have evidence records or an explicit reason why verification is unavailable.
- [x] Examples can be reproduced; any counterexample is traceable to source records.
- [x] Observation time, calculation history and extraction time are distinguished.
- [x] Sample findings are not generalized to all columns or addresses without evidence.
- [x] No test-label analysis or test-performance-based model/feature selection was performed.
- [x] Options state their consequences for the tabular branch, transaction inputs, H-GCL and fusion.

An **indeterminate** finding is a valid investigation result; proving leakage is not the
acceptance criterion. Finish the bounded investigation when these outputs are complete,
then record the researcher's decision and rationale under D018 and FR-005. If uncertainty
remains, the decision must either restrict the claims/inputs or specify a bounded follow-up.

Checking these three features does not approve all remaining wallet or transaction features.
A claim of temporal validity for the full pipeline requires an explicit policy covering
every admitted input, as well as the evaluation and label-availability limitations.

## Planning readiness

INV-001 completion alone does not make the feature ready for scientific implementation.
These requirements are now resolved in D023–D030 and the linked input/training contracts.
Technical planning, initial library/resource settings and acceptance checks are now
recorded in plan.md and contracts/. Runtime verification belongs to implementation. The constitution is still a draft, not silently ratified.
Encoder, augmentations, downstream training regime and fusion details belong in the plan
once their scientific requirements are settled. The 600-second smoke training target
remains an engineering acceptance criterion with preparation measured separately.

## Sources

- **S4 — empirical analysis**: [s02-2026-inv001-temporal-audit](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/s02-2026-inv001-temporal-audit.md>),
  BibTeX `s02inv001temporalaudit2026`; [evidence](../../docs/data/inv-001/README.md).

- **S1**: [elmougy-2023-demystifying](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/elmougy-2023-demystifying.md>),
  BibTeX `elmougy2023demystifying`; Table 3, Section 3.2, Appendix A.1;
  [primary text](https://arxiv.org/html/2306.06108v1).
- **S2**: [lo-2022-inspection-l](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/lo-2022-inspection-l.md>),
  BibTeX `lo2022inspectionl`; arXiv v4, Sections 4.1–4.2;
  [primary text](https://arxiv.org/html/2203.10465v4).
- **S3**: [lu-2024-gcpal](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/lu-2024-gcpal.md>),
  BibTeX `lu2024gcpal`; Section 5.1;
  [primary text](https://link.springer.com/article/10.1007/s44196-024-00720-4).

## Decision outcome

The researcher explicitly accepted option A on 2026-09-07 (D018): snapshot-derived
attributes in both branches for the principal experiment, plus a limited complementary
setting with native attributes and explicit temporal limitations. The policy covers
address and transaction inputs and downstream classifiers. Feature lists/formulas, deduplication, complementary runs and evaluation are now
specified in D023–D030 and the linked contracts. Acceptance
of the policy is not approval of an unspecified feature set or evidence of historical
label availability. The current phase is milestone 1 implementation. The source size unit remains an explicit limitation.

## Subsequent contract checks — separate from INV-001

The 2026-09-07 transaction-input check supports the explicit financial allowlist, not a
new temporal-leakage conclusion. It checks 15 fields numerically on training steps 1–28,
without labels, and verifies the source hash before/after. See
[input-contract.md](input-contract.md) and the registered empirical source
[s02-2026-transaction-input-check](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/s02-2026-transaction-input-check.md>),
BibTeX `s02transactioninputcheck2026`. No training has been implemented or run.

## Implementation planning — 2026-09-07

Phase 0 decisions are recorded below; Phase 1 artifacts are plan.md, data-model.md,
contracts/ and quickstart.md. The original INV-001 report and its scope remain unchanged.

| Decision | Rationale | Alternatives considered |
| --- | --- | --- |
| Python 3.11, torch 2.6.0, PyG 2.6.1, sklearn 1.6.1 baseline | Isolated conservative environment with documented installation paths; lock and probe in T002 | Reusing installed Python 3.14.6 would couple the project to unverified binary availability |
| CPU full-recut smoke; lab PyG neighbor batches | Avoid optional sampler binaries for the small Mac graph; retain bounded lab computation | Full 49-graph GPU load rejected for 6-GB memory; custom sampler not the first implementation |
| Lab torch CUDA 12.4 candidate and matching PyG wheel index | Official binary paths exist; actual OS/driver compatibility still requires doctor | CUDA 11.8 documented alternative if the lab probe requires it; never auto-change scientific config |
| Polars/PyArrow projected ingest, Parquet/NumPy storage | Read required columns and process snapshots without Python-object copies of all features | Whole-CSV dataframe copies and database server unnecessary |
| LayerNorm and exact chunked inference | Stable per-node normalization and all-neighbor inference without full GPU residency | Batch-statistic inference dependence and stochastic test sampling avoided |
| Small fixed candidate grids | Concrete bounded selection budget, identical graph controls | Unbounded tuning would consume scarce validation information and compute |
| Preserve published size unit | Input value is usable without byte/vbyte conversion | Guessing extraction units or dropping required size would change the contract |

Sources: PyTorch v2.6.0 installation section (`pytorchinstallation2026`), PyG minimal /
additional libraries sections (`pyginstallation2026`), PyG CUDA12.4 wheel inventory
(`pygcu124wheels2026`), sklearn installation (`sklearninstallation2026`). Registered notes:
- [pytorch-2026-version-installation](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/pytorch-2026-version-installation.md>), `pytorchinstallation2026`; [official source](https://pytorch.org/get-started/previous-versions/).
- [pyg-2026-installation](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/pyg-2026-installation.md>), `pyginstallation2026`; [official source](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html).
- [pyg-2026-torch26-cu124-wheels](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/pyg-2026-torch26-cu124-wheels.md>), `pygcu124wheels2026`; [official source](https://data.pyg.org/whl/torch-2.6.0+cu124.html).
- [sklearn-2026-installation](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/sklearn-2026-installation.md>), `sklearninstallation2026`; [official source](https://scikit-learn.org/1.6/install.html).

These sources establish installation options, not a passed local/lab test. Transitive
versions and binary hashes will be frozen by T002; the lab lock is validated by T030.
No package installation, remote laboratory inspection or training occurred while planning.

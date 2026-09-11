# INV-001 — Temporal feature evidence

**Completed**: 2026-09-07  
**Status**: Investigation completed and verified; option A subsequently accepted (D018). Detailed feature contract pending.  
**Evidence source**: `s02-2026-inv001-temporal-audit`, BibTeX `s02inv001temporalaudit2026`,
[source record](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/s02-2026-inv001-temporal-audit.md>).

## Findings

The published step count contains future dataset activity for early observations of
recurrent addresses. Published transaction totals also reproduce the complete observed
dataset history for every recurrent address, rather than the prefix available at its
first step. The exact temporal cutoff of the last-block field remains indeterminate.
These conclusions concern the inspected columns and files, not all 55 wallet features.

| Feature and unit | Observed evidence | Temporal conclusion | Scope and limitation |
| --- | --- | --- | --- |
| `num_timesteps_appeared_in`, distinct active steps | Equals the complete set of observed steps for all 822,942 addresses. For 61,487 recurrent addresses it exceeds the observed prefix at earlier steps; 97,749 distinct address–step pairs are affected. | Future dataset information demonstrated for those early observations: a count of subsequent active steps is already present. | This does not establish how the other columns were extracted or when labels became known. |
| `total_txs`, transaction count | Equals distinct incident dataset transaction IDs for 822,941 of 822,942 addresses, including all 61,487 recurrent addresses. For those recurrent addresses the value already includes the full observed count at the first occurrence. It exceeds the prefix on the same 97,749 pairs. | Full-history dataset-count equivalence demonstrated; the native value is not the observed dataset count available by `t`. Treat it as future-inclusive for that task, rather than as a verified historical input. | The extraction procedure was not recovered. The paper's broader blockchain-total description is not independently verified by this equality; unseen blockchain history could not be reconstructed. One non-recurrent exception is documented below. |
| `last_block_appeared_in`, block height | Invariant across occurrences; all recurrent addresses have different first/last block values. Example below retains 485,959 even at its first observed step. | Indeterminate numeric cutoff: no verified block-height-to-step-boundary mapping was established. It is not approved as historically available. | No explicit block-height/timestamp column exists in `txs_features.csv`. Inferring boundaries from the same suspect last-block field would not provide independent verification. |

All counts above cover addresses regardless of class. No label file was read, and no
class-specific prevalence or performance effect was estimated. Matching a final count
alone is not evidence of a particular extraction algorithm. The transaction conclusion
is grounded in the explicit join to observed future transactions and full-dataset equality;
it is not a claim that every blockchain transaction of every address is in this dataset.

## Reproducible example

Address: `1111DAYXhoxZx2tsRnzimfozo783x1yC2` (first lexical recurrent-address anchor).

| Step | Published active-step count | Observed steps through this step | Published `total_txs` | Distinct dataset transactions through this step |
| --- | --- | --- | --- | --- |
| 25 | 6 | 1 | 8 | 1 |
| 29 | 6 | 2 | 8 | 2 |
| 39 | 6 | 3 | 8 | 4 |
| 43 | 6 | 4 | 8 | 6 |
| 47 | 6 | 5 | 8 | 7 |
| 48 | 6 | 6 | 8 | 8 |

At step 25 the address is incident to transaction `50030829`. The other seven observed
transactions occur at steps 29, 39, 43, 47 and 48. Thus filtering the row to step 25
leaves the native values 6 and 8; it does not restore the prefix values 1 and 1.
The corresponding wallet source rows start at CSV line 3 (header is line 1).
Edge `50030829` is at `TxAddr_edgelist.csv` line 466160. See all source locators in
[example-source-rows.csv](example-source-rows.csv) and [example-edges.csv](example-edges.csv).

The prefix column above is a diagnostic, not a switch to a cumulative-graph model.
For the agreed snapshot-only design, a derived transaction count would instead count
distinct incident transactions within `G_t`; the count of active steps within a single
snapshot is always one and offers no within-snapshot differentiation.

## Exception and secondary observations

- `12C5SxqPCLdsiyxUaXUFypQdMpcLmZvUhx` appears only at step 49 with `total_txs=4`,
  while the two edge files contain one distinct incident transaction, `157659071`.
  Sender/receiver fields are 3 and 1. The discrepancy is real, but its cause is unresolved;
  it does not demonstrate future information for this address. Retain it as a data-contract
  exception, without silently correcting the original CSV.
- No duplicate `(address, txId)` rows occur within either inspected incidence relation.
  Transactions in which an address appears in both roles are counted once in the union.
  Sender/receiver feature counts do not universally equal distinct relation degrees;
  do not assume these semantics from names alone. These auxiliary counts are in `report.json`.
- 10,471 addresses occur on both sides of the candidate 34/35 cut. This is an unlabeled
  identity-overlap count, not approval of that cut or a claim about the labeled test subset.

## Method and verification

The full scan reads four original files only: wallet features, transaction features and
the two address–transaction edge lists. It verifies each SHA-256 against the existing
source manifest and again after analysis. Original files remain unchanged.

The scan covers 1,268,260 wallet rows, 822,942 addresses, 920,691 distinct address–step
pairs and 203,769 transaction IDs. It joins 477,117 sender edges and 837,124 receiver
edges, deduplicating incident transactions across roles. Selected feature values are
checked for invariance across occurrences; row widths, integral counts/blocks, endpoints
and agreement between wallet and incident-transaction step sets are checked.

Selection uses the ten recurrent addresses with smallest SHA-256 of the UTF-8 address,
plus the first three recurrent addresses in lexical order. The union contains 13 addresses
and 34 distinct address–step examples. The rule uses no labels or outcomes.

A separate verifier rereads the original CSVs and uses SQLite `COUNT(DISTINCT ...)`
queries, independently of the main script's bitmasks and dictionaries. It confirms all
34 examples, 44 wallet source rows and the single transaction-total exception. It also
checks input/evidence hashes. This is independent verification of the sample and exception,
not a second independent full-population audit. See [verification.json](verification.json).

The final full scan took 21.535 seconds and approximately 708 MiB peak RSS on this Mac
(Python 3.14.6). This is a measurement of data analysis, not a training benchmark.

```sh
python3 scripts/investigate_temporal_features.py \
  --data-root elliptic-plus-plus/raw \
  --output-dir artifacts/inv-001/results
python3 scripts/verify_temporal_evidence.py \
  --data-root elliptic-plus-plus/raw \
  --evidence-dir artifacts/inv-001/results
```

Run these commands from the technical repository. No external Python dependencies are
required. Reruns change completion time/runtime metadata; deterministic selections,
counts and example CSV contents should remain the same for identical inputs.
Script hashes and four input hashes are stored in [report.json](report.json).

## Upstream documentation inspection

Associated source: `elmougy-2023-demystifying`, BibTeX `elmougy2023demystifying`;
[source record](</Users/stan/Projects/stan-os/10-projects/Masters Degree/01-evidence/sources/elmougy-2023-demystifying.md>).
Inspected on 2026-09-07:

- [Paper](https://arxiv.org/html/2306.06108v1), Table 3, Section 3.2 and Appendix A.1:
  describes active steps, transaction totals, first/last blocks and blockchain extraction;
  it does not establish a per-row historical cutoff for these native values.
- [Official repository](https://github.com/git-disl/EllipticPlusPlus), root plus Actors
  Dataset and Transactions Dataset directory listings, and Actors README: the inspected
  listings expose datasets and tutorial notebooks; no address-extraction implementation
  or verified block-to-step mapping was located there.
- [Actors statistics notebook](https://github.com/git-disl/EllipticPlusPlus/blob/main/Actors%20Dataset/Elliptic%2B%2B_Actors_Dataset_Statistics.ipynb),
  cell ID `dNNEwGmae2Eo`: loads existing feature CSVs. Its stored overview includes the
  same lexical example and repeated first/last block values. This corroborates the
  example's presence upstream; it does not authenticate every local file or its extraction.
- [Actors classification notebook](https://github.com/git-disl/EllipticPlusPlus/blob/main/Actors%20Dataset/Elliptic%2B%2B_Actors_Classification.ipynb),
  setup/overview: consumes existing attributes; the inspected content did not provide
  their original extraction cutoff. No notebook was executed and no reported model
  scores were used to choose examples or inputs.

Repository references are to `main` as consulted; a commit-pinned copy was not obtained.
Direct terminal retrieval failed DNS resolution; inspection used the web research tool.
The author's presentation PDF could not be opened by that tool because of its size and
was not used as evidence. No author was contacted and no local dataset was uploaded.

## Design implications and recommendation at investigation delivery

For a principal experiment intended to control future feature information, prefer
explicitly computed snapshot descriptors in **both** branches, reserving native-attribute
runs as a qualified complementary setting. This preserves the address–step/global-label
target and tests SSL/fusion under a common information policy. It does not establish
real-time crime anticipation or historical label availability.

Possible snapshot inputs include distinct sending/receiving transaction counts and
transaction/address degrees computed from the selected step's edges. These represent
the observed dataset subgraph, not complete blockchain history. Global active-step
counts and unknown-cutoff block fields would be omitted from this setting. Any native
transaction attributes would need their own availability policy; wallet filtering alone
does not protect messages received from transaction nodes.

| Alternative | Tabular branch | H-GCL/downstream | Fusion and scientific claim |
| --- | --- | --- | --- |
| Native inputs as principal | Published attributes, documented retrospective limitations | Explicit native input policy; same boundary limitations | Tests performance under released feature conditions; not evidence of prospective, future-free detection. |
| Snapshot-derived inputs as principal — recommended for temporal control | Features computed from `G_t` only | Node type and allowed snapshot descriptors, with structural augmentations still to design | Both branches respect the same observed information horizon; label-time and recurrent-address limitations remain. |
| Both settings | Separate feature configurations | Same comparison protocol within each setting | Measures sensitivity to native attributes; increases run count. Choose primary/complementary status before test evaluation. |

Native-versus-restricted performance differences would reflect all changed information,
not a pure quantitative estimate of leakage. A snapshot-derived feature contract and
appropriate contrastive augmentations must be specified before implementation. IDs are
indices rather than learned identity inputs by default; constant features cannot support
an informative within-type feature-shuffling corruption on their own.

After this investigation, the researcher accepted option A (D018): snapshot-derived
principal inputs in both branches and a limited native-feature complementary setting.
The next SDD action is the exact feature contract, followed by remaining evaluation
decisions. No feature list, training code, hyperparameters or scientific performance
result is approved merely by this policy decision.

# Principal input contract

Status: technical specification under D018, D021 and D022, 2026-09-07.
This defines principal and complementary inputs. The accepted complement uses four
methods at 100% labels with five seeds (D029). It does not certify historical availability of global labels.

## Source identity and scope

Resolve the data root through configuration; defaults to `elliptic-plus-plus/raw`.
Verify originals against `docs/data/source-manifest.json`. Never rewrite source files.
Use exact header names rather than positions copied from the original proposal.

| File | Use |
| --- | --- |
| `txs_features.csv` | `txId`, `Time step` and the explicit allowlist below |
| `AddrTx_edgelist.csv` | Unique `(input_address, txId)` sender relationships |
| `TxAddr_edgelist.csv` | Unique `(txId, output_address)` receiver relationships |
| `wallets_features.csv` | Distinct `(address, Time step)` identity/coverage only in the principal setting |
| `wallets_classes.csv` | Separate global targets: 1 → illicit/positive, 2 → licit/negative, 3 → unknown/masked |

The combined features/classes file is not an input to model preprocessing. Transaction
classes, address–address edges and transaction–transaction edges are not required for
this principal bipartite graph. Classes never enter feature tensors or contrastive views.

## Identity and deduplication

- One transaction per `txId`, one time step per transaction. Conflicting duplicates fail
  validation; exact duplicates, if introduced in a derived import, collapse with a count.
- One address node per `(address, t)`. Collapse repeated wallet occurrence rows by key,
  then derive its features from incident transactions; never sum repeated native rows.
- One edge per `(relation, address, txId)`. Sender and receiver are distinct roles: an
  address occurring in both roles retains both relationships. The incident union counts
  a transaction once. Multiple UTXOs are not recoverable from these unweighted edges.
- Build each graph from transactions with `Time step == t` and their incident addresses.
  Validate endpoints and agreement with wallet occurrence keys. Report unsupported
  wallet occurrences rather than silently creating or dropping evaluation targets.
- Raw IDs are opaque keys, not numeric inputs or learned identity embeddings. Map them
  deterministically within each snapshot. Time step is metadata for slicing, not a
  predictive input. No adjacency or hidden-state carryover between snapshots.

## Transaction allowlist

| Exact columns | Meaning and unit | Handling |
| --- | --- | --- |
| `total_BTC` | Published transaction BTC volume | Preserve source name; not an individual address amount |
| `fees` | Transaction fee, BTC | Context of the incident transaction, not a fee attributed to every address |
| `size` | Published transaction size | Preserve raw numeric unit; exact byte/vbyte extraction semantics remain unverified; do not derive fee-per-byte |
| `num_input_addresses`, `num_output_addresses` | Published input/output address counts | Do not assume these equal unique addresses represented in the supplied graph |
| `in_BTC_min`, `in_BTC_max`, `in_BTC_mean`, `in_BTC_median`, `in_BTC_total` | Input amount summaries, BTC | Transaction-level summaries |
| `out_BTC_min`, `out_BTC_max`, `out_BTC_mean`, `out_BTC_median`, `out_BTC_total` | Output amount summaries, BTC | Transaction-level summaries |

These 15 explicit properties describe the transaction itself according to Elliptic++
Table 2 and Section 3.1 (`elmougy2023demystifying`). Availability by the end of its observed
step is the input-policy assumption; extraction code has not been independently verified.
The unit of `size` is not asserted as bytes merely from its name or typical values.

Recompute two additional transaction features from the current graph: number of distinct
input addresses and number of distinct output addresses. Do not consume native
`in_txs_degree`/`out_txs_degree`: their wider graph semantics are unnecessary here.
Native `Local_feature_*`, `Aggregate_feature_*` and all 55 native wallet attributes
are outside the principal allowlist. Anonymous local fields lack individual semantic
mapping; the exclusion does not claim that every excluded field leaks future information.

Transaction model vector: 15 numeric properties, 15 missingness flags, two recomputed
counts = 32 inputs. Model dimensions must be generated from the contract, not hard-coded
from the original draft's 183-feature description.

## Address features shared by Random Forest and graph encoder

For each address `a` and step `t`, define unique transaction sets `S(a,t)` (sender),
`R(a,t)` (receiver), and their union. Use only original, unaugmented snapshot relations.

- Three counts: `n_sender_txs`, `n_receiver_txs`, `n_incident_txs` (union).
- For each of the 15 allowlisted transaction properties, and each of the two roles,
  compute `sum`, `mean`, `max` over observed values and `observed_fraction` (observed
  transactions / transactions in that role). This gives 120 summaries, or 123 address inputs.
- No transactions in a role: counts and summaries are zero. With transactions but no
  observed value: numeric summaries are zero and observed fraction is zero; the role
  count distinguishes this from no activity. Real observed zeros stay observed zeros.
- Name fields `sender_context__fees__mean`, for example. A sum of transaction volume
  over incident transactions is context volume, not the address's BTC sent/received.
  No equal division across addresses or reconstruction of unavailable per-edge BTC.
- Aggregate raw observed values before imputation, scaling or graph augmentation.
  Augmented views do not recalculate descriptors or purport to be new valid ledgers.

The tabular branch and address nodes receive the same 123 descriptors. Transaction nodes
add their own 32 inputs; the graph model also receives topology. This is a controlled
comparison of specified representations, not proof of topology's isolated contribution.

## Missing values and preprocessing

- Parse blanks as missing. Unexpected nonnumeric/nonfinite values or negative values
  in this nonnegative allowlist fail the input check with locators; do not silently clip.
- For transaction inputs, impute each missing numeric property with its training-step
  median and preserve its missingness flag. Fit on unique training transactions only,
  without labels. Fail if a whole training column lacks observations.
- Apply `log1p` to nonnegative amounts, size, counts and sum/mean/max summaries.
  Keep masks and observed fractions unchanged. Standardize graph numeric inputs with
  statistics from training snapshots only; zero-variance scale becomes 1. RF uses the
  same descriptors and deterministic log transform, without mandatory standardization.
- Fit address statistics on distinct training address–time pairs, transaction statistics
  on distinct training transactions. Never fit on validation/test, including unlabeled data.
- Preserve raw values and masks in derived artifacts for audit. Schema/dimension,
  missingness counts and preprocessing parameters accompany every run.

## Evidence checked in this clarification

`scripts/check_transaction_inputs.py` produces
[`transaction-input-check.json`](../../docs/data/transaction-input-check.json).
Registered empirical source: `s02-2026-transaction-input-check`, BibTeX
`s02transactioninputcheck2026` (linked in research.md).
It verifies the transaction file hash before and after a numerical check restricted to
steps 1–28. No labels are read and no model is trained.

- 116,529 training transactions; 116,251 have all 15 finite fields.
- Each field has 278 blanks; those rows account for the incomplete vectors.
- No observed negative, nonnumeric or nonfinite values in the selected train fields.
- On complete rows, `in_BTC_total - out_BTC_total == fees` and
  `total_BTC == out_BTC_total` within 1e-7 BTC in this check. These are diagnostics,
  not a claim about extraction code or automatic rules for dropping other rows.

Scope excludes validation/test numerical distribution analysis, full pipeline execution,
per-address financial attribution and unit confirmation for `size`. Prior identity and
edge coverage evidence remains in the initial audit and INV-001; it was not rerun here.

## Limited native-wallet complement — D029

Use the same graph relations, 32 transaction inputs, split, models, five seeds and label
policy as the principal setting. At 100% labels only, replace the 123 snapshot address
summaries in BOTH branches with the 55 native wallet properties listed below. Do not
append anonymous transaction features. This isolates the wallet input-policy change;
it is not a full reproduction of the source paper or a pure leakage-effect estimate.

Collapse wallet rows by `(address, t)` and require identical non-key values within each
key; fail on conflicts. Join the global label separately. Preserve native values and
units; the historical horizon of every column has not been certified. Do not interpret
these inputs as known in real time. Fit median imputation and numeric scaling only on
training pairs, retaining one missingness flag per property (110 address inputs).
Apply the same nonnegative log policy, with malformed values reported before fitting.
A source-native zero is not silently reinterpreted as missing.

Exact 55-column allowlist, in source order:

```text
num_txs_as_sender
num_txs_as receiver
first_block_appeared_in
last_block_appeared_in
lifetime_in_blocks
total_txs
first_sent_block
first_received_block
num_timesteps_appeared_in
btc_transacted_total
btc_transacted_min
btc_transacted_max
btc_transacted_mean
btc_transacted_median
btc_sent_total
btc_sent_min
btc_sent_max
btc_sent_mean
btc_sent_median
btc_received_total
btc_received_min
btc_received_max
btc_received_mean
btc_received_median
fees_total
fees_min
fees_max
fees_mean
fees_median
fees_as_share_total
fees_as_share_min
fees_as_share_max
fees_as_share_mean
fees_as_share_median
blocks_btwn_txs_total
blocks_btwn_txs_min
blocks_btwn_txs_max
blocks_btwn_txs_mean
blocks_btwn_txs_median
blocks_btwn_input_txs_total
blocks_btwn_input_txs_min
blocks_btwn_input_txs_max
blocks_btwn_input_txs_mean
blocks_btwn_input_txs_median
blocks_btwn_output_txs_total
blocks_btwn_output_txs_min
blocks_btwn_output_txs_max
blocks_btwn_output_txs_mean
blocks_btwn_output_txs_median
num_addr_transacted_multiple
transacted_w_address_total
transacted_w_address_min
transacted_w_address_max
transacted_w_address_mean
transacted_w_address_median
```

## Acceptance checks for implementation

Verify duplicate occurrence invariance, unique edges per role, same-step joins, one
prediction per labeled address–time pair, exact feature allowlists, preserved real zeros,
missingness flags, train-only preprocessing, and absence of classes/IDs in model inputs.
Include an address in both roles and a transaction with unavailable financial fields.
Changing future rows must not change training features or fitted preprocessing.

Source: [Elliptic++, Section 3.1 and Table 2](https://arxiv.org/html/2306.06108v1#S3.SS1),
BibTeX `elmougy2023demystifying`; local evidence uses the original-file manifest above.

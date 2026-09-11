# Training and fusion design

2026-09-07. D027 confirms supervised fine-tuning after SSL; D028 retains recurring test addresses;
D029 fixes the 20-evaluation native-wallet complement. Routine defaults below are
specified under D022; framework versions, memory sizing and execution tasks belong to
the [implementation plan](plan.md) and [tasks](tasks.md). This document records the
scientific design; those artifacts define its implementation.

## Data and information access

Use [input-contract.md](input-contract.md). Train 1–28; validate 29–34; test 35–49.
Use five paired seeds (technical default: 11, 23, 37, 53, 71), with separate deterministic
random streams for label selection, model initialization and graph augmentations.
The same selected addresses and permitted validation labels apply to all four methods.
Training label masks cannot expand when a global label is subsequently used in validation.

Validation steps are used for inference, selection and fusion only, not encoder gradients
or SSL. Unselected known labels and unknown labels are masked in supervised losses.
Shared graph context is allowed within an observed snapshot; supervision uses selected
training addresses only. Each address–time pair contributes once to supervised loss.

## Matched graph comparison — accepted regime

The researcher selected fine-tuning of the encoder after SSL. Use identical encoder,
input contract and binary classification head for the SSL and no-SSL graph methods.
The head is a trainable affine map from address embedding to one illicit-class logit.

1. SSL branch: pretrain on all training snapshots without labels; discard the contrastive
   projection head; attach the classification head; optimize encoder and classifier with
   the permitted training labels.
2. No-SSL branch: initialize the same encoder from scratch; attach an identically
   initialized classification head for the paired run; optimize using the same supervised
   loss, label subset, optimizer policy, epoch cap and selection rule.
3. Use training-derived class weights consistently if enabled; fix the policy before
   evaluating test. Default supervised loss: binary cross-entropy with logits; weight
   the positive class using the selected training occurrence counts.
4. Restore the best validation checkpoint; gradients never use validation labels.
   Technical selection default: validation AP, avoiding a threshold search every epoch.
   Final model ranking remains the agreed test illicit-class F1.

Separate SSL initialization per seed can be reused across label fractions because it
does not consume their labels. Each fine-tuning run starts from an unchanged copy;
never continue the 5% run from a checkpoint already fitted on 1% labels. Record actual
pretraining and supervised cost: this is not an equal-total-compute comparison.

This adapts Inspection-L rather than reproducing its frozen embeddings + RF downstream.
There is no additional frozen-encoder experiment in the four-method principal matrix.

## Encoder and self-supervised objective — technical starting design

Use relation-specific message aggregation inspired by GIN, with separate address and
transaction input projections. Do not claim a theoretical homogeneous GIN expressivity
guarantee for this adaptation. Start with two synchronous message-passing layers and
128-dimensional hidden representations; laboratory memory measurements govern batching.

Retain sender and receiver roles and add separately typed computational reverse edges.
This lets a sender address receive context from its outgoing transactions as well as
letting receivers see incoming context. Reverse edges are not reverse BTC transfers and
never cross steps. Use one encoder shared across all snapshots.

Construct two views with feature masking and relation-preserving edge dropout. Drop
a physical edge and its computational reverse together. Match positives by the same
typed node ID in the same snapshot across views. Negatives have the same node type and
are distinct nodes; never use labels to choose views or negatives. Compute a standard
normalized temperature-scaled contrastive loss with a projection head, separately per
type, then average the two type losses so node-count imbalance does not decide weighting.
Sampled batches must retain aligned anchor IDs and sufficient distinct negative nodes.

Addresses represented in several snapshots are not assumed distinct entities for
cross-snapshot negatives: this design uses negatives within one snapshot only.
Rates, temperature, batch sizes and resource limits are fixed in
[run-config.md](contracts/run-config.md) before test.
No full-graph quadratic similarity matrix or whole-dataset GPU residency is required.

## Tabular reference

Random Forest uses the 123 address descriptors and the same labeled training pairs.
Technical starting point: 300 trees, balanced class weights, deterministic seed and a
bounded hyperparameter search specified in the plan. This classifier is also the tabular
component of fusion; the graph classifier is the fine-tuned neural head above.

## Late fusion and threshold selection

Reuse the fitted tabular and H-GCL models and their aligned illicit-class scores:

`p_fused = alpha * p_RF + (1 - alpha) * p_HGCL`.

Technical default: alpha in `{0, 0.05, ..., 1}`. For each alpha, choose a threshold
maximizing illicit F1 using only the selected validation labels; select the best pair.
Use the same threshold-selection procedure for standalone methods. Enumerate distinct
score cut points including constant-prediction endpoints. Specify deterministic ties:
closest threshold to 0.5, then lower threshold; alpha closest to 0.5, then lower alpha.
At inference use `score >= threshold`. Freeze all choices before test evaluation.

No additional calibration model in the initial design. Treat these as model scores;
do not claim calibrated risk probabilities. Validation performance is selection evidence,
not an unbiased final estimate. The fixed small alpha grid limits search flexibility;
it does not remove small-validation-set uncertainty.

At alpha 0 or 1, report that validation selected one branch. Do not force a nontrivial
fusion weight to obtain a favorable scientific narrative. Retain selected weights,
thresholds, checkpoint choices and all validation search results as run artifacts.

## Recurrent addresses and accepted complement

- D028: retain all eligible test address–time pairs, including recurrent addresses.
  Report the overall metrics plus seen/unseen strata. For this report, seen means an
  address occurred in development steps 1–34; unseen means absent from 1–34. Report
  additionally whether its label was actually used in training or validation. Presence
  in graph context and prior label exposure are distinct. Do not tune on stratum scores.
- Apply the analogous seen/unseen diagnostic to validation relative to training steps
  1–28, without changing its accepted label budget or model-selection population.
- Count training, validation and their union of selected labeled addresses, with the
  overlap explicit. Never copy validation-only labels into training masks. Global labels
  on recurrent addresses preclude claiming that the overall test contains only novel
  entities; the unseen stratum supports that narrower analysis.
- D029: four native-wallet methods at 100% labels and five seeds, 20 additional evaluations.
  With 80 principal evaluations, the total is 100. Both branches use the native wallet
  allowlist in input-contract.md; transaction content and graph relations stay the same.
  The comparison measures an input-policy change, not a pure causal effect of leakage.
- Exact byte/vbyte meaning of the published size field remains unverified. Retain the
  numeric property without fee-per-byte interpretation; pursue source mapping in planning.

## Validation for the subsequent plan

Check gradient updates of encoder and head after SSL, identical architecture in the
control, unchanged cached SSL checkpoints across label fractions, label-mask isolation,
no validation/test gradient updates, identical address keys across predictions, correctly
paired forward/reverse edge augmentation, alpha endpoints, deterministic ties, and one
frozen evaluation pass after selection. Smoke uses the same stages with bounded data
and training under 600 seconds, separately timed from preparation.

Reference: [Inspection-L, Sections 4.1–4.2](https://arxiv.org/html/2203.10465v4#S4.SS2),
BibTeX `lo2022inspectionl`. It supports the SSL inspiration; architecture, fine-tuning
and scalar late fusion here are S02 design choices, not attributed to that paper.

# Run configuration contract

YAML schema 1, typed dataclass validation; reject unknown keys. Resolve and hash before
fitting. Sections: paths, profile, split, labels, features, encoder, ssl, supervised, rf,
fusion, evaluation, resources, provenance. Paths include explicit project_root.

Scientific split 1–28/29–34/35–49; principal fractions .01/.05/.10/1; native_wallet 1;
methods rf/graph_supervised/hgcl/fusion; seeds 11/23/37/53/71. Validation uses the same
fraction on its own unique labeled population. Floor per-class nested hash selection,
no replacement or automatic class-budget inflation. Same masks across all methods.

| Setting | Smoke | Lab |
| --- | --- | --- |
| device | CPU | CUDA after doctor |
| hidden/layers | 32/2 | 128/2 |
| seed/fraction/regime | 11/1/principal | scientific matrix |
| steps train/validation/test | 1,2 /29 /35 | accepted full split |
| recut | 256 transactions per retained step | none |
| graph loading | full reduced snapshot | NeighborLoader, 10 then 5 per relation |
| SSL epochs | 2 | 100, fixed |
| SSL lr/decay | .001/.00001 | .001/.00001 |
| temperature/projection dim | .2/32 | .2/128 |
| numeric feature mask/edge dropout | .1/.1 | .1/.1 |
| supervised epoch cap/patience | 3/3 | 100/10 |
| supervised LR candidates | .001 | .001, .0003 |
| supervised decay/optimizer | .00001/Adam | .00001/Adam |
| anchor batch per type | 32 | 128 |
| RF trees/min-leaf candidates | 50 /1 | 300 /1,5 |
| RF max-features/class-weight | sqrt/balanced | sqrt/balanced |
| workers/RF jobs | 0/4 | 0/4 initially |
| alpha grid | 0..1 by .05 | 0..1 by .05 |
| fitting+selection timeout | 600 seconds | none; resource guards |
| RSS/GPU allocation guard | 8 GiB/n.a. | 24 GiB/4.5 GiB |
| sampled nodes/edges cap | 50k/200k | 50k/200k |

All floating-point model work float32 initially. Supervised positive BCE weight derives
from selected training occurrence counts; identical for both graph methods. Head is one
linear logit and sigmoid for scores. LayerNorm, no batch-statistic normalization.
Candidates/checkpoints selected by validation AP, ties earlier candidate then earlier
epoch; same search budget for both graph methods. SSL uses final fixed-epoch checkpoint.
F1 remains the primary reported test metric. Record candidate fits separately.

Recut: order tx IDs by SHA256 of UTF-8 `seed|step|txId`, take 256 per selected step,
retain all incident addresses/edges, recompute descriptors on the reduced snapshot.
No label-based test selection. Missing either known class in a smoke partition makes
it invalid. The initial size requires measurement, not a guarantee of 600 seconds.

Missing flags are not masked by augmentation. Pair physical/reverse edge dropout.
For full-graph smoke, compute loss only on anchor batches to avoid quadratic all-node
similarity. Exact inference chunks start at 4096 destinations/100,000 edges, aggregating
all neighbors; chunk changes must preserve fixture outputs within float32 tolerance.

Threshold candidates: distinct scores plus all-positive/all-negative endpoints; predict
score >= threshold. Max validation F1; ties closest threshold to .5 then lower threshold.
Fusion alpha ties closest to .5 then lower alpha. Persist finite sentinel above 1 for an
all-negative threshold when necessary (scores remain [0,1]); never serialize infinity.
No extra calibrator. A validation-selected alpha endpoint is a valid result.

Resource/NaN failures preserve diagnostics/checkpoint. New batch/recut limits create a new
resolved run; no silent topology/feature/class removal. Data/schema/source/config/mask
hashes protect resume and SSL cache reuse. Final test is excluded from every selection.

## Smoke GPU operacional — T030/D043

configs/smoke-gpu.yaml usa o mesmo recorte, splits, arquitetura pequena, épocas,
seed e fração do smoke CPU. Diferenças: CUDA, NeighborLoader com fanouts 10/5,
limite GPU 4.5 GiB/RSS 24 GiB e lock lab-cuda.lock. Continua limitado a 600 s,
classificado como engenharia e excluído da matriz de 100 avaliações científicas.

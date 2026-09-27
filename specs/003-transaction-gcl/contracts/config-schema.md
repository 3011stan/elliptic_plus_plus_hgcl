# Contract — S003 Configuration Schema v1

Todo arquivo em `configs/s003/` é YAML estrito: chaves desconhecidas, ausentes ou com tipo incorreto são erro. Overrides de CLI são limitados a caminhos, dispositivo e limites de engenharia; parâmetros científicos exigem novo arquivo versionado.

## Required top-level shape

```yaml
schema_version: 1
study_id: s003
task: transaction_classification
method_id: s003_txgcl
graph_schema: tx_tx
target_node_type: transaction
profile: smoke | dry-run | lab
paths: {}
data: {}
split: {}
labels: {}
features: {}
graph: {}
ssl: {}
downstream: {}
baselines: {}
evaluation: {}
statistics: {}
resources: {}
provenance: {}
```

## Invariants

- `split.development_steps` é exatamente `[1..34]`; `split.test_steps` é exatamente `[35..49]`. Perfis de engenharia declaram `shadow_test_steps` contidos em 1–34.
- `labels.seeds` é `[11,23,37,53,71]`; `labels.fractions` é `[0.01,0.05,0.10,1.00]`; `fit_ratio=0.8`.
- `features.model_count=183`, `features.maskable_count=182`; blocos têm tamanhos `[93,72,17]`; `txId` e `Time step` ficam excluídos do masking.
- `graph.directed=true`; snapshots e KNN não atravessam time steps.
- `ssl.encoder=gin`; `ssl.loss=gcpal_multi_positive`; `ssl.use_labels=false`.
- `downstream.encoder_frozen=true`; representação principal é `h_concat_x`.
- `evaluation.selection.primary=f1_illicit`; primeiro desempate é `mcc`.
- `statistics.test=paired_t_two_sided`, `correction=holm`, `alpha=0.05`, `effect_size=cohen_dz`.
- paths de artefato devem ficar sob `artifacts/s003/`; data root não pode estar dentro de artifacts nem vice-versa.
- `lab` não pode conter limites de recorte do smoke; `matrix` só aceita profile `lab` e aprovação do dry-run compatível.

## Profiles

- `smoke`: recorte determinístico de CSVs originais, duas épocas SSL, três downstream, uma seed de engenharia e shadow test em 1–34; não produz evidência científica.
- `dry-run`: uma seed aprovada, limites reduzidos declarados e pipeline completo com shadow test em 1–34; mede recursos e valida contratos, sem abrir 35–49 nem integrar o relatório científico.
- `lab`: dados completos, cinco seeds e protocolo científico congelado.

O config canônico é serializado com ordenação estável e recebe SHA-256. O digest integra todos os manifests e checkpoints.

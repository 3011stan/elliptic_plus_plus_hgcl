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
- `features.source_non_id_count=183`, `features.model_count=182`, `features.maskable_count=182`; blocos têm tamanhos `[93,72,17]`; `txId` e `Time step` ficam excluídos de `X_tx` e do masking.
- `graph.directed=true`; snapshots e KNN não atravessam time steps.
- `ssl.encoder=gin`, `ssl.layers=2`, `ssl.hidden=128`; `ssl.loss=gcpal_multi_positive`; `ssl.use_labels=false`; `ssl.epochs` é inteiro fixo no profile `lab` e não admite early stopping por label.
- `graph.edge_index_order=source_target` e `graph.message_flow=source_to_target`; o destino agrega mensagens da origem. `positives.structural_neighbors=successors`; somente a ablação P2 identificada pode declarar `add_reverse_edges=true`.
- `downstream.encoder_frozen=true`; representação principal é `h_concat_x`; seleção supervisionada usa F1 ilícito e MCC.
- `evaluation.selection.primary=f1_illicit`; primeiro desempate é `mcc`.
- `statistics.test=paired_t_two_sided`, `correction=holm`, `alpha=0.05`, `effect_size=cohen_dz`.
- paths de artefato devem ficar sob `artifacts/s003/`; data root não pode estar dentro de artifacts nem vice-versa.
- `lab` não pode conter limites de recorte do smoke; `matrix` só aceita profile `lab` e aprovação do dry-run compatível.
- `matrix.expected_p1_cells=205`; a variante bidirecional P2 acrescenta cinco células somente quando `approval.reverse_edge_ablation=true`.
- `dry_run` exige zero violações temporais, smoke ≤10 minutos de treinamento, VRAM do laboratório <4,5 GiB, RAM <24 GiB, margem de projeção de 20% e pausa para nova aprovação se a estimativa exceder a janela aceita em mais de 25%.
- `doctor` bloqueia o ambiente se versões, CUDA/dispositivo ou capacidade obrigatória divergirem; não existe fallback automático de GPU para CPU no profile `lab`.
- A seção supervisionada declara uma única política de prevalência fit-only, traduzida para pesos de loss, `class_weight` e `scale_pos_weight` conforme a família.
- Reexecução na mesma plataforma exige scores `allclose(rtol=1e-5, atol=2e-6)` e decisões idênticas; hashes e IDs são sempre exatos.

## Profiles

- `smoke`: `max_nodes_per_snapshot=256`, seleção por ordem de `SHA-256("s003-smoke" || tx_id)`, `engineering_fit_steps=[1..29]`, `shadow_test_steps=[30..34]`, `snapshot_batch_size=1`, duas épocas SSL, três downstream e uma seed de engenharia; não produz evidência científica.
- `dry-run`: todos os nós de 1–34, `engineering_fit_steps=[1..29]`, `shadow_test_steps=[30..34]`, `snapshot_batch_size=1`, seed 11, fração 0,01, dez épocas SSL, vinte downstream, paciência cinco, uma célula representativa dos nove métodos e quatro ablações adicionais; mede recursos e valida contratos sem abrir 35–49 nem integrar o relatório científico.
- `lab`: dados completos, cinco seeds e protocolo científico congelado.

`approval` segue o schema `DryRunApproval`; `evaluation` exige `cohort_release=single_global` e proíbe alteração de membros após a liberação.

O config canônico é serializado com ordenação estável e recebe SHA-256. O digest integra todos os manifests e checkpoints.

# Contract — S003 Artifacts

## Layout

```text
artifacts/s003/
├── prepared/<data_digest>/
│   ├── source-manifest.json
│   ├── preparation-manifest.json
│   ├── preprocessing.json
│   ├── feature-blocks.json
│   ├── evaluation-access.json
│   ├── budgets/<seed>.json
│   └── snapshots/<step>/{nodes.parquet,graph.pt,manifest.json}
├── runs/<run_id>/
│   ├── config.canonical.json
│   ├── run.json
│   ├── environment.json
│   ├── selection.json
│   ├── checkpoints/
│   ├── predictions/<step>.parquet
│   ├── metrics/<step>.json
│   └── diagnostics.json
├── matrices/<matrix_id>/
│   ├── design.json
│   ├── approval.json
│   ├── evaluation-cohort.json
│   ├── coverage.json
│   ├── results.parquet
│   ├── statistics.json
│   └── report.json
└── gates/hetero/<decision_id>.json
```

## Common envelope

Todo JSON possui:

```json
{
  "schema_version": 1,
  "study_id": "s003",
  "artifact_type": "...",
  "created_at": "UTC timestamp",
  "config_digest": "sha256",
  "data_digest": "sha256",
  "code_revision": {"commit": "...", "dirty": false},
  "payload": {}
}
```

## Integrity rules

- Paths registrados são relativos ao root de artifacts; cada arquivo material possui bytes e SHA-256.
- Arquivos são escritos em temporário no mesmo filesystem, sincronizados e renomeados atomicamente.
- Um artefato `completed` não pode ser sobrescrito.
- Checkpoints contêm época, modelo, otimizador, RNGs, config/data/code digests e versão das dependências.
- Predições preservam `tx_id`, `time_step`, label conhecido, score e decisão; a ordem não define identidade.
- Resultados pooled registram que as predições conhecidas de 35–49 foram concatenadas antes do cálculo; `f1_illicit` e `micro_f1` são campos distintos e o segundo é secundário quando reportado.
- `coverage.json` enumera todas as células esperadas e classifica cada uma como completed/interrupted/invalid/failed/missing.
- `evaluation-access.json` não contém labels; registra o digest do store selado e cada abertura autorizada dos rótulos 35–49.
- `approval.json` segue `DryRunApproval` e vincula dados, config, código, evidências e design; divergência falha fechado.
- O pacote do dry-run estrutural contém `design.json`, plano de cache/checkpoint/retomada, diagnóstico de capacidade e projeção documentada; declara `training_performed=false` e `test_labels_materialized=false` e não contém checkpoints, predições ou métricas de modelo.
- `evaluation-cohort.json` contabiliza as 205 células P1, sela o subconjunto selecionado antes do teste e registra uma única transição global de `sealed` para `released`; falhas terminais permanecem explícitas e acessos físicos são auditados por membro.
- Um `technical_rerun` registra o run original e a causa técnica; divergência de qualquer digest, peso ou threshold invalida essa classificação.
- Manifests pequenos e relatórios destinados à dissertação podem ser exportados para localização versionada; checkpoints, Parquet volumoso e predições permanecem fora do Git.

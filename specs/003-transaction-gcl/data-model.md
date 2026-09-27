# Data Model — S003-TxGCL

## 1. SourceDataset

Representa os nove CSVs originais somente leitura.

| Campo | Tipo | Regra |
|---|---|---|
| `dataset_id` | string | `elliptic-plus-plus` |
| `root` | path | fora de `artifacts/s003`; nunca gravado |
| `files` | map | nome, bytes e SHA-256 de cada CSV |
| `schema_version` | integer | inicia em `1` |
| `verified_at` | timestamp | UTC |

Validação: todos os arquivos do manifest devem existir, ter hash correspondente e schema esperado antes da preparação.

## 2. Transaction

Unidade de predição e nó do grafo.

| Campo | Tipo | Regra |
|---|---|---|
| `tx_id` | integer/string externo | único; nunca índice tensorial implícito |
| `node_index` | integer local | contíguo apenas dentro do snapshot; mapeamento persistido |
| `time_step` | integer | `[1,49]`; disponível, não mascarável |
| `local_features` | float[93] | finitos após preprocessing |
| `aggregate_features` | float[72] | finitos após preprocessing |
| `augmented_features` | float[17] | finitos após preprocessing |
| `class` | enum | `illicit`, `licit`, `unknown` |

`model_attributes` contém `time_step` e 182 atributos financeiros. `maskable_features` e `knn_features` contêm somente os 182 atributos financeiros.

## 3. TransactionEdge

| Campo | Tipo | Regra |
|---|---|---|
| `source_tx_id` | external ID | deve existir em Transaction |
| `target_tx_id` | external ID | deve existir em Transaction |
| `time_step` | integer | igual no source e target |

Duplicatas são removidas de forma determinística e contabilizadas no manifest. A direção original é preservada; transformações que adicionem reversas devem declará-las na configuração do método.

## 4. TemporalSnapshot

| Campo | Tipo | Regra |
|---|---|---|
| `time_step` | integer | chave única |
| `tx_ids` | array | ordenação canônica persistida |
| `x_model` | tensor `[N,183]` | sem identificador |
| `x_financial` | tensor `[N,182]` | entrada de masking/KNN |
| `edge_index` | tensor `[2,E]` | índices locais válidos e intrassnapshot |
| `labels` | tensor `[N]` ou null | `1`, `0`, `-1` em 1–34; null na visão de desenvolvimento de 35–49 |
| `digest` | SHA-256 | cobre IDs, features, labels e arestas |

Relacionamento: SourceDataset 1→49 TemporalSnapshots; Transaction pertence a exatamente um snapshot.

## 5. PreprocessingState

| Campo | Tipo | Regra |
|---|---|---|
| `fit_steps` | list[int] | exatamente `1..34` |
| `feature_names` | list[string] | ordem canônica |
| `center` | float[182] | calculado somente em 1–34 |
| `scale` | float[182] | zero scale tratado explicitamente |
| `time_step_divisor` | integer | `34`, sem consulta a 35–49 |
| `policy` | enum | transformação declarada no config |
| `digest` | SHA-256 | obrigatório |

O estado é somente aplicado em 35–49; nunca reajustado. O time step normalizado pode exceder 1 no teste, o que é registrado e não provoca clipping ajustado pelo teste.

## 5A. TestLabelStore

Artefato selado que mapeia `tx_id` de 35–49 para classe. Não é referenciado pelo `PreparedDataset` oferecido a treino, validação, dry-run ou seleção. Somente um `EvaluationAccessGuard` no comando `evaluate` pode abri-lo; a abertura registra run, revisão, horário, seleção e threshold congelados.

## 6. LabelBudget

| Campo | Tipo | Regra |
|---|---|---|
| `seed` | enum | `11`, `23`, `37`, `53`, `71` |
| `fraction` | enum | `0.01`, `0.05`, `0.10`, `1.00` |
| `fit_ids` | set[tx_id] | passos 1–34, classes conhecidas |
| `validation_ids` | set[tx_id] | disjunto de fit, passos 1–34 |
| `refit_ids` | set[tx_id] | união exata de fit e validation |
| `class_counts` | map | ambas as classes em cada subconjunto |
| `parent_fraction` | fraction/null | IDs devem ser subconjunto do próximo budget |
| `digest` | SHA-256 | obrigatório |

Validação: estratificação 80/20, nesting por seed e nenhuma classe `unknown`.

## 7. FunctionalBlockMap

| Campo | Tipo | Regra |
|---|---|---|
| `schema_version` | integer | `1` |
| `local` | list[int/name] | 93 posições |
| `aggregate` | list[int/name] | 72 posições |
| `augmented` | list[int/name] | 17 posições |
| `excluded` | list[string] | `txId`, `Time step` |
| `digest` | SHA-256 | versionado |

Os três blocos são disjuntos e cobrem exatamente as 182 features financeiras.

## 8. RandomGroupMap

Partição derivada deterministicamente de `FunctionalBlockMap` e da seed, com tamanhos `[93,72,17]`. Não consulta classes. Seu digest integra o run manifest.

## 9. ContrastiveView / PositiveSet

`ContrastiveView` registra política, seed/RNG state, máscara de features, arestas removidas e snapshot de origem. `PositiveSet(anchor)` é a união deduplicada de self, vizinhos `Tx→Tx` e vizinhos KNN. Deve ser disjunto de `NegativeSet(anchor)` e nunca atravessar snapshots.

## 10. ExperimentConfiguration

Configuração validada segundo [contracts/config-schema.md](contracts/config-schema.md). Campos identitários obrigatórios: `schema_version`, `study_id`, `task`, `method_id`, `graph_schema`, `target_node_type`, `profile`, split, seed, fraction e hashes das seções científicas.

## 11. ExperimentRun

| Campo | Tipo | Regra |
|---|---|---|
| `run_id` | string | prefixo `s003-`, único e imutável |
| `config_digest` | SHA-256 | obrigatório |
| `data_digest` | SHA-256 | obrigatório |
| `code_revision` | git SHA + dirty flag | obrigatório |
| `state` | enum | conforme máquina abaixo |
| `started_at`, `ended_at` | timestamp/null | UTC |
| `durations` | map | prepare/pretrain/downstream/evaluate |
| `environment` | object | SO, Python, libs, CPU/GPU/RAM |
| `artifacts` | map | path relativo + hash |
| `failure` | object/null | categoria e mensagem |

Transições permitidas:

```text
planned -> running -> selected -> evaluating -> completed
                  -> interrupted -> fase registrada
                  -> invalid
                  -> failed
```

Estados `completed`, `invalid` e `failed` são terminais. `evaluate` só aceita `selected`.

## 12. ModelSelection

Registra todos os candidatos, F1 ilícito, MCC, threshold, época e motivo da escolha. Ordenação: maior F1 ilícito, depois maior MCC, depois chave canônica de hiperparâmetros para desempate determinístico. O threshold é congelado antes do teste.

## 13. EvaluationResult

| Campo | Tipo | Regra |
|---|---|---|
| `run_id` | referência | execução `completed` e congelada |
| `snapshot` | int/`aggregate` | 35–49 ou agregado |
| `support` | map | ilícita, lícita, desconhecida excluída |
| `threshold` | float | vindo de ModelSelection |
| `metrics` | map | MCC, F1/precision/recall ilícito, PR-AUC; secundárias marcadas |
| `predictions_digest` | SHA-256 | obrigatório |

Métrica indefinida por ausência de classe é `null` com razão, nunca zero inventado.

## 14. StatisticalComparison

Contém método proposto, baseline, fração, métrica, cinco diferenças pareadas, média, desvio padrão, IC 95%, t, p bruto, p Holm, Cohen's `d_z` e interpretação limitada. Existem exatamente quatro comparações primárias por métrica.

## 15. HeterogeneousExtensionDecision

| Campo | Tipo | Regra |
|---|---|---|
| `causality_gate` | pass/fail | evidência vinculada |
| `supervision_gate` | pass/fail | evidência vinculada |
| `comparability_gate` | pass/fail | evidência vinculada |
| `resources_gate` | pass/fail | evidência vinculada |
| `decision` | include/defer | `include` somente com quatro passes |
| `approved_by` | string/null | aceite explícito para include |

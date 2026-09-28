# Data Model — S003-TxGCL

## 1. SourceDataset

Representa os CSVs originais somente leitura. O núcleo `Tx→Tx` exige apenas `txs_features.csv`, `txs_classes.csv` e `txs_edgelist.csv`; os seis arquivos de endereço permanecem registrados quando disponíveis, mas só são obrigatórios para o gate/extensão `Addr↔Tx`.

| Campo | Tipo | Regra |
|---|---|---|
| `dataset_id` | string | `elliptic-plus-plus` |
| `root` | path | fora de `artifacts/s003`; nunca gravado |
| `files` | map | nome, bytes e SHA-256 de cada CSV |
| `schema_version` | integer | inicia em `1` |
| `verified_at` | timestamp | UTC |

Validação: os três arquivos do núcleo devem existir, ter hash correspondente e schema esperado antes da preparação homogênea. Ausência de um arquivo de endereço não bloqueia o núcleo e é registrada como `hetero_unavailable`; arquivo presente deve corresponder ao manifest antes de qualquer uso.

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

`source_attributes` contém `time_step` e 182 features financeiras. `model_features`, `maskable_features` e `knn_features` contêm somente as 182 features financeiras.

## 3. TransactionEdge

| Campo | Tipo | Regra |
|---|---|---|
| `source_tx_id` | external ID | deve existir em Transaction |
| `target_tx_id` | external ID | deve existir em Transaction |
| `time_step` | integer | igual no source e target |

Duplicatas são removidas de forma determinística e contabilizadas no manifest. A direção original é preservada; transformações que adicionem reversas devem declará-las na configuração do método.

IDs duplicados, referências ausentes e arestas entre time steps invalidam a preparação. Self-loops de entrada são removidos e contabilizados; a contribuição própria do GIN é responsabilidade do operador, não do CSV.

## 4. TemporalSnapshot

| Campo | Tipo | Regra |
|---|---|---|
| `time_step` | integer | chave única |
| `tx_ids` | array | ordenação canônica persistida |
| `x_model` | tensor `[N,182]` | entrada comum de encoder, downstream, masking e KNN |
| `edge_index` | tensor `[2,E]` | linha 0 = source, linha 1 = target; índices locais válidos e intrassnapshot |
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
| `policy` | enum | transformação declarada no config |
| `digest` | SHA-256 | obrigatório |

O estado é somente aplicado em 35–49; nunca reajustado. `Time step` não integra o estado porque não é feature de modelo.

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

`ContrastiveView` registra política, seed/RNG state, máscara de features, arestas removidas e snapshot de origem. Existem somente duas representações aumentadas, estocástica e por blocos; KNN apenas expande positivos. `PositiveSet(anchor)` é a união deduplicada de self, sucessores estruturais para os quais `anchor` é `source_tx_id` e vizinhos KNN. Predecessores não entram automaticamente. O conjunto deve ser disjunto de `NegativeSet(anchor)` e nunca atravessar snapshots.

Para snapshot com `N` nós, KNN usa `min(K,N-1)`. Nó isolado permanece representável. Âncora sem negativo elegível não contribui à loss; zero âncoras válidas invalida a época/run.

## 10. ExperimentConfiguration

Configuração validada segundo [contracts/config-schema.md](contracts/config-schema.md). Campos identitários obrigatórios: `schema_version`, `study_id`, `task`, `method_id`, `graph_schema`, `target_node_type`, `profile`, split, seed, fraction e hashes das seções científicas.

## 11. ExperimentRun

| Campo | Tipo | Regra |
|---|---|---|
| `run_id` | string | prefixo `s003-`, único e imutável |
| `config_digest` | SHA-256 | obrigatório |
| `data_digest` | SHA-256 | obrigatório |
| `code_revision` | git SHA + dirty flag | obrigatório |
| `state` | enum | conforme máquina abaixo, incluindo `technical_rerun` de avaliação |
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

Um `technical_rerun` não é uma nova seleção: referencia o run original e exige igualdade de config, dados, pesos, threshold e revisão. Qualquer diferença cria análise exploratória separada.

## 11A. DryRunApproval

| Campo | Tipo | Regra |
|---|---|---|
| `approval_id` | string | prefixo `s003-`, único |
| `approved_by` | string | identidade declarada do pesquisador; não vazio |
| `approved_at` | timestamp | UTC |
| `data_digest` | SHA-256 | igual ao dry-run e à matriz |
| `dry_run_config_digest` | SHA-256 | config usada para produzir as evidências |
| `lab_config_digest` | SHA-256 | config científica congelada que será executada |
| `code_revision` | git SHA + dirty flag | igual ao pacote revisado |
| `evidence_digest` | SHA-256 | cobre relatório, auditorias e projeção |
| `design_digest` | SHA-256 | cobre as 205 células P1 |
| `max_projected_duration_seconds` | integer | janela máxima aceita, positiva |
| `reverse_edge_ablation` | boolean | autorização P2 separada; default false |

O artefato é imutável. Qualquer divergência de digest, expiração/revogação registrada ou tentativa de uso com outro design bloqueia `matrix` com código 4.

## 11B. EvaluationCohort

| Campo | Tipo | Regra |
|---|---|---|
| `cohort_id` | string | prefixo `s003-`, único |
| `design_digest` | SHA-256 | exatamente o design aprovado |
| `cells` | list | exatamente 205 chaves P1, cada uma `selected` ou em falha terminal explícita |
| `members` | list | subconjunto `selected`; cada membro contém run, pesos, threshold e digests congelados |
| `sealed_at` | timestamp | anterior à abertura do teste |
| `release_state` | enum | `sealed`, `released`, `completed`, `interrupted` |
| `released_at` | timestamp/null | preenchido uma única vez |
| `test_store_digest` | SHA-256 | identidade do store selado, sem labels |

Somente uma coorte `sealed`, com as 205 células contabilizadas e ao menos um membro selecionado, pode ser liberada. Células em falha terminal permanecem na cobertura e não são inventadas como membros avaliáveis. Após `released`, membros não podem ser adicionados, removidos ou novamente selecionados; retomada percorre somente membros ainda não concluídos com os mesmos digests.

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

O objeto somente pode ter estado `inferential` com cinco pares completos e definidos; caso contrário é `descriptive_incomplete` e os campos `ci`, `t`, `p` e `effect_size` são `null` com razão.

## 15. HeterogeneousExtensionDecision

| Campo | Tipo | Regra |
|---|---|---|
| `causality_gate` | pass/fail | evidência vinculada |
| `supervision_gate` | pass/fail | evidência vinculada |
| `comparability_gate` | pass/fail | evidência vinculada |
| `resources_gate` | pass/fail | evidência vinculada |
| `decision` | include/defer | `include` somente com quatro passes |
| `approved_by` | string/null | aceite explícito para include |

# Status — Estudo 003

Atualizado em: 2026-09-28

## Estado atual

- Proposta científica consolidada.
- Decisões pré-SDD registradas.
- Branch dedicada: `003-transaction-gcl`.
- Estudo 002 congelado na tag `s02-001-final`.
- Namespaces do Estudo 003 preparados.
- SDD iniciado: `specs/003-transaction-gcl/spec.md` criado e validado.
- Clarificação concluída com cinco decisões incorporadas à especificação.
- Planejamento técnico concluído em `specs/003-transaction-gcl/plan.md`, com pesquisa,
  modelo de dados, contratos e quickstart de validação.
- Checklist pós-plano gerado em
  `specs/003-transaction-gcl/checklists/scientific-protocol.md` e concluído após
  revisão das decisões com o NotebookLM autorizado.
- Decomposição da implementação revisada em
  `specs/003-transaction-gcl/tasks.md`: 65 tarefas rastreáveis, das quais T001–T050
  estão concluídas e T051–T065 permanecem abertas. O agrupamento reduz a
  fragmentação operacional sem retirar métodos, frações, seeds, ablações,
  estatística, proveniência ou qualquer uma das 205 células P1.
- Primeira análise cruzada encontrou cinco bloqueios de especificação; as correções
  aceitas foram incorporadas: limites exatos de engenharia, duas representações
  com KNN apenas como expansão de positivos, semântica `source_to_target`, coorte
  única de avaliação e aprovação do dry-run vinculada por digests.
- A repetição da análise cruzada após a remediação não encontrou requisito sem
  tarefa, tarefa sem rastreabilidade, conflito constitucional ou ambiguidade
  bloqueante remanescente.
- Em 2026-09-28, o pesquisador substituiu o dry-run treinado de uma seed por um
  dry-run estrutural. O smoke continua treinando ponta a ponta; o dry-run passa a
  validar design, identidades, recursos, retomada e projeção sem treino ou
  inferência. O protocolo `lab` usa exatamente 100 épocas SSL.
- T042 reconciliou os contratos executáveis: comandos `smoke` e `dry-run` são
  distintos, o dry-run usa zero épocas e não cria artefatos de modelo, e o profile
  `lab` exige exatamente 100 épocas SSL. A suíte S003 completa possuía 62 testes
  aprovados naquele checkpoint; a declaração anterior de disponibilidade local do
  XGBoost estava incorreta e foi corrigida abaixo.
- T043/T044 completaram e validaram a orquestração US1, incluindo reutilização
  segura da preparação entre profiles, auditoria por hashes, design estrutural de
  205 células sem treino e bloqueio de avaliação antes do registry. A suíte S003
  passou com 64 testes; Ruff não integra o toolchain nem o processo do S02 e não
  foi acrescentado ao escopo.
- T045 implementou o registry exato dos nove métodos, contrato comum dos adapters,
  invariantes de comparação, prevalência fit-only e marcação de `approximation`.
  A suíte S003 passou com 69 testes.
- Após a instalação autorizada de `xgboost==3.2.0` e do runtime OpenMP no Mac,
  T046 implementou MLP X-only, Random Forest e XGBoost com seleção comum por F1
  ilícito/MCC e desbalanceamento calculado somente no fit, inclusive no refit. Os
  defaults científicos permanecem RF 300 árvores e XGBoost até 500 árvores; os
  testes de engenharia usam limites menores. A suíte S003 passou com 73 testes.
- T047 implementou adapters supervisionados GCN, GraphSAGE e GIN, todos com duas
  camadas, dimensão 128, propagação PyG `source_to_target`, pesos de classe
  derivados somente do fit e seleção de threshold por F1 ilícito/MCC. A suíte
  S003 passou com 77 testes.
- T048 implementou as adaptações Inspection-L (GIN 2×128, DGI e RF 100) e
  GCPAL (GIN 2×128, duas visões estocásticas, KNN 10, perda multi-positivo e MLP
  `H‖X_tx`), com a migração de 166 para 182 features declarada. A suíte S003
  passou com 80 testes.
- T049 implementou S003-TxGCL, `X-only`, `H-only`, `H‖X_tx`, os controles de
  mascaramento pareados, variantes sem KNN/edge dropout e o gate de arestas
  reversas. A suíte S003 passou com 87 testes.
- T050 implementou e testou o design canônico de 205 células P1 e cinco células P2
  opcionais, parsing semântico de cell keys, identidades imutáveis de cache de
  embeddings (reúso exato por frações e por H-only, totalizando exatamente 35
  pré-treinos SSL únicos em P1), compatibilidade da aprovação vinculada a digests
  com autorização estrita de arestas reversas (P2), e os planos estruturais de
  cobertura (SC-001/SC-005), cache, checkpoints atômicos, retomada estrita,
  recursos e projeção de duração (+20% de margem) para o dry-run. A suíte S003
  passou com 109 testes.
- T051 implementou e testou agendamento retomável da matriz (`MatrixScheduler.from_dict`
  e `to_dict`), estados explícitos e imutáveis de falha (`failed`, `invalid`),
  reúso atômico em memória e em disco do store de cache de embeddings (`EmbeddingCacheStore`),
  execução integrada de células da matriz (`execute_matrix_cell`) com reutilização de
  embeddings entre frações e por `H-only`, zero acesso aos rótulos de teste 35–49 antes da
  avaliação (`test_labels.access_log == []`) e selagem de coorte de avaliação com
  contabilização estrita de 205 células. A suíte S003 passou com 113 testes (179 na
  suíte global).
- T052 implementou e testou diagnósticos de representação em `diagnostics.py`
  (alinhamento, uniformidade esférica, posto efetivo de Roy & Vetterli via SVD, agregações
  ponderadas por âncoras em snapshots e agregação com média/std entre seeds, sem threshold
  universal arbitrário conforme FR-025), análise estatística pré-declarada em
  `statistics.py` com teste t bilateral pareado com 4 graus de liberdade para as 5 seeds
  canônicas `[11, 23, 37, 53, 71]`, intervalo de confiança exato de 95%, Cohen's d_z,
  correção step-down de Holm-Bonferroni para a família de 4 comparações primárias
  (S003-TxGCL vs GCPAL e vs Inspection-L em 1% e 5%), comportamento estrito com pares
  incompletos (FR-044: puramente descritivo, sem p-value/IC/Cohen's d_z nem alegação de
  superioridade), gates estritos para alegações científicas (FR-045: média > 0, IC 95% > 0,
  Holm p < 0.05, direção positiva em >= 4 seeds e ausência de degradação significativa na
  outra métrica primária; "eficiente em rótulos" condicionado a 1% ou 5%; rejeição de
  rótulos binários de robustez temporal e de "publicável"), atribuição quantitativa de
  ablação para representação (H||X vs H, H||X vs X, H vs X) e mecanismos (KNN, edge dropout,
  mascaramento individual e grupos de mesma cardinalidade), e cálculo de série temporal
  independente por snapshot em 35–49 com suporte gracioso a classes ausentes. A suíte S003
  passou com 125 testes (191 na suíte global).
- T053 implementou e testou a integração dos comandos CLI `approve-dry-run`, `matrix` e `report`
  em `cli.py` e `pipeline.py`, incluindo validação fail-closed estrita de compatibilidade de
  digests (`data_digest`, `dry_run_config_digest`, `lab_config_digest`, `evidence_digest`,
  `design_digest`), busca e verificação de aprovação prévia para o profile lab, execução de
  células em fixture reduzido não-científico com zero acesso aos rótulos 35–49
  (`test_labels.access_log == []`), persistência atômica de envelopes versionados e geração
  do relatório consolidado com contabilização explícita de células (SC-001, SC-004, SC-005).
- T055 implementou e testou os manifestos de linhagem (`SourceManifest`, `EnvironmentManifest`,
  `SelectionManifest`, `EvaluationManifest`), máquina de estados estrita (`RunStateMachine`) com
  transições atômicas e imutabilidade de estados terminais (`completed`, `failed`, `invalid`),
  categorização padronizada de falhas (`data`, `resource`, `runtime`, `preemption`, `contract`, `fixture`)
  e checkpoints reprodutíveis (`RunCheckpoint`) persistidos atomicamente via arquivo temporário e renomeação,
  com captura e restauração determinística de estados de RNG do PyTorch, NumPy e Python random (FR-004,
  FR-026–FR-027, FR-042, SC-006). A suíte S003 passou com 134 testes (200 na global).
- T056 implementou e testou retomada estrita (somente runs no estado `interrupted` podem retomar; estados
  terminais e planned/running rejeitam resume), equivalência numérica e de decisão ao recarregar checkpoints
  conforme FR-042 (`rtol=1e-5`, `atol=2e-6`, com 100% de concordância de classes no threshold congelado via
  `check_reload_equivalence` e `assert_reload_equivalence`), auditoria detalhada de acesso aos rótulos de teste
  em `TestLabelStore.access_log` com timestamp UTC, identidade do acessor e propósito, e garantia pós-descegamento
  de que reruns técnicos (`technical_rerun_of`) exigem identificador original, justificativa técnica e proíbem
  qualquer alteração de pesos, threshold ou configuração (sem reseleção conforme FR-038, FR-042, SC-003).
  A suíte S003 passou com 140 testes (206 na global).
- T057 implementou e testou linhagem por linha em `report.json`, cobertura explícita de estados incompletos
  em `coverage.json` (classificando 100% das células como selected, failed, invalid, interrupted ou planned
  com preservação de tipo e mensagem de falha, impedindo estados incompletos de serem mascarados como sucesso),
  o comando CLI `hgcl-s003 resume --run PATH` com saída estruturada em JSON e avanço de transições, e o
  cenário ponta a ponta de reconstrução arbitrária de resultado (`reconstruct_result_lineage`) rastreando
  dados, config, código, ambiente, semente, pesos, threshold, digest de labels de teste e métricas pooled/snapshot
  sem consultar estado do Estudo 002 (FR-026–FR-027, SC-001, SC-006).
- T058 validou a Fase 5 (User Story 3), com aprovação de 100% dos testes de US3 e comprovação de imutabilidade
  de artefatos concluídos (`ArtifactStore` rejeita sobrescrita de arquivos existentes, `EvaluationCohort`
  proíbe adição de células após selagem e nova liberação após unblinding, e `RunStateMachine` bloqueia
  qualquer transição a partir de estados terminais).
- T059 implementou o contrato formal e testes de integração não-bloqueantes para a extensão heterogênea
  `Addr↔Tx` em `src/hgcl/studies/s003/hetero/gate.py` e `tests/s003/contract/test_hetero_gate_contract.py`,
  cobrindo 100% dos quatro gates: causalidade (rejeição de atributos de endereço com agregação global ou vazamento
  futuro e arquivos ausentes), supervisão (alvo estritamente transação `tx` sem rótulos de carteira na perda),
  comparabilidade (identidade estrita de splits 1..34/35..49, frações 1%/5%/10%/100% e seeds 11/23/37/53/71) e
  recursos (projeção de RAM e duração dentro dos limites do hardware).
- T060 implementou o inventário de evidências de endereço, verificações causais/supervisão/target/recursos,
  a entidade persistida `HeterogeneousExtensionDecision` e a integração com o CLI `hgcl-s003 hetero-gate`,
  garantindo decisão rastreável de inclusão (`include`) ou adiamento (`defer`), mantendo o núcleo homogêneo
  desbloqueado e sem criar encoders ou loops de treino heterogêneos fora de escopo (FR-030–FR-032, SC-008).
  A suíte S003 passou com 151 testes (217 na global).
- T061 atualizou o `README.md`, `docs/studies/s003/context.md` e este documento com a documentação consolidada
  de uso do S003, exemplos de comandos CLI, ciclo de vida dos artefatos, rastreabilidade de requisitos e
  fronteira estrita de isolamento com o Estudo 002 (FR-026, FR-033–FR-034, SC-006).
- T062 validou a suíte completa com 218 testes aprovados e 2 skipped em dados históricos de S02,
  contratos de isolamento, auditoria de caminhos congelados do S02 com zero diferenças em relação à tag
  `s02-001-final`, ausência total de imports de `hgcl.*` no S003 e `git diff --check` limpo.
- D003-017 formalizou o tratamento de valores nulos nas 17 features aumentadas (965 transações, ~0,47% do total,
  decorrentes dos nós não desanonimizados segundo Elmougy & Liu, 2023) por preenchimento com constante `0.0`
  (zeros estruturais no protocolo UTXO), conforme recomendação da literatura no NotebookLM autorizado,
  assegurando entradas finitas e ausência de NaNs no pré-processamento fit-only e no modelo.
- T063 executou com sucesso o smoke test treinado em ambiente Mac (`hgcl-s003 smoke`) com dataset preparado
  sob digest `44940a242cf76682d1afc2874afb1418fa82aa43b42735f6b7cfd73d8fa154fb`, concluindo o treinamento
  e inferência do shadow test (passos 30–34) em aproximadamente 36 segundos (critério ≤ 10 minutos plenamente
  atendido), gerando `artifacts/s003/runs/s003-smoke-001/run.json` com `engineering_only: true`, métricas
  pooled e por snapshot e zero acesso físico aos rótulos de teste 35–49 (`test_labels_opened: false`,
  `test_label_accesses: 0`).
- T064 executou o `doctor` de laboratório e o dry-run estrutural sem treino (`hgcl-s003 dry-run`): o `doctor`
  comprovou fail-closed estrito perante os limites de recursos de laboratório (exigência de CUDA, 4,5 GiB VRAM
  e 24 GiB RAM); o dry-run estrutural validou exatamente 205 células P1, `ssl.epochs=100` no template lab
  (`configs/s003/lab.template.yaml`), zero treino (`training_performed=false`), zero inferência
  (`inference_performed=false`), zero materialização de rótulos de teste (`test_labels_materialized=false`),
  planos completos de cache (35 pré-treinos SSL únicos reutilizados), checkpoints atômicos, retomada estrita
  (`interrupted` apenas) e projeção de duração formal com margem de 20%, persistindo `run.json` e `design.json`
  em `artifacts/s003/runs/s003-dry-run-001/`.

- T065 concluiu o ciclo formal de aceite e aprovação: após autorização explícita do pesquisador
  e double check com o notebook NotebookLM (`66fb9e95-d225-4eaf-b45b-f8d232053677`), materializou
  `configs/s003/lab.yaml` com 100 épocas SSL imutáveis e aprovação configurada, executou com sucesso
  `hgcl-s003 approve-dry-run` gerando `artifacts/s003/runs/s003-dry-run-001/approval.json` vinculado
  aos digests de dados (`44940a242cf76682d1afc2874afb1418fa82aa43b42735f6b7cfd73d8fa154fb`), lab config
  (`ccff3e2177ab48ae52dd2380b5dfdd7e99653f127aad09b353bf2c07db49c9b7`), evidências
  (`45fcb6fb3a66c76cba3eab61627e3a11259893ce3396365818aa5bb914c7cebc`) e design
  (`0a76f84623ae452aeac389e4e0633bb86901c311394b71eb1929650f2cba25c8`), e revalidou 100% dos checks de
  release (218 testes aprovados, zero acessos a 35–49 e caminhos do S02 intocados).

**Checkpoint Final Atingido**: O software, a metodologia e o protocolo do Estudo 003 (`S003-TxGCL`) estão 100%
implementados, validados e aprovados. Todas as tarefas T001–T065 foram concluídas. Nenhuma célula científica da matriz
foi executada automaticamente; a matriz de laboratório permanece congelada e pronta para execução no ambiente com GPU.

## Próximo passo autorizado

Implementação e validação concluídas integralmente. A execução da matriz científica completa no ambiente de laboratório
com GPU CUDA (`hgcl-s003 matrix --config configs/s003/lab.yaml ...`) cabe à operação deliberada do pesquisador.

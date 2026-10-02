# Tasks: S003-TxGCL Transaction Classification

**Input**: Design documents from `/specs/003-transaction-gcl/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Obrigatórios devido aos riscos científicos de leakage, pareamento contrastivo, alinhamento de classes, proveniência e reprodução.

**Organization**: Tasks are grouped by user story so each story can be implemented and validated as an independent increment. Every task stays inside the S003 namespaces unless it explicitly adds the isolated `hgcl-s003` entry point or dependency metadata.

**Revision 2026-09-28**: T001–T041 record completed work. The remaining work was regrouped into larger, verifiable increments after the researcher replaced the trained one-seed dry-run with a structural dry-run. This revision does not reduce the scientific matrix, methods, ablations, seeds, fractions, statistics, provenance, or test isolation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode ser executada em paralelo quando seus predecessores estiverem completos e não houver conflito de arquivo.
- **[Story]**: associa a task à user story da especificação.
- Referências `FR-*` e `SC-*` indicam o requisito verificável coberto.

## Phase 1: Setup — Isolated S003 Surface

**Purpose**: Criar a superfície isolada do estudo sem alterar comportamento ou configuração do S02.

- [X] T001 Add the `hgcl-s003` console entry point plus direct `scipy==1.17.1` and `xgboost==3.2.0` dependencies in `pyproject.toml` — FR-021, FR-033, FR-046
- [X] T002 Refresh and verify the exact Python 3.11 dependency versions declared in the plan for macOS CPU and laboratory CUDA in `requirements/mac-cpu.lock`, `requirements/lab-cuda.in`, and `requirements/lab-cuda.lock` — FR-042, FR-046
- [X] T003 [P] Create the planned S003 module skeleton and `hetero` subpackage in `src/hgcl/studies/s003/` without importing S02 configuration or model modules — FR-033–FR-034
- [X] T004 [P] Create strict profile skeletons `configs/s003/smoke.yaml`, `configs/s003/dry-run.yaml`, and `configs/s003/lab.template.yaml` with the exact engineering sample/batch/epoch limits, `study_id=s003`, and no S02 keys — FR-001, FR-028, FR-033
- [X] T005 [P] Create `contract`, `unit`, and `integration` test package structure under `tests/s003/` and shared S003-only fixture declarations in `tests/s003/conftest.py` — FR-033
- [X] T006 [P] Ensure `artifacts/s003/`, prepared tensors, predictions, and checkpoints are ignored while small exported evidence remains allow-listed in `.gitignore` — FR-004, FR-026, FR-033

**Checkpoint**: O entry point e os namespaces do S003 existem, mas nenhum pipeline científico está implementado.

---

## Phase 2: Foundational — Blocking Contracts and Infrastructure

**Purpose**: Implementar contratos compartilhados que bloqueiam todas as user stories.

**⚠️ CRITICAL**: Nenhuma implementação de user story começa antes deste checkpoint.

- [X] T007 [P] Write strict config contract tests for required keys, exact smoke/dry-run limits, S02-key rejection, 182 model features, `source_to_target` flow, successor positives, fixed SSL epochs, 205 P1 cells, and compatible lab approval in `tests/s003/contract/test_config_contract.py` — FR-001, FR-003, FR-017, FR-028, FR-033, FR-037, FR-046
- [X] T008 Implement canonical S003 config parsing, exact profile-limit validation, approval/cohort invariants, overrides, and SHA-256 serialization in `src/hgcl/studies/s003/config.py` to satisfy T007 — FR-001, FR-026, FR-028–FR-029, FR-033, FR-038, FR-046
- [X] T009 [P] Write entity and state-transition tests for `SourceDataset`, `TemporalSnapshot`, `LabelBudget`, `ExperimentRun`, and `StatisticalComparison` in `tests/s003/unit/test_domain.py` — FR-017–FR-018, FR-027, FR-044
- [X] T010 Implement typed domain entities, invariants, and terminal-state rules in `src/hgcl/studies/s003/domain.py` to satisfy T009 — FR-017–FR-018, FR-027, FR-044
- [X] T011 [P] Write artifact-envelope, atomic-write, immutability, relative-path, `DryRunApproval`, and sealed `EvaluationCohort` contract tests in `tests/s003/contract/test_artifact_contract.py` — FR-004, FR-026–FR-027, FR-029, FR-038, SC-006
- [X] T012 Implement the common envelope, hashing, atomic JSON/Parquet registration, immutable artifact index, `DryRunApproval`, and `EvaluationCohort` persistence in `src/hgcl/studies/s003/artifacts.py` to satisfy T011 — FR-004, FR-026–FR-027, FR-029, FR-038
- [X] T013 [P] Write CLI parser, JSON-output, exit-code, identifier-prefix, `approve-dry-run`, incompatible-approval rejection, and no-S02-fallback contract tests in `tests/s003/contract/test_cli_contract.py` — FR-001, FR-029, FR-033
- [X] T014 Implement the `hgcl-s003` command shell and structured error mapping in `src/hgcl/studies/s003/cli.py` to satisfy T013, leaving story handlers injectable — FR-001, FR-027, FR-033
- [X] T015 [P] Write environment compatibility and fail-closed doctor tests for Python, dependencies, CUDA, device, RAM, VRAM, disk, and path overlap in `tests/s003/unit/test_environment.py` — FR-028, FR-046
- [X] T016 Implement the S003 environment doctor and resource probes in `src/hgcl/studies/s003/environment.py` to satisfy T015, with no automatic lab fallback — FR-028, FR-046
- [X] T017 [P] Add deterministic tiny transaction CSV fixtures and expected hashes/counts under `tests/s003/fixtures/` plus fixture loaders in `tests/s003/conftest.py` — FR-004, FR-042–FR-043
- [X] T018 [P] Add an isolation regression that rejects `native_wallet`, `fusion.alphas`, `hgcl`, S02 artifact roots, and modifications to frozen S02 paths in `tests/s003/contract/test_study_isolation.py` — FR-033–FR-034, SC-009

**Checkpoint**: Configuração, domínio, artefatos, CLI e ambiente possuem contratos testáveis; user stories podem começar.

---

## Phase 3: User Story 1 — Execute the Main Experiment Without Temporal Leakage (Priority: P1) 🎯 MVP

**Goal**: Preparar snapshots `Tx→Tx`, pré-treinar o S003-TxGCL, selecionar/refazer o downstream e avaliar somente por meio do guard temporal.

**Independent Test**: Executar o smoke treinado com CSVs originais e shadow test em 1–34, seguido do dry-run estrutural; auditar manifests para demonstrar 182 features, encoder GIN 2×128 compartilhado, zero influência de 35–49 e design íntegro sem treino adicional.

### Tests for User Story 1

- [X] T019 [P] [US1] Write source-schema tests for required core CSVs, optional address CSVs, 183 non-ID columns, 182 model features, class mapping, duplicate IDs, and invalid references in `tests/s003/contract/test_data_contract.py` — FR-003–FR-004, FR-039, FR-043
- [X] T020 [P] [US1] Write causal feature-audit tests for local, aggregate, augmented, identifier, and time-step groups in `tests/s003/unit/test_feature_audit.py` — FR-003, FR-010–FR-011, FR-039
- [X] T021 [P] [US1] Write snapshot tests for `edge_index[0]=source`, `edge_index[1]=target`, PyG `source_to_target`, intrastep edges, duplicate/self-loop accounting, empty-edge graphs, isolated nodes, and canonical external-ID mapping in `tests/s003/unit/test_snapshots.py` — FR-002, FR-007, FR-037, FR-042–FR-043
- [X] T022 [P] [US1] Write nested stratified budget tests for all fractions/seeds, 80/20 fit-validation pools, rounding, class preservation, refit union, and hash stability in `tests/s003/unit/test_budgets.py` — FR-017–FR-018, FR-042
- [X] T023 [P] [US1] Write augmentation and positive-set tests for exactly two stochastic/block representations, KNN-only positive expansion, successor-only structural positives, snapshot-local KNN, `min(K,N-1)`, deduplication, and positive-negative disjointness in `tests/s003/unit/test_augmentations.py` and `tests/s003/unit/test_positives.py` — FR-009–FR-014, FR-037, FR-040, FR-043
- [X] T024 [P] [US1] Write GIN 2×128, shared-encoder, symmetric multi-positive loss, invalid-anchor, and finite-gradient tests in `tests/s003/unit/test_models.py` — FR-008, FR-013–FR-015, FR-040, FR-043
- [X] T025 [P] [US1] Write TestLabelStore sealing, exact shadow partition, 205-cell cohort accounting with selected/terminal states, single global release, frozen-threshold, and post-release mutation/repeated-release rejection tests in `tests/s003/contract/test_evaluation_access.py` — FR-005–FR-007, FR-028, FR-038, SC-001–SC-002
- [X] T026 [P] [US1] Write an end-to-end temporal-isolation profile test with original-data marker, deterministic 256-node trained smoke, full-snapshot batching, and structural dry-run with zero training epochs in `tests/s003/integration/test_temporal_isolation.py` — FR-005–FR-008, FR-028, FR-038, SC-002, SC-007
- [X] T027 [P] [US1] Write reload determinism tests for exact manifests and `rtol=1e-5`, `atol=2e-6` neural scores/classes in `tests/s003/integration/test_reproducibility.py` — FR-042, SC-003

### Implementation for User Story 1

- [X] T028 [US1] Implement read-only core-source discovery, schema validation, hashes, class mapping, and optional address-file inventory in `src/hgcl/studies/s003/data.py` — FR-003–FR-004, FR-039, FR-043
- [X] T029 [US1] Implement the versioned per-feature-group causal availability audit and stop condition in `src/hgcl/studies/s003/data.py` — FR-039
- [X] T030 [US1] Implement 49 canonical directed snapshots, external-ID maps, edge validation/deduplication, and manifests in `src/hgcl/studies/s003/data.py` — FR-002, FR-007, FR-037, FR-042–FR-043
- [X] T031 [US1] Implement fit-only normalization for the 182 financial features and immutable preprocessing state in `src/hgcl/studies/s003/data.py` — FR-003, FR-005–FR-006
- [X] T032 [P] [US1] Implement deterministic nested label budgets and 80/20 internal pools in `src/hgcl/studies/s003/splits.py` — FR-017–FR-018
- [X] T033 [P] [US1] Implement sealed test-label storage, exact engineering shadow partition, cohort-level single global release, per-member access audit, and post-release mutation guard in `src/hgcl/studies/s003/evaluation.py` — FR-005–FR-007, FR-028, FR-038
- [X] T034 [P] [US1] Implement PyG `source_to_target` directed GIN 2×128, projection head, and frozen embedding extraction in `src/hgcl/studies/s003/models.py` — FR-008, FR-015–FR-016, FR-037
- [X] T035 [P] [US1] Implement stochastic and functional-block transformations with deterministic generators in `src/hgcl/studies/s003/augmentations.py` — FR-009–FR-011, FR-020
- [X] T036 [US1] Implement per-snapshot cosine KNN, self/successor/KNN positive masks, full-snapshot negatives, invalid-anchor handling, and stable ID tie-breaking in `src/hgcl/studies/s003/positives.py` — FR-012–FR-014, FR-037, FR-040, FR-043
- [X] T037 [US1] Implement the symmetric GCPAL-compatible multi-positive loss over stochastic/block embeddings in `src/hgcl/studies/s003/models.py` — FR-013–FR-014, FR-040
- [X] T038 [US1] Implement label-free fixed-epoch SSL training, diagnostics capture, resource guards, and checkpoint hooks in `src/hgcl/studies/s003/training.py` — FR-008, FR-023, FR-025, FR-028, FR-046
- [X] T039 [US1] Implement frozen-encoder `H‖X_tx` MLP search, F1/MCC selection, threshold freezing, and full-budget refit in `src/hgcl/studies/s003/training.py` — FR-016, FR-018, FR-023
- [X] T040 [P] [US1] Implement pooled and per-snapshot MCC, illicit F1/precision/recall, PR-AUC, null metric reasons, and support accounting in `src/hgcl/studies/s003/evaluation.py` — FR-023–FR-024, FR-044
- [X] T041 [US1] Implement independent snapshot inference, immutable predictions, and resumable evaluation of every frozen cohort member after one global test release in `src/hgcl/studies/s003/evaluation.py` — FR-005–FR-007, FR-024, FR-038
- [X] T042 [US1] Reconcile config, artifact, and CLI contracts with the approved profile change: add the trained `smoke` command, make `dry-run` structural with zero training epochs, require `ssl.epochs=100` for `lab`, require `training_performed=false` and `test_labels_materialized=false`, and update the affected contract/integration tests and profile YAMLs — FR-023, FR-028–FR-029, FR-038
- [X] T043 [US1] Complete and test `prepare`, `audit`, trained smoke, structural dry-run, and fail-closed guarded-evaluation validation plus their `doctor`/`prepare`/`audit`/`smoke`/`dry-run`/`evaluate` CLI handlers in `src/hgcl/studies/s003/pipeline.py` and `src/hgcl/studies/s003/cli.py`; defer real 205-cell cohort sealing/evaluation to T051/T053 after the registry exists — FR-004–FR-008, FR-026–FR-029, FR-038–FR-039, FR-046
- [X] T044 [US1] Run and make green the US1 contract/unit/integration suite, proving that smoke trains only on the engineering partition and structural dry-run creates no checkpoint, prediction, model metric, or test-label access — SC-002–SC-003, SC-007

**Checkpoint**: O método principal percorre o smoke em shadow data e o dry-run estrutural audita o design sem abrir os rótulos reais de teste. Este é o MVP implementável.

---

## Phase 4: User Story 2 — Measure Label Efficiency and Pretraining Contribution (Priority: P1)

**Goal**: Implementar baselines, ablações, matriz canônica, diagnósticos e comparação estatística justa.

**Independent Test**: Gerar o design de 205 células P1, executar uma matriz reduzida com IDs compartilhados e demonstrar que baselines/ablações reutilizam budgets, que masking tem cardinalidade pareada e que estatística só é inferencial com cinco pares completos.

- [X] T045 [P] [US2] Implement and contract-test the nine-method registry, shared adapter interface, identical budgets/splits/seeds/target/evaluation policy, fit-only prevalence, directed-flow invariant, and `approximation` marking for incomplete reproductions — FR-015, FR-021–FR-023, FR-037, FR-041
- [X] T046 [US2] Implement and test MLP X-only, Random Forest 300-tree, and XGBoost adapters with the shared prevalence and selection policies in `src/hgcl/studies/s003/baselines.py` — FR-021–FR-023, FR-041
- [X] T047 [US2] Implement and test directed two-layer GCN, GraphSAGE, and supervised GIN adapters with 128-dimensional embeddings — FR-015, FR-021–FR-023, FR-037, FR-041
- [X] T048 [US2] Implement and test faithful Elliptic++ adaptations of Inspection-L (GIN 2×128, DGI, RF 100) and GCPAL (GIN 2×128, two stochastic views, K=10 multi-positive loss, two-layer `H‖X_tx` MLP), declaring unavoidable migration differences — FR-014–FR-015, FR-021–FR-023
- [X] T049 [P] [US2] Implement and test the S003-TxGCL adapter, `X-only`/`H-only`/`H‖X_tx`, exact-cardinality functional/random-group/random-individual controls, no-KNN/no-edge-dropout variants, and gated reverse-edge P2 variant — FR-016, FR-019–FR-020, FR-037
- [X] T050 [US2] Implement and test the canonical design of 205 P1 cells, optional five P2 cells, immutable embedding-cache keys, digest-bound approval compatibility, and the structural dry-run coverage/resource/resume plan — FR-017, FR-019–FR-022, FR-026, FR-028–FR-029, SC-001, SC-005
- [X] T051 [US2] Implement and test resumable matrix scheduling, explicit failure states, cache reuse across fractions, zero pre-evaluation test access, 205-cell accounting, and final evaluation-cohort sealing — FR-016–FR-019, FR-027–FR-029, FR-038, FR-042
- [X] T052 [P] [US2] Implement and test representation diagnostics, pooled/per-snapshot reporting, temporal series, ablation attribution, paired t intervals/tests, Cohen `d_z`, Holm correction, incomplete-pair behavior, and scientific claim gates — FR-024–FR-025, FR-035–FR-036, FR-044–FR-045, SC-004
- [X] T053 [US2] Wire and integration-test `approve-dry-run`, `matrix`, and `report`, including fail-closed digest binding and a tiny non-scientific matrix fixture — FR-027–FR-029, SC-001, SC-004–SC-005
- [X] T054 [US2] Run and make green all US2 tests, persisting only non-scientific fixture evidence under `artifacts/s003/` — SC-001, SC-004–SC-005

**Checkpoint**: Toda comparação científica está implementada e verificável em escala reduzida, sem autorização para executar a matriz completa.

---

## Phase 5: User Story 3 — Audit Every Reported Result (Priority: P1)

**Goal**: Tornar cada número rastreável a dados, configuração, código, IDs, estados, seleção e acesso ao teste.

**Independent Test**: Selecionar uma linha do relatório reduzido e reconstruir todos os insumos e decisões por `run_id`; interromper/retomar uma run e demonstrar que estados incompletos não entram como sucesso.

- [X] T055 [P] [US3] Implement and test source/preparation/run/environment/selection/evaluation manifests, atomic run-state transitions, failure categorization, terminal immutability, and checkpoint contents with hashes, RNGs and dependency versions — FR-004, FR-026–FR-027, FR-042, SC-006
- [X] T056 [US3] Implement and integration-test strict resume, reload equivalence, evaluation-access logging, and post-unblinding `technical_rerun_of` enforcement without permitting reselection — FR-026–FR-027, FR-038, FR-042, SC-003
- [X] T057 [US3] Implement and test row-level lineage, explicit incomplete-state coverage, `resume`/provenance CLI output, and an end-to-end arbitrary-result reconstruction scenario — FR-026–FR-027, SC-001, SC-006
- [X] T058 [US3] Run and make green all US3 tests and verify that no completed artifact is mutated — SC-003, SC-006

**Checkpoint**: Qualquer resultado reduzido possui cadeia de custódia reconstruível e estados incompletos permanecem explícitos.

---

## Phase 6: User Story 4 — Decide on the Heterogeneous Extension (Priority: P3)

**Goal**: Implementar somente o gate causal/de recursos de `Addr↔Tx`, sem colocar treinamento heterogêneo no caminho P1.

**Independent Test**: Executar o gate com arquivos de endereço ausentes, features globais não causais e projeção acima do orçamento; cada caso deve produzir `defer` rastreável sem bloquear o núcleo homogêneo.

- [X] T059 [P] [US4] Write the heterogeneous-gate contract and nonblocking integration tests for causality, supervision, comparability, resources, approval, missing files, and include/defer invariants — FR-030–FR-032, SC-008
- [X] T060 [US4] Implement address evidence inventory, causal/supervision/target/resource checks, persisted `HeterogeneousExtensionDecision`, and `hetero-gate` CLI; make US4 tests green without creating an encoder or training run — FR-030–FR-032, SC-008

**Checkpoint**: A extensão possui decisão reproduzível; implementação heterogênea continua fora de escopo até aceite futuro explícito.

---

## Phase 7: Polish, Validation, and Researcher Gate

**Purpose**: Fechar documentação, validação integrada e evidências do dry-run sem iniciar a matriz científica.

- [X] T061 [P] Update S003 usage, command examples, artifact lifecycle, requirement traceability, implementation status, and the S02 boundary in `README.md`, `docs/studies/s003/context.md`, and `docs/studies/s003/status.md` — FR-026, FR-033–FR-034, SC-006
- [X] T062 Run the complete `tests/s003/` suite, historical regression tests, contract checks, `git diff --check`, and frozen-S02-path audit; record exact commands/results and remaining limitations — FR-033–FR-035, SC-009–SC-010
- [X] T063 Execute the trained Mac smoke from `quickstart.md`, verify ≤10 minutes and zero test-label access, and record its non-scientific evidence under `artifacts/s003/` — FR-028, FR-038, SC-007
- [X] T064 Execute the laboratory `doctor` and structural dry-run without training; verify exactly 205 P1 cells, `ssl.epochs=100` in the proposed immutable lab config, zero test-label materialization, cache/checkpoint/resume plans, resource capacity, and a documented duration projection with 20% margin — FR-023, FR-028–FR-029, SC-007
- [X] T065 Stop and request explicit researcher acceptance of the smoke/dry-run evidence and proposed lab config; only after acceptance materialize the byte-identical proposal as `configs/s003/lab.yaml`, invoke `approve-dry-run`, rerun the release checks, and record `approval.json` without launching the matrix — FR-023, FR-029, FR-037, SC-009–SC-010

**Final Checkpoint**: Software e protocolo estão prontos para uma decisão humana sobre a matriz; nenhuma célula científica completa foi executada automaticamente.

---

## Phase 8: Post-Execution Defect Remediation — Snapshot-by-Snapshot SSL Alignment (D003-018)

**Purpose**: Corrigir o estouro de memória (74,3 GB) identificado na execução do cluster pela adesão estrita ao pré-treino temporal snapshot por snapshot (FR-008, FR-012, FR-040) em `S003TxGCLAdapter` e `GCPALAdapter`, e enriquecer a observabilidade de erros na matriz.

- [x] T066 [Story 1 & 2] Remediate contrastive SSL pre-training in `S003TxGCLAdapter` and `GCPALAdapter` to strictly iterate snapshot-by-snapshot over training snapshots (1–34), computing KNN ($k=10$) and in-batch negatives locally per snapshot ($< 100\text{ MB}$ peak RAM); extract and concatenate frozen embeddings for downstream tabular optimization; enrich `execute_matrix_cell` and `pipeline.py` with full traceback logging and `failure_message` recording in `progress.json`; and validate via regression tests in `tests/s003/integration/test_snapshot_ssl_execution.py` — FR-008, FR-012, FR-040, SC-006

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 — Setup**: sem dependências.
- **Phase 2 — Foundational**: depende da Phase 1 e bloqueia todas as stories.
- **US1 / Phase 3**: depende da fundação; entrega o MVP e bloqueia a integração de US2 e US3.
- **US2 / Phase 4**: depende dos snapshots, budgets, encoder, treino e avaliação de US1.
- **US3 / Phase 5**: testes/entidades podem começar após a fundação, mas a integração depende dos fluxos US1/US2.
- **US4 / Phase 6**: pode começar após a fundação e o inventário de dados de US1; não depende de US2/US3.
- **Phase 7**: depende de US1–US3; o gate heterogêneo pode terminar em `defer` sem bloquear o núcleo.

### User Story Dependency Graph

```text
Setup -> Foundation -> US1 (MVP) -> US2 -> Polish/structural gate
                         └-------> US3 ----┘
                   └------------> US4 (non-blocking gate)
```

### Within Each User Story

- Escrever os testes da story e observar falha antes da implementação correspondente.
- Implementar dados/modelos antes de orquestração e handlers CLI.
- Congelar interfaces e manifests antes dos testes de integração.
- Não avançar ao teste real 35–49 em smoke, dry-run ou implementação.
- Não iniciar T065 sem evidências de T062–T064 completas.

## Parallel Opportunities

- Setup T003–T006 pode avançar em paralelo após confirmação de T001/T002 quando aplicável.
- Testes foundational T007, T009, T011, T013, T015, T017 e T018 ocupam arquivos distintos.
- Em US1, testes T019–T027 podem ser escritos em paralelo; após dados básicos, splits, guard, modelo e augmentations também se separam por arquivo.
- Em US2, adapters que compartilham `baselines.py` são sequenciais; T049 e T052 podem avançar em módulos distintos depois de T045.
- US3 e US4 podem avançar paralelamente após US1, pois provenance e gate heterogêneo têm superfícies distintas.

## Parallel Example: User Story 1

```text
Task T020: testes da auditoria causal em tests/s003/unit/test_feature_audit.py
Task T022: testes de budgets em tests/s003/unit/test_budgets.py
Task T023: testes de augmentations/positivos em tests/s003/unit/test_augmentations.py e test_positives.py
Task T025: contrato de blindagem em tests/s003/contract/test_evaluation_access.py
```

## Parallel Example: User Story 2

```text
Task T046: baselines tabulares em src/hgcl/studies/s003/baselines.py
Task T049: controles de ablação em src/hgcl/studies/s003/augmentations.py
Task T052: diagnósticos e estatística em módulos distintos
```

T046 não deve ser executada simultaneamente com T047–T048 porque compartilham `baselines.py`; T049 e partes de T052 permanecem paralelizáveis.

---

## Implementation Strategy

### MVP First — User Story 1

1. Completar Setup e Foundation.
2. Implementar US1 com testes primeiro.
3. Parar após T044 e demonstrar o smoke em shadow data e o dry-run estrutural.
4. Não adicionar baselines ou matriz até o pipeline temporal estar auditável.

### Incremental Delivery

1. **US1**: núcleo dirigido e blindado.
2. **US2**: baselines, ablações, matriz reduzida e estatística.
3. **US3**: cadeia de custódia e retomada completa.
4. **US4**: decisão heterogênea não bloqueante.
5. **Polish**: smoke treinado, dry-run estrutural e parada para aceite.

### Research Matrix Boundary

- Implementar o executor da matriz faz parte do SDD.
- Executar o smoke treinado e o dry-run estrutural faz parte da validação de engenharia.
- Executar as 205 células P1 ou cinco células P2 não está autorizado por este `tasks.md`.
- Qualquer impedimento, incompatibilidade do laboratório ou necessidade de alterar protocolo exige interrupção e consulta ao pesquisador.

## Notes

- `[P]` indica paralelismo somente quando arquivos e predecessores não conflitam.
- Cada task deve preservar os testes do S02; extração compartilhada exige decisão registrada antes da mudança.
- Checkpoints e resultados volumosos permanecem fora do Git.
- Marcar uma task concluída exige evidência observável correspondente.
- O próximo passo do SDD é `$speckit-analyze`, não `$speckit-implement`.

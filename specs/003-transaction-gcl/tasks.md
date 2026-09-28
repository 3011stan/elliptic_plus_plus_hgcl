# Tasks: S003-TxGCL Transaction Classification

**Input**: Design documents from `/specs/003-transaction-gcl/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Obrigatórios devido aos riscos científicos de leakage, pareamento contrastivo, alinhamento de classes, proveniência e reprodução.

**Organization**: Tasks are grouped by user story so each story can be implemented and validated as an independent increment. Every task stays inside the S003 namespaces unless it explicitly adds the isolated `hgcl-s003` entry point or dependency metadata.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode ser executada em paralelo quando seus predecessores estiverem completos e não houver conflito de arquivo.
- **[Story]**: associa a task à user story da especificação.
- Referências `FR-*` e `SC-*` indicam o requisito verificável coberto.

## Phase 1: Setup — Isolated S003 Surface

**Purpose**: Criar a superfície isolada do estudo sem alterar comportamento ou configuração do S02.

- [ ] T001 Add the `hgcl-s003` console entry point plus direct `scipy==1.17.1` and `xgboost==3.2.0` dependencies in `pyproject.toml` — FR-021, FR-033, FR-046
- [ ] T002 Refresh Python 3.11 dependency pins for macOS CPU and laboratory CUDA in `requirements/mac-cpu.lock`, `requirements/lab-cuda.in`, and `requirements/lab-cuda.lock` — FR-046
- [ ] T003 [P] Create the planned S003 module skeleton and `hetero` subpackage in `src/hgcl/studies/s003/` without importing S02 configuration or model modules — FR-033–FR-034
- [ ] T004 [P] Create strict profile skeletons `configs/s003/smoke.yaml`, `configs/s003/dry-run.yaml`, and `configs/s003/lab.template.yaml` with `study_id=s003` and no S02 keys — FR-001, FR-033
- [ ] T005 [P] Create `contract`, `unit`, and `integration` test package structure under `tests/s003/` and shared S003-only fixture declarations in `tests/s003/conftest.py` — FR-033
- [ ] T006 [P] Ensure `artifacts/s003/`, prepared tensors, predictions, and checkpoints are ignored while small exported evidence remains allow-listed in `.gitignore` — FR-004, FR-026, FR-033

**Checkpoint**: O entry point e os namespaces do S003 existem, mas nenhum pipeline científico está implementado.

---

## Phase 2: Foundational — Blocking Contracts and Infrastructure

**Purpose**: Implementar contratos compartilhados que bloqueiam todas as user stories.

**⚠️ CRITICAL**: Nenhuma implementação de user story começa antes deste checkpoint.

- [ ] T007 [P] Write strict config contract tests for required keys, profile invariants, S02-key rejection, 182 model features, directed flow, fixed SSL epochs, 205 P1 cells, and lab approval in `tests/s003/contract/test_config_contract.py` — FR-001, FR-003, FR-017, FR-033, FR-037, FR-046
- [ ] T008 Implement canonical S003 config parsing, validation, overrides, and SHA-256 serialization in `src/hgcl/studies/s003/config.py` to satisfy T007 — FR-001, FR-026, FR-033, FR-046
- [ ] T009 [P] Write entity and state-transition tests for `SourceDataset`, `TemporalSnapshot`, `LabelBudget`, `ExperimentRun`, and `StatisticalComparison` in `tests/s003/unit/test_domain.py` — FR-017–FR-018, FR-027, FR-044
- [ ] T010 Implement typed domain entities, invariants, and terminal-state rules in `src/hgcl/studies/s003/domain.py` to satisfy T009 — FR-017–FR-018, FR-027, FR-044
- [ ] T011 [P] Write artifact-envelope, atomic-write, immutability, and relative-path contract tests in `tests/s003/contract/test_artifact_contract.py` — FR-004, FR-026–FR-027, SC-006
- [ ] T012 Implement the common envelope, hashing, atomic JSON/Parquet registration, and immutable artifact index in `src/hgcl/studies/s003/artifacts.py` to satisfy T011 — FR-004, FR-026–FR-027
- [ ] T013 [P] Write CLI parser, JSON-output, exit-code, identifier-prefix, and no-S02-fallback contract tests in `tests/s003/contract/test_cli_contract.py` — FR-001, FR-029, FR-033
- [ ] T014 Implement the `hgcl-s003` command shell and structured error mapping in `src/hgcl/studies/s003/cli.py` to satisfy T013, leaving story handlers injectable — FR-001, FR-027, FR-033
- [ ] T015 [P] Write environment compatibility and fail-closed doctor tests for Python, dependencies, CUDA, device, RAM, VRAM, disk, and path overlap in `tests/s003/unit/test_environment.py` — FR-028, FR-046
- [ ] T016 Implement the S003 environment doctor and resource probes in `src/hgcl/studies/s003/environment.py` to satisfy T015, with no automatic lab fallback — FR-028, FR-046
- [ ] T017 [P] Add deterministic tiny transaction CSV fixtures and expected hashes/counts under `tests/s003/fixtures/` plus fixture loaders in `tests/s003/conftest.py` — FR-004, FR-042–FR-043
- [ ] T018 [P] Add an isolation regression that rejects `native_wallet`, `fusion.alphas`, `hgcl`, S02 artifact roots, and modifications to frozen S02 paths in `tests/s003/contract/test_study_isolation.py` — FR-033–FR-034, SC-009

**Checkpoint**: Configuração, domínio, artefatos, CLI e ambiente possuem contratos testáveis; user stories podem começar.

---

## Phase 3: User Story 1 — Execute the Main Experiment Without Temporal Leakage (Priority: P1) 🎯 MVP

**Goal**: Preparar snapshots `Tx→Tx`, pré-treinar o S003-TxGCL, selecionar/refazer o downstream e avaliar somente por meio do guard temporal.

**Independent Test**: Executar o smoke/dry-run com CSVs originais e shadow test em 1–34; auditar manifests para demonstrar 182 features, encoder GIN 2×128 compartilhado, zero influência de 35–49 e inferência independente por snapshot.

### Tests for User Story 1

- [ ] T019 [P] [US1] Write source-schema tests for required core CSVs, optional address CSVs, 183 non-ID columns, 182 model features, class mapping, duplicate IDs, and invalid references in `tests/s003/contract/test_data_contract.py` — FR-003–FR-004, FR-039, FR-043
- [ ] T020 [P] [US1] Write causal feature-audit tests for local, aggregate, augmented, identifier, and time-step groups in `tests/s003/unit/test_feature_audit.py` — FR-003, FR-010–FR-011, FR-039
- [ ] T021 [P] [US1] Write snapshot tests for intrastep directed edges, duplicate/self-loop accounting, empty-edge graphs, isolated nodes, and canonical external-ID mapping in `tests/s003/unit/test_snapshots.py` — FR-002, FR-007, FR-037, FR-042–FR-043
- [ ] T022 [P] [US1] Write nested stratified budget tests for all fractions/seeds, 80/20 fit-validation pools, rounding, class preservation, refit union, and hash stability in `tests/s003/unit/test_budgets.py` — FR-017–FR-018, FR-042
- [ ] T023 [P] [US1] Write augmentation and positive-set tests for stochastic/block views, snapshot-local KNN, `min(K,N-1)`, deduplication, and positive-negative disjointness in `tests/s003/unit/test_augmentations.py` and `tests/s003/unit/test_positives.py` — FR-009–FR-014, FR-040, FR-043
- [ ] T024 [P] [US1] Write GIN 2×128, shared-encoder, symmetric multi-positive loss, invalid-anchor, and finite-gradient tests in `tests/s003/unit/test_models.py` — FR-008, FR-013–FR-015, FR-040, FR-043
- [ ] T025 [P] [US1] Write TestLabelStore sealing, shadow-test, single-unblinding, frozen-threshold, and repeated-access rejection tests in `tests/s003/contract/test_evaluation_access.py` — FR-005–FR-007, FR-038, SC-002
- [ ] T026 [P] [US1] Write an end-to-end temporal-isolation dry-run test with original-data marker in `tests/s003/integration/test_temporal_isolation.py` — FR-005–FR-008, FR-028, FR-038, SC-002, SC-007
- [ ] T027 [P] [US1] Write reload determinism tests for exact manifests and `rtol=1e-5`, `atol=2e-6` neural scores/classes in `tests/s003/integration/test_reproducibility.py` — FR-042, SC-003

### Implementation for User Story 1

- [ ] T028 [US1] Implement read-only core-source discovery, schema validation, hashes, class mapping, and optional address-file inventory in `src/hgcl/studies/s003/data.py` — FR-003–FR-004, FR-039, FR-043
- [ ] T029 [US1] Implement the versioned per-feature-group causal availability audit and stop condition in `src/hgcl/studies/s003/data.py` — FR-039
- [ ] T030 [US1] Implement 49 canonical directed snapshots, external-ID maps, edge validation/deduplication, and manifests in `src/hgcl/studies/s003/data.py` — FR-002, FR-007, FR-037, FR-042–FR-043
- [ ] T031 [US1] Implement fit-only normalization for the 182 financial features and immutable preprocessing state in `src/hgcl/studies/s003/data.py` — FR-003, FR-005–FR-006
- [ ] T032 [P] [US1] Implement deterministic nested label budgets and 80/20 internal pools in `src/hgcl/studies/s003/splits.py` — FR-017–FR-018
- [ ] T033 [P] [US1] Implement sealed test-label storage, evaluation access audit, shadow partition, and single-unblinding guard in `src/hgcl/studies/s003/evaluation.py` — FR-005–FR-007, FR-038
- [ ] T034 [P] [US1] Implement directed GIN 2×128, projection head, and frozen embedding extraction in `src/hgcl/studies/s003/models.py` — FR-008, FR-015–FR-016, FR-037
- [ ] T035 [P] [US1] Implement stochastic and functional-block transformations with deterministic generators in `src/hgcl/studies/s003/augmentations.py` — FR-009–FR-011, FR-020
- [ ] T036 [US1] Implement per-snapshot cosine KNN, positive masks, in-batch negatives, invalid-anchor handling, and stable ID tie-breaking in `src/hgcl/studies/s003/positives.py` — FR-012–FR-014, FR-040, FR-043
- [ ] T037 [US1] Implement the symmetric GCPAL-compatible multi-positive loss over stochastic/block embeddings in `src/hgcl/studies/s003/models.py` — FR-013–FR-014, FR-040
- [ ] T038 [US1] Implement label-free fixed-epoch SSL training, diagnostics capture, resource guards, and checkpoint hooks in `src/hgcl/studies/s003/training.py` — FR-008, FR-023, FR-025, FR-028, FR-046
- [ ] T039 [US1] Implement frozen-encoder `H‖X_tx` MLP search, F1/MCC selection, threshold freezing, and full-budget refit in `src/hgcl/studies/s003/training.py` — FR-016, FR-018, FR-023
- [ ] T040 [P] [US1] Implement pooled and per-snapshot MCC, illicit F1/precision/recall, PR-AUC, null metric reasons, and support accounting in `src/hgcl/studies/s003/evaluation.py` — FR-023–FR-024, FR-044
- [ ] T041 [US1] Implement independent snapshot inference, immutable predictions, and final evaluation state transitions in `src/hgcl/studies/s003/evaluation.py` — FR-005–FR-007, FR-024, FR-038
- [ ] T042 [US1] Orchestrate `prepare`, `audit`, shadow dry-run, selected model, and guarded evaluation flows in `src/hgcl/studies/s003/pipeline.py` — FR-004–FR-008, FR-026–FR-029, FR-038–FR-039
- [ ] T043 [US1] Wire `doctor`, `prepare`, `audit`, `dry-run`, and `evaluate` handlers in `src/hgcl/studies/s003/cli.py` — FR-028–FR-029, FR-038, FR-046
- [ ] T044 [US1] Run and make green the US1 contract/unit/integration suite in `tests/s003/`, recording the smoke evidence path in `docs/studies/s003/status.md` — SC-002–SC-003, SC-007

**Checkpoint**: O método principal percorre o pipeline em shadow data sem abrir os rótulos reais de teste. Este é o MVP implementável.

---

## Phase 4: User Story 2 — Measure Label Efficiency and Pretraining Contribution (Priority: P1)

**Goal**: Implementar baselines, ablações, matriz canônica, diagnósticos e comparação estatística justa.

**Independent Test**: Gerar o design de 205 células P1, executar uma matriz reduzida com IDs compartilhados e demonstrar que baselines/ablações reutilizam budgets, que masking tem cardinalidade pareada e que estatística só é inferencial com cinco pares completos.

### Tests for User Story 2

- [ ] T045 [P] [US2] Write exact-cardinality functional, random-group, random-individual, no-KNN, no-edge-dropout, and reverse-edge ablation tests in `tests/s003/unit/test_ablations.py` — FR-020, FR-037
- [ ] T046 [P] [US2] Write method-adapter contract tests for all nine matrix methods, including Inspection-L and GCPAL characteristic components in `tests/s003/contract/test_method_registry.py` — FR-015, FR-021–FR-022, FR-041
- [ ] T047 [P] [US2] Write canonical design tests for 180 main cells, five H-only cells, twenty additional P1 ablations, optional five P2 cells, and deduplication in `tests/s003/unit/test_matrix_design.py` — FR-017, FR-019–FR-022, SC-001, SC-005
- [ ] T048 [P] [US2] Write fairness tests for identical IDs, splits, seeds, target, directed graph, fit-only prevalence, selection policy, and evaluation policy across adapters in `tests/s003/integration/test_method_fairness.py` — FR-022–FR-023, FR-037, FR-041
- [ ] T049 [P] [US2] Write alignment, uniformity, and effective-rank aggregation tests by snapshot/seed in `tests/s003/unit/test_diagnostics.py` — FR-025
- [ ] T050 [P] [US2] Write paired t, t-interval, Cohen `d_z`, Holm-family, zero-SD, incomplete-pair, and claim-gate tests in `tests/s003/unit/test_statistics.py` — FR-036, FR-044–FR-045
- [ ] T051 [P] [US2] Write pooled-versus-snapshot reporting and 205-cell coverage tests in `tests/s003/integration/test_matrix_reporting.py` — FR-024, FR-026–FR-027, SC-001, SC-004–SC-005

### Implementation for User Story 2

- [ ] T052 [US2] Implement MLP X-only, Random Forest 300-tree, and XGBoost adapters with shared fit-only prevalence policy in `src/hgcl/studies/s003/baselines.py` — FR-021–FR-023, FR-041
- [ ] T053 [US2] Implement directed two-layer GCN, GraphSAGE, and supervised GIN adapters with 128d embeddings in `src/hgcl/studies/s003/baselines.py` — FR-015, FR-021–FR-023, FR-037, FR-041
- [ ] T054 [US2] Implement the Inspection-L adaptation with GIN 2×128, DGI, RF 100 trees, and declared Elliptic++ feature migration in `src/hgcl/studies/s003/baselines.py` — FR-015, FR-021–FR-023
- [ ] T055 [US2] Implement the GCPAL adaptation with GIN 2×128, two stochastic views, KNN `K=10`, multi-positive loss, and two-layer `H‖X_tx` MLP in `src/hgcl/studies/s003/baselines.py` — FR-014–FR-015, FR-021–FR-023
- [ ] T056 [US2] Implement the S003-TxGCL adapter plus `X-only`, `H-only`, and `H‖X_tx` representations in `src/hgcl/studies/s003/baselines.py` — FR-016, FR-019
- [ ] T057 [P] [US2] Implement deterministic random-group/random-individual controls, no-KNN/no-edge-dropout variants, and gated reverse-edge augmentation in `src/hgcl/studies/s003/augmentations.py` — FR-020, FR-037
- [ ] T058 [US2] Implement the validated method/variant registry and reject incomplete reproductions as `approximation` in `src/hgcl/studies/s003/baselines.py` — FR-021–FR-022
- [ ] T059 [P] [US2] Implement immutable frozen-embedding cache keys and reuse across downstream fractions in `src/hgcl/studies/s003/artifacts.py` — FR-016–FR-019, FR-026, FR-042
- [ ] T060 [US2] Implement canonical matrix design, 205-cell coverage, approval checks, failure states, and resumable scheduling in `src/hgcl/studies/s003/pipeline.py` — FR-017, FR-019–FR-022, FR-027–FR-029
- [ ] T061 [P] [US2] Implement alignment, uniformity, and effective-rank per-snapshot/seed aggregation in `src/hgcl/studies/s003/evaluation.py` — FR-025
- [ ] T062 [P] [US2] Implement paired comparisons, t intervals/tests, Cohen `d_z`, Holm correction, incomplete-pair handling, and scientific claim gates in `src/hgcl/studies/s003/statistics.py` — FR-036, FR-044–FR-045
- [ ] T063 [US2] Implement matrix coverage, pooled/per-snapshot tables, temporal series, ablation attribution, and statistical report assembly in `src/hgcl/studies/s003/evaluation.py` — FR-024–FR-025, FR-035–FR-036, FR-044–FR-045
- [ ] T064 [US2] Wire `matrix` and `report` handlers with dry-run approval enforcement in `src/hgcl/studies/s003/cli.py` — FR-027–FR-029
- [ ] T065 [US2] Run and make green all US2 tests, persisting only non-scientific reduced-matrix evidence under `artifacts/s003/` — SC-001, SC-004–SC-005

**Checkpoint**: Toda comparação científica está implementada e verificável em escala reduzida, sem autorização para executar a matriz completa.

---

## Phase 5: User Story 3 — Audit Every Reported Result (Priority: P1)

**Goal**: Tornar cada número rastreável a dados, configuração, código, IDs, estados, seleção e acesso ao teste.

**Independent Test**: Selecionar uma linha do relatório reduzido e reconstruir todos os insumos e decisões por `run_id`; interromper/retomar uma run e demonstrar que estados incompletos não entram como sucesso.

### Tests for User Story 3

- [ ] T066 [P] [US3] Write lineage tests from a report row through run, config, source, budget, selection, predictions, metrics, and code revision in `tests/s003/contract/test_lineage.py` — FR-004, FR-026, SC-006
- [ ] T067 [P] [US3] Write run-state, atomic-transition, terminal-state, timeout, disk-full, and no-overwrite tests in `tests/s003/unit/test_run_state.py` — FR-027, FR-038
- [ ] T068 [P] [US3] Write checkpoint interruption/resume and post-unblinding `technical_rerun` integration tests in `tests/s003/integration/test_resume.py` — FR-026–FR-027, FR-038, FR-042

### Implementation for User Story 3

- [ ] T069 [P] [US3] Implement source, preparation, run, environment, selection, and evaluation manifests with file hashes in `src/hgcl/studies/s003/provenance.py` — FR-004, FR-026, SC-006
- [ ] T070 [P] [US3] Implement atomic run-state transitions, failure categorization, phase-aware interruption, and terminal immutability in `src/hgcl/studies/s003/artifacts.py` — FR-027
- [ ] T071 [P] [US3] Implement checkpoints containing model, optimizer, epoch, RNGs, dependency versions, and compatibility digests in `src/hgcl/studies/s003/training.py` — FR-026–FR-027, FR-042
- [ ] T072 [US3] Implement strict resume validation and prediction reload equivalence in `src/hgcl/studies/s003/pipeline.py` — FR-027, FR-042, SC-003
- [ ] T073 [US3] Implement evaluation-access logging and `technical_rerun_of` enforcement in `src/hgcl/studies/s003/evaluation.py` — FR-038
- [ ] T074 [US3] Implement row-level lineage and explicit missing/failed/invalid/interrupted coverage in `src/hgcl/studies/s003/evaluation.py` — FR-026–FR-027, SC-001, SC-006
- [ ] T075 [US3] Wire `resume` and provenance-query output into `src/hgcl/studies/s003/cli.py` — FR-026–FR-027, FR-038
- [ ] T076 [US3] Add the end-to-end arbitrary-result reconstruction scenario in `tests/s003/integration/test_lineage_roundtrip.py` — SC-006
- [ ] T077 [US3] Run and make green all US3 provenance/resume tests, inspect that no completed artifact is mutated, and record the result in `docs/studies/s003/status.md` — SC-003, SC-006

**Checkpoint**: Qualquer resultado reduzido possui cadeia de custódia reconstruível e estados incompletos permanecem explícitos.

---

## Phase 6: User Story 4 — Decide on the Heterogeneous Extension (Priority: P3)

**Goal**: Implementar somente o gate causal/de recursos de `Addr↔Tx`, sem colocar treinamento heterogêneo no caminho P1.

**Independent Test**: Executar o gate com arquivos de endereço ausentes, features globais não causais e projeção acima do orçamento; cada caso deve produzir `defer` rastreável sem bloquear o núcleo homogêneo.

### Tests for User Story 4

- [ ] T078 [P] [US4] Write gate-decision contract tests for causality, supervision, comparability, resources, approval, and include/defer invariants in `tests/s003/contract/test_hetero_gate.py` — FR-030–FR-032
- [ ] T079 [P] [US4] Write integration tests proving missing address files and failed gates do not block homogeneous preparation/reporting in `tests/s003/integration/test_hetero_nonblocking.py` — FR-030, FR-032, SC-008

### Implementation for User Story 4

- [ ] T080 [P] [US4] Implement address-file inventory and causal timestamp/aggregation evidence collection in `src/hgcl/studies/s003/hetero/gate.py` — FR-030–FR-031
- [ ] T081 [P] [US4] Implement supervision and target-alignment checks against homogeneous budgets and transaction IDs in `src/hgcl/studies/s003/hetero/gate.py` — FR-030–FR-031
- [ ] T082 [US4] Implement memory/time projection and four-gate include/defer decision assembly in `src/hgcl/studies/s003/hetero/gate.py` — FR-030–FR-032
- [ ] T083 [US4] Persist `HeterogeneousExtensionDecision` evidence and wire `hetero-gate` in `src/hgcl/studies/s003/cli.py` — FR-030–FR-032, SC-008
- [ ] T084 [US4] Run and make green all US4 tests without creating an Addr↔Tx encoder or training run, recording the result in `docs/studies/s003/status.md` — SC-008

**Checkpoint**: A extensão possui decisão reproduzível; implementação heterogênea continua fora de escopo até aceite futuro explícito.

---

## Phase 7: Polish, Validation, and Researcher Gate

**Purpose**: Fechar documentação, validação integrada e evidências do dry-run sem iniciar a matriz científica.

- [ ] T085 [P] Update S003 usage, command examples, artifact lifecycle, and S02 boundary in `README.md` and `docs/studies/s003/context.md` — FR-033–FR-034
- [ ] T086 [P] Add requirement-to-test/task traceability and implementation status fields to `docs/studies/s003/status.md` — FR-026, SC-006
- [ ] T087 Run the complete `tests/s003/` suite plus historical regression tests and record command/results in `docs/studies/s003/status.md` — FR-033–FR-034, SC-009
- [ ] T088 Execute the Mac smoke path from `specs/003-transaction-gcl/quickstart.md`, verify ≤10 minutes training and zero test-label access, and record evidence under `artifacts/s003/` — FR-028, FR-038, SC-007
- [ ] T089 Execute the one-seed laboratory dry-run, produce resource/timing/matrix projections with 20% margin, and write the review package under `artifacts/s003/` — FR-028–FR-029, SC-007
- [ ] T090 Stop and request explicit researcher acceptance of the dry-run evidence; only after acceptance generate `configs/s003/lab.yaml` and its signed `approval.json` without launching the matrix — FR-023, FR-029, FR-037
- [ ] T091 Re-run contract checks, `git diff --check`, S02 frozen-path audit, and the quickstart release gate; update `docs/studies/s003/status.md` with remaining limitations — FR-033–FR-035, SC-009–SC-010

**Final Checkpoint**: Software e protocolo estão prontos para uma decisão humana sobre a matriz; nenhuma célula científica completa foi executada automaticamente.

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
Setup -> Foundation -> US1 (MVP) -> US2 -> Polish/dry-run gate
                         └-------> US3 ----┘
                   └------------> US4 (non-blocking gate)
```

### Within Each User Story

- Escrever os testes da story e observar falha antes da implementação correspondente.
- Implementar dados/modelos antes de orquestração e handlers CLI.
- Congelar interfaces e manifests antes dos testes de integração.
- Não avançar ao teste real 35–49 em smoke, dry-run ou implementação.
- Não iniciar T090 sem evidências de T087–T089 completas.

## Parallel Opportunities

- Setup T003–T006 pode avançar em paralelo após confirmação de T001/T002 quando aplicável.
- Testes foundational T007, T009, T011, T013, T015, T017 e T018 ocupam arquivos distintos.
- Em US1, testes T019–T027 podem ser escritos em paralelo; após dados básicos, splits, guard, modelo e augmentations também se separam por arquivo.
- Em US2, adapters tabulares, GNNs supervisionadas, Inspection-L, GCPAL, controles e estatística ocupam módulos/testes independentes antes da integração no registry.
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
Task T052: baselines tabulares em src/hgcl/studies/s003/baselines.py
Task T057: controles de ablação em src/hgcl/studies/s003/augmentations.py
Task T061: diagnósticos em src/hgcl/studies/s003/evaluation.py
Task T062: estatística em src/hgcl/studies/s003/statistics.py
```

T052 não deve ser executada simultaneamente com T053–T056 porque compartilham `baselines.py`; os testes correspondentes e T057/T061/T062 permanecem paralelizáveis.

---

## Implementation Strategy

### MVP First — User Story 1

1. Completar Setup e Foundation.
2. Implementar US1 com testes primeiro.
3. Parar após T044 e demonstrar o dry-run em shadow data.
4. Não adicionar baselines ou matriz até o pipeline temporal estar auditável.

### Incremental Delivery

1. **US1**: núcleo dirigido e blindado.
2. **US2**: baselines, ablações, matriz reduzida e estatística.
3. **US3**: cadeia de custódia e retomada completa.
4. **US4**: decisão heterogênea não bloqueante.
5. **Polish**: smoke, dry-run de uma seed e parada para aceite.

### Research Matrix Boundary

- Implementar o executor da matriz faz parte do SDD.
- Executar smoke e dry-run faz parte da validação de engenharia.
- Executar as 205 células P1 ou cinco células P2 não está autorizado por este `tasks.md`.
- Qualquer impedimento, incompatibilidade do laboratório ou necessidade de alterar protocolo exige interrupção e consulta ao pesquisador.

## Notes

- `[P]` indica paralelismo somente quando arquivos e predecessores não conflitam.
- Cada task deve preservar os testes do S02; extração compartilhada exige decisão registrada antes da mudança.
- Checkpoints e resultados volumosos permanecem fora do Git.
- Marcar uma task concluída exige evidência observável correspondente.
- O próximo passo do SDD é `$speckit-analyze`, não `$speckit-implement`.

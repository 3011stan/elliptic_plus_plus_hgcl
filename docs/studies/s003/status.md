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
  `specs/003-transaction-gcl/tasks.md`: 65 tarefas rastreáveis, das quais T001–T049
  estão concluídas e T050–T065 permanecem abertas. O agrupamento reduz a
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

## Próximo passo autorizado

Executar T050 para consolidar o design canônico, identidades de cache e vínculo
do dry-run estrutural à aprovação.
Não iniciar a matriz científica; ela permanece
condicionada à implementação, ao smoke, ao dry-run estrutural e ao aceite explícito
de suas evidências.

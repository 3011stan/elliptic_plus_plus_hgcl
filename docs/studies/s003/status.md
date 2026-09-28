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
  `specs/003-transaction-gcl/tasks.md`: 65 tarefas rastreáveis, das quais T001–T042
  estão concluídas e T043–T065 permanecem abertas. O agrupamento reduz a
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
  `lab` exige exatamente 100 épocas SSL. A suíte S003 completa possui 62 testes
  aprovados no ambiente local, inclusive a dependência XGBoost.

## Próximo passo autorizado

Executar T043 para completar a orquestração e os handlers do smoke, dry-run
estrutural e avaliação guardada. Não iniciar a matriz científica; ela permanece
condicionada à implementação, ao smoke, ao dry-run estrutural e ao aceite explícito
de suas evidências.

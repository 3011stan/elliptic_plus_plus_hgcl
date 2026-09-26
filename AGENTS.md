# Instruções permanentes para agentes — Estudos Elliptic++

## Interrupção obrigatória diante de impedimentos ou desvio de escopo

Instrução explícita do pesquisador, registrada em 2026-09-11, válida para toda a
implementação do projeto:

> Se tiver algum impedimento ou algo que saia do escopo, interrompa e me consulte.

Ao encontrar um impedimento ou uma ação fora do escopo autorizado, interrompa a
execução e consulte o pesquisador antes de continuar. Informe a tarefa afetada,
a evidência do problema, seu impacto e a decisão necessária. Não contorne o
impedimento nem altere escopo, protocolo, dependências ou critérios de aceite
silenciosamente. Não marque tarefas incompletas como concluídas.

Esta regra prevalece sobre orientações locais de continuidade autônoma. Uma autorização
anterior para executar tarefas não é autorização para contornar novos impedimentos.
Registre o ponto de parada para permitir retomada após a resposta do pesquisador.

## Escopo e rastreabilidade

- Escopo ativo: preparação e execução do Estudo 003, cujo alvo é a classificação
  de transações no grafo homogêneo Tx→Tx. O identificador interno do método é
  `S003-TxGCL`.
- O Estudo 002 está congelado na tag `s02-001-final`. Seus specs, configurações,
  relatórios, artefatos e implementação são históricos e não devem ser alterados
  nem carregados como contexto por padrão.
- Reutilização de código do Estudo 002 exige identificação explícita do componente,
  teste de compatibilidade e registro da decisão. Não reutilize silenciosamente
  conceitos como `native_wallet`, `fusion.alphas` ou o método `hgcl`.
- Contexto canônico do estudo ativo: `docs/studies/s003/context.md`.
- Decisões do estudo ativo: `docs/studies/s003/decisions.md`.
- Estado operacional do estudo ativo: `docs/studies/s003/status.md`.
- A proposta científica aprovada está em
  `docs/reference/s003/proposta-003.md`.
- Use somente o notebook NotebookLM `66fb9e95-d225-4eaf-b45b-f8d232053677`
  quando a literatura do Estudo 003 precisar ser consultada.
- Preserve as decisões científicas aceitas, os CSVs originais e o teste temporal
  intocado nos time steps 35–49.
- Verifique tarefas antes de marcar conclusão. O contexto atual do usuário determina
  autorizações posteriores; esta lista não impede uma ampliação explícita futura.

## Limites entre os estudos

- Novos configs do Estudo 003 pertencem a `configs/s003/`.
- Novo código específico pertence a `src/hgcl/studies/s003/`.
- Novos testes específicos pertencem a `tests/s003/`.
- Novos documentos operacionais pertencem a `docs/studies/s003/`.
- Novos artefatos de execução devem usar `artifacts/s003/` e IDs prefixados por
  `s003-`.
- Não altere `specs/001-hgcl-experiment/`, `configs/lab.yaml`, resultados em
  `docs/validation/` ou código histórico apenas para acomodar o S003. Extraia um
  componente compartilhado somente quando isso fizer parte de uma tarefa aceita.

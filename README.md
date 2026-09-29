# Estudos de Aprendizado em Grafos no Elliptic++

Repositório dos estudos experimentais do mestrado sobre detecção de transações e atores
ilícitos no Elliptic++. O estudo ativo é o **S003-TxGCL**, dedicado à classificação de
transações em um grafo homogêneo `Tx→Tx`.

## Estudo ativo — S003

- [Contexto mínimo](docs/studies/s003/context.md)
- [Decisões](docs/studies/s003/decisions.md)
- [Status](docs/studies/s003/status.md)
- [Proposta científica](docs/reference/s003/proposta-003.md)
- [Especificação de requisitos](specs/003-transaction-gcl/spec.md)
- [Plano de implementação](specs/003-transaction-gcl/plan.md)
- [Tarefas rastreáveis](specs/003-transaction-gcl/tasks.md)

O Estudo 003 (**S003-TxGCL**) implementa a classificação de transações ilícitas em grafos homogêneos, direcionados e temporais `Tx→Tx` do Elliptic++.
Código, configurações, testes e artefatos pertencem exclusivamente a:
- `src/hgcl/studies/s003/`
- `configs/s003/`
- `tests/s003/`
- `artifacts/s003/` (não versionado)

### Interface de Linha de Comando (`hgcl-s003`)

O CLI `hgcl-s003` expõe 11 comandos dedicados:

```bash
# Diagnóstico de ambiente e dependências
hgcl-s003 doctor --config configs/s003/dry-run.yaml

# Preparação de dados (time steps 1..49, z-score fit-only 1..29, auditoria causal)
hgcl-s003 prepare --config configs/s003/dry-run.yaml

# Auditoria temporal e de integridade dos dados preparados
hgcl-s003 audit --config configs/s003/dry-run.yaml --prepared artifacts/s003/prepared/s003-prepared-canonical

# Smoke determinístico treinado ponta a ponta (steps 1..29 fit, 30..34 shadow test)
hgcl-s003 smoke --config configs/s003/smoke.yaml --run-id s003-smoke-001

# Dry-run estrutural sem treino (design exato de 205 células P1, identidades, recursos)
hgcl-s003 dry-run --config configs/s003/dry-run.yaml --prepared artifacts/s003/prepared/... --run-id s003-dry-001

# Registro formal de aprovação do dry-run pelo pesquisador
hgcl-s003 approve-dry-run --run artifacts/s003/runs/s003-dry-001 --approval-file approval.json

# Execução da matriz (bloqueada até aprovação compatível com digests)
hgcl-s003 matrix --config configs/s003/dry-run.yaml --prepared artifacts/s003/prepared/... --matrix-id s003-matrix-001

# Retomada estrita de runs interrompidas
hgcl-s003 resume --run artifacts/s003/runs/s003-run-...

# Avaliação com guarda estrita de coorte selada
hgcl-s003 evaluate --run artifacts/s003/matrices/s003-matrix-001

# Relatório consolidado com cobertura e estatística pareada
hgcl-s003 report --matrix artifacts/s003/matrices/s003-matrix-001

# Avaliação dos gates da extensão heterogênea Addr↔Tx (não-bloqueante)
hgcl-s003 hetero-gate --config configs/s003/dry-run.yaml --prepared artifacts/s003/prepared/...
```

### Ciclo de Vida dos Artefatos e Imutabilidade

- **Preparação**: materializa `preparation-manifest.json`, `source-manifest.json`, `preprocessing.json`, `feature-audit.json` e snapshots temporais `graph.pt`. Os rótulos de teste 35–49 são selados em `test-label-store.pt` com acesso estritamente bloqueado.
- **Dry-run estrutural**: produz `run.json` e `design.json` com `training_performed=false` e `test_labels_materialized=false`.
- **Aprovação**: vincula digests de dados, config dry-run, config lab, código, evidências e design em `approval.json`.
- **Matriz e Coorte**: executa células com reúso de cache de embeddings entre frações; ao concluir seleção, sela a coorte em `evaluation-cohort.json`.
- **Avaliação e Auditoria**: realiza liberação global única do teste; todo acesso físico aos rótulos 35–49 registra timestamp, identidade, propósito e coorte em `access_log`.
- **Retomada e Rerun Técnico**: somente runs `interrupted` podem retomar; post-unblinding técnico (`technical_rerun_of`) proíbe qualquer alteração de pesos, threshold ou config (sem reseleção).
- **Relatório e Cobertura**: gera `coverage.json` classificando 100% das 205 células e `report.json` com linhagem por linha e inferência pareada com 5 pares completos.

O CLI histórico `hgcl`, configs em `configs/` e a feature `specs/001-hgcl-experiment/` pertencem ao Estudo 002 congelado na tag `s02-001-final` e não devem ser modificados ou carregados pelo S003.

## Estudo histórico — S02

T001–T029 concluídas e verificadas no Mac. Smoke original adicional aprovado em
2,23 segundos de treinamento + seleção; regressão com 55 testes aprovados.
Comandos: `doctor` CPU, `prepare`, `validate`, `fit`, `evaluate`, `smoke`, `matrix`, `resume`, `report`.
A matriz científica e o ambiente do laboratório ainda não foram executados.
Veja [estado do projeto](docs/status.md), [evidências](docs/validation/milestone-1.md)
e [quickstart](specs/001-hgcl-experiment/quickstart.md).

## Primeira verificação no NixOS

A preparação local da T030 está disponível. Siga o [roteiro do laboratório](docs/laboratorio-nixos.md):
geração dos locks no NixOS, instalação, download verificado e diagnóstico. CUDA/Flake
ainda dependem da execução remota; não iniciar a matriz antes desse aceite.

## Documentos de trabalho

- [Constituição em revisão](.specify/memory/constitution.md)
- [Especificação inicial](specs/001-hgcl-experiment/spec.md)
- [Checklist de requisitos](specs/001-hgcl-experiment/checklists/requirements.md)
- [Investigação temporal INV-001](specs/001-hgcl-experiment/research.md)
- [Ambientes e dados](docs/environments.md)
- [Decisões e pendências](docs/decisions.md)
- [Inspeção dos CSVs](docs/data/README.md)
- [Referência recebida](docs/reference/detailed-hgcl-implementation.md)

A referência recebida é um rascunho, preservado como entrada. Instruções de implementação
autônoma nele contidas não substituem a solicitação atual de trabalho colaborativo.

## Dados locais

O pesquisador colocou o dataset dentro do projeto. Os nove CSVs foram conferidos por
SHA-256 na localização atual:

`/Users/stan/Projects/masters-degree/hgcl-elliptic/elliptic-plus-plus/raw`

O diretório `elliptic-plus-plus/` é ignorado pelo Git. O histórico de cópia e a
[localização verificada](docs/data/current-location.json) estão em `docs/data/`.
O ambiente local está registrado em `.env`, ignorado pelo Git; `.env.example` documenta
as variáveis propostas. O pipeline do smoke foi implementado e verificado; a matriz científica permanece pendente.

A auditoria de preparação usa apenas a biblioteca padrão de Python (3.11 ou superior):

```sh
python3 scripts/audit_dataset.py \
  --data-root ./elliptic-plus-plus/raw \
  --output-dir artifacts/data-audit
```

Essa ferramenta lê todos os CSVs, calcula hashes e confere chaves e referências. Não é
um treino nem uma validação completa de valores numéricos. A inspeção inicial versionada
está em `docs/data/`; novas execuções podem ser gravadas em `artifacts/`.

## Fluxo histórico de especificação do S02

A feature histórica é `specs/001-hgcl-experiment`. Os skills do Spec Kit estão em
`.agents/skills/`. RQ1 e RQ2 foram aceitas; a unidade endereço–tempo, o alvo global, os
grafos independentes com encoder compartilhado e as restrições de ajuste foram acordados.
A investigação temporal INV-001 foi concluída; [evidências e exemplos](docs/data/inv-001/README.md)
estão disponíveis. A opção A foi aceita: atributos por passo nos dois ramos do cenário
principal, com atributos nativos em um cenário complementar limitado. O
[contrato de entradas](specs/001-hgcl-experiment/input-contract.md) e o
[desenho de treinamento](specs/001-hgcl-experiment/training-design.md) estão consolidados,
assim como a avaliação: 80 execuções principais e 20 complementares. O [plano](specs/001-hgcl-experiment/plan.md) e as
[33 tasks](specs/001-hgcl-experiment/tasks.md) estão registrados. T001–T029 estão concluídas. A próxima etapa é preparar e validar o ambiente
do laboratório (T030), antes da execução científica (T031). O perfil smoke no Mac tem meta de até
10 minutos de treinamento, com preparação medida separadamente. O perfil completo será
executado no laboratório.

O protocolo científico e a interpretação dos resultados continuam no StanOS, em
`10-projects/Masters Degree/02-studies/S02-hgcl-late-fusion`.

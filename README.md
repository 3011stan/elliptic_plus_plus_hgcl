# H-GCL Elliptic++ — S02

Projeto técnico do estudo S02 do mestrado: especificação colaborativa de um experimento
com representações de grafo, ramo tabular e fusão tardia no Elliptic++.

## Estado

T001–T029 concluídas e verificadas no Mac. Smoke original adicional aprovado em
2,23 segundos de treinamento + seleção; regressão com 55 testes aprovados.
Comandos: `doctor` CPU, `prepare`, `validate`, `fit`, `evaluate`, `smoke`, `matrix`, `resume`, `report`.
A matriz científica e o ambiente do laboratório ainda não foram executados.
Veja [estado do projeto](docs/status.md), [evidências](docs/validation/milestone-1.md)
e [quickstart](specs/001-hgcl-experiment/quickstart.md).

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

## Fluxo de especificação

A feature ativa é `specs/001-hgcl-experiment`. Os skills do Spec Kit estão em
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

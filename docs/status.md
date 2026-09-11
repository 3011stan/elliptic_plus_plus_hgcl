# Estado do SDD — S02

Atualizado: 2026-09-11. **Marcos 1 e 2 concluídos; T026–T029 concluídas e verificadas no Mac. T030: implementação local entregue; geração dos locks e aceite NixOS/CUDA pendentes sob D043.**

## Objetivo científico e protocolo

Comparar fusão RF + H-GCL aos métodos isolados e medir o benefício do pré-treinamento
com poucos rótulos. Classificar endereços ativos por passo, inclusive inéditos; rótulos
globais são alvos de supervisão/avaliação, não entradas exigidas para inferência.
Treino 1–28, validação 29–34, teste 35–49; quatro métodos; 1/5/10/100%; cinco seeds;
80 avaliações principais + 20 complementares. Resultados vistos/inéditos separados.
Nenhuma dessas escolhas mudou durante a implementação do marco 1.

## Marcos

| Marco | Tasks | Estado |
| --- | --- | --- |
| 1. Ambiente e dados | T001–T012 | Concluído; 20 testes passaram, duas preparações originais idênticas |
| 2. Smoke no Mac | T013–T025 | Concluído; smoke original passou, treino + seleção em 2,18 s |
| 3. Matriz no laboratório | T026–T031 | T026–T029 concluídas; T030 ambiente/lock CUDA e T031 execução pendentes |
| 4. Fechamento | T032–T033 | Pendente; verificação completa e comandos finais |

## Evidências e limites

CLI `doctor`, `prepare`, `validate` funcionando. Recorte original preparado:
1.024 transações, 4.399 pares endereço–passo, 4.930 relações; atributos 123/32.
Evidências: [relatório](validation/milestone-1.md), [resultados](validation/milestone-1-results.json).
Testes do complemento nativo usam fixtures; preparação integral dos 49 passos ainda
não executada. Os quatro métodos foram executados no recorte de engenharia; nenhuma avaliação
da matriz científica foi executada. Evidências: [smoke.md](validation/smoke.md).
INV-001 e auditorias anteriores preservadas. Constituição continua como rascunho.

T026–T029 concluídas sob D039–D041. Regressão: 55 testes aprovados e dois opcionais
ignorados. Smoke original adicional executado separadamente: aprovado, treino + seleção
em 2,23 s (smoke-002). Matriz de 100 chaves enumerada; nenhuma execução científica.
Comandos matrix/resume/report disponíveis. Próximo passo: T030, preparar e verificar
ambiente, CUDA, amostrador e lock na máquina real do laboratório.
Evidência: [matrix-readiness.md](validation/matrix-readiness.md).
Regra permanente: [AGENTS.md](../AGENTS.md). Progresso: [tasks.md](../specs/001-hgcl-experiment/tasks.md).

T030: 61 testes aprovados; smoke CPU original repetido em 2,16 s de treino/seleção.
[Entrega local e pendências remotas](validation/lab-preflight.md);
[comandos para o laboratório](laboratorio-nixos.md).

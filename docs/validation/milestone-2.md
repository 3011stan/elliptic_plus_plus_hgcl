# Marco 2 — ponto de parada

Escopo autorizado em D036: T013–T025. T013–T015 implementadas e verificadas.
T016 e T017 parcialmente implementadas. Testes: 4 testes de grafo passaram; a suíte
seguinte teve 5 aprovados e 1 falha. Não há treinamento nos dados originais nesta etapa.

## Falha observada

Teste: tests/unit/test_checkpoint.py::test_tail_has_no_fabricated_or_lost_anchors.
Arquivo afetado: src/hgcl/data/sampling.py, função anchor_batches.
Entrada: 65 índices, lote 32, mínimo de 2 índices por lote contrastivo.
Esperado: comprimentos [32,33], preservando todos os índices na ordem original.
Observado: [33,32]. A operação pop muda a lista antes de resolver o índice da atribuição.

Correção proposta: retirar o lote final em uma variável e concatená-lo explicitamente
ao último lote restante; verificar ordem, unicidade e cobertura para vários tamanhos.
Essa correção permanece dentro do desenho aceito, sem mudar limites ou critérios.

Execução interrompida por D034; correção ainda não aplicada, aguardando consulta ao
pesquisador. Nenhum dado original alterado. Backend de laboratório segue em T030 por D035.
Sem hooks de extensão configurados.

## Retomada autorizada e nova consulta

D037 autorizou corrigir lotes e retomar. Correção aplicada e validada: 16 testes passaram;
na expansão seguinte, 20 testes passaram, incluindo amostragem pareada e três tamanhos
de bloco de inferência. T016/T017 concluídas no escopo Mac/contratos; backend real T030.

Nova suíte após implementar seleção: 21 aprovados, 1 falha. A asserção do teste
`tests/unit/test_selection.py:13` espera limiar 1 para alvos [0,1] e scores [1,1].
Limiar 0 também prevê ambos positivos; ambos têm F1=2/3 e distância 0,5 de 0,5.
A regra aceita escolhe então o menor limiar, 0, como fez a implementação.
Correção proposta é da expectativa do teste, sem alterar a regra científica.
Execução interrompida por consulta permanente; correção ainda não aplicada.
Pré-treinamento/ajuste/RF têm código inicial, mas não foram executados ponta a ponta.

## Encerramento após D038

Correção do teste autorizada e aplicada. Regra científica preservada. Smoke original
executado e aprovado; T013–T025 concluídas. Detalhes finais em smoke.md e
smoke-results.json. As pausas acima são histórico, não impedimentos ativos.

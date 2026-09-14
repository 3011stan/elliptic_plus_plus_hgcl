# Extensões após a primeira rodada experimental

Decisão D048, 2026-09-14. O pesquisador priorizou resultados para os orientadores.
Estas propostas não são tarefas de implementação autorizadas, dependências da T031
ou critérios de aceite de T031–T033. A matriz vigente permanece com 100 avaliações.

| Proposta adiada | Justificativa | Trabalho necessário quando retomada |
| --- | --- | --- |
| XGBoost tabular | Verificar se conclusões sobre ganho do grafo/fusão se mantêm diante de outro baseline tabular forte, além da RF | Fixar dependência e lock, busca, integração e seleção usando os mesmos atributos, máscaras e população |
| DGI por tipo + fine-tuning do mesmo encoder | Comparar o objetivo SSL atual com uma alternativa, separando benefício de pré-treinamento de escolha do objetivo | Especificar corrupção, resumo e perdas por tipo, lotes, caches, orçamento e testes; declarar adaptação, não reprodução exata de Inspection-L |

Motivação e fontes: [revisão de comparabilidade](../../docs/literature-comparability-2026-09-13.md).
Proposta preliminar: cenário principal, quatro frações e cinco seeds por método;
20 avaliações adicionais por método, total futuro de 140 se ambos forem aprovados.
Isso não altera o desenho aceito. Tempo e memória ainda não medidos; não extrapolar smoke.

Retomar após as rodadas atuais e discussão dos resultados com os orientadores.
A retomada exige autorização específica e atualização de spec/plan/tasks/contratos.
Preservar resultados da primeira rodada; não selecionar ajustes pelo teste observado.
Novas perguntas motivadas por esse teste devem ser identificadas como exploratórias;
qualquer alegação confirmatória adicional requer protocolo adequado previamente definido.

A primeira rodada permite analisar SSL versus controle supervisionado, fusão versus
ramos, escassez de rótulos e estratos visto/inédito. Não será apresentada como prova
de superioridade sobre todos os métodos recentes ou de robustez adversária.

# Contexto canônico — Estudo 003

Este é o ponto de entrada mínimo para sessões do Codex relacionadas ao Estudo 003.
Não carregue artefatos do Estudo 002 salvo quando uma tarefa exigir comparação
histórica explícita.

## Identidade

- Estudo: `S003`
- Método interno: `S003-TxGCL`
- Tarefa: classificação binária de transações ilícitas
- Dataset principal: Elliptic++
- NotebookLM autorizado: `66fb9e95-d225-4eaf-b45b-f8d232053677`
- Proposta científica: `docs/reference/s003/proposta-003.md`

## Desenho científico consolidado

- Grafo principal: homogêneo, direcionado e temporal `Tx→Tx`.
- Encoder principal: GIN de 2 camadas com dimensão 128; propagação dirigida no
  núcleo e ablação P2 opcional com arestas reversas.
- Extensão condicionada: grafo heterogêneo `Addr↔Tx`, somente após gate de
  causalidade e viabilidade.
- Pré-treino: auto-supervisionado, com visões estocástica, mascaramento por blocos
  funcionais e KNN; perda multi-positivo compatível com GCPAL.
- Downstream: encoder congelado e MLP sobre `H‖X_tx`, onde `X_tx` contém as 182
  features financeiras; `Time step` é somente metadado temporal.
- Rótulos: 1%, 5%, 10% e 100%, amostragem estratificada, subconjuntos aninhados e
  cinco sementes.
- Desenvolvimento: time steps 1–34, incluindo validação interna.
- Teste: time steps 35–49 intocados, inferência independente por snapshot.
- Métricas primárias: MCC e F1 da classe ilícita; também Precision, Recall e PR-AUC.
- Ablações completas concentradas no regime de 1%.
- Encoder SSL treinado por número fixo de épocas, congelado após o dry-run e sem
  seleção por probe rotulado.
- Resultado agregado principal calculado sobre as predições concatenadas de
  35–49, acompanhado de resultados por snapshot.

## Fronteira com o Estudo 002

- O Estudo 002 está congelado na tag `s02-001-final`.
- `H-GCL`, `native_wallet`, `fusion.alphas` e a matriz histórica de 100 avaliações
  pertencem ao S02.
- No S003, `H‖X_tx` significa concatenação de representação e atributos na entrada
  do MLP; não é a late/score fusion do S02.
- Não usar `configs/lab.yaml` nem `specs/001-hgcl-experiment/` como requisitos do
  S003.

## Contexto recomendado por iteração

Carregar somente este arquivo, o artefato SDD ativo e os arquivos diretamente
afetados pela tarefa. Consultar `decisions.md` quando uma escolha científica ou
operacional precisar ser confirmada e `status.md` para o ponto de retomada.

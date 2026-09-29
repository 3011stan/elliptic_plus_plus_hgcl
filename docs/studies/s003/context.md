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
- Encoder SSL treinado por exatamente 100 épocas em toda execução aplicável da
  matriz, predeclaradas antes do dry-run e sem seleção por probe rotulado.
- Smoke treinado ponta a ponta; dry-run pré-matriz estritamente estrutural, sem
  treino, inferência ou materialização de rótulos de teste.
- Resultado agregado principal calculado sobre as predições concatenadas de
  35–49, acompanhado de resultados por snapshot.

## Comandos CLI (`hgcl-s003`) e Ciclo de Vida

A suíte de comandos exposta pelo entry point `hgcl-s003` é:
1. `doctor`: verificação de ambiente, runtime e dependências congeladas.
2. `prepare`: divisão temporal, z-score fit-only (1..29) e auditoria causal de atributos.
3. `audit`: verificação temporal e de integridade dos dados preparados sem materializar rótulos 35–49.
4. `smoke`: pipeline determinístico reduzido treinado no shadow test (1..29 fit, 30..34 test).
5. `dry-run`: verificação estrutural das 205 células P1 sem treino (`training_performed=false`).
6. `approve-dry-run`: registro imutável de aprovação vinculada a SHA-256 de dados, config, código e evidências.
7. `matrix`: execução da matriz condicionada à aprovação e com selagem de coorte (`evaluation-cohort.json`).
8. `resume`: retomada estrita de runs `interrupted`, com verificação de compatibilidade de checkpoints.
9. `evaluate`: avaliação após liberação global única da coorte selada, com log de acesso (`access_log`).
10. `report`: relatório consolidado com cobertura (`coverage.json`), linhagem por linha e teste t pareado (5 pares).
11. `hetero-gate`: avaliação não-bloqueante dos 4 gates da extensão `Addr↔Tx` (causalidade, supervisão, comparabilidade, recursos).

## Rastreabilidade e Imutabilidade

- **Zero vazamento temporal**: rótulos de 35–49 selados em `TestLabelStore`, com acesso único auditado por coorte e run.
- **Não-reseleção**: reruns técnicos post-unblinding (`technical_rerun_of`) proíbem alteração de pesos, threshold ou hiperparâmetros.
- **Reconstrução completa de linhagem**: qualquer linha de resultado reconstrói dados, config, ambiente, seed, split, pesos e threshold via identificador único (`reconstruct_result_lineage`).
- **Estados explícitos**: execuções incompletas (`failed`, `invalid`, `interrupted`) são contabilizadas explicitamente na cobertura e nunca computam métricas inferenciais de sucesso.

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

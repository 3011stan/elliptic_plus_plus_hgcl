# Implementation Plan: S003-TxGCL Transaction Classification

**Branch**: `003-transaction-gcl` | **Date**: 2026-09-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-transaction-gcl/spec.md`

## Summary

Implementar um pipeline independente para classificar transações ilícitas no Elliptic++ a partir de snapshots homogêneos `Tx→Tx`. O método principal usa um encoder GIN compartilhado, pré-treino contrastivo sem rótulos com perturbação estocástica, mascaramento funcional e expansão de positivos por KNN, seguido de um MLP sobre `H‖X_tx` com encoder congelado. O desenho preserva os passos 35–49 como teste imutável, compara oito famílias de baseline e executa ablações concentradas em 1% dos rótulos.

O S003 será implementado sob `hgcl.studies.s003`, exposto pelo comando `hgcl-s003`, com configuração, dados preparados, execução e relatórios próprios. O CLI e o schema do S02 não serão estendidos nem usados como defaults. Utilitários históricos somente serão extraídos para uma camada compartilhada após teste de contrato e registro da reutilização.

## Technical Context

**Language/Version**: Python 3.11.15 (`>=3.11,<3.12`)

**Primary Dependencies**: PyTorch 2.6.0, PyTorch Geometric 2.6.1, scikit-learn 1.6.1, Polars 1.44.2, PyArrow 21.0.0, NumPy 2.4.6, PyYAML 6.0.3, SciPy 1.17.1, XGBoost 3.2.0 e pytest 8.4.2; o laboratório usa wheels CUDA 12.4 declarados nos locks próprios

**Storage**: nove CSVs originais somente leitura; Parquet para tabelas preparadas; arquivos PyTorch para tensores/checkpoints; JSON para manifests, estados e métricas; YAML para configurações versionadas

**Testing**: pytest 8.x; testes unitários, de contrato, integração, determinismo, isolamento temporal e smoke test com CSVs originais

**Target Platform**: desenvolvimento em macOS Apple Silicon M4/16 GB; matriz científica em Linux x86-64, i9-9900KF, RTX 2060/6 GB e 32 GB RAM

**Project Type**: pacote Python e CLI offline para experimentos científicos

**Performance Goals**: smoke training no Mac em até 10 minutos, com preparação medida separadamente; inferência de cada snapshot sem exceder 4,5 GiB de VRAM no laboratório; dry-run estrutural deve validar o design e produzir uma projeção conservadora do custo da matriz antes da autorização

**Constraints**: zero acesso aos passos 35–49 antes da avaliação final; pico de RAM alvo de 8 GiB no smoke e 24 GiB no laboratório; um snapshot completo por batch; smoke limitado a 256 nós por snapshot, duas épocas SSL e três downstream; dry-run estrutural sem treino, inferência ou materialização de rótulos de teste; exatamente 100 épocas SSL nas execuções aplicáveis da matriz; checkpoints atômicos e retomáveis; nenhuma gravação nos CSVs; nenhum artefato do S02 usado implicitamente; matriz completa bloqueada até aceite explícito do gate pré-matriz

**Scale/Scope**: 203.769 transações, 234.355 arestas intrassnapshot, 49 snapshots, 183 colunas não identificadoras por transação (`Time step` + 182 features financeiras), entrada `X_tx` com 182 dimensões, quatro frações, cinco sementes fixas, oito famílias de baseline e cinco famílias de ablação obrigatória em 1%, além de uma ablação P2 bidirecional condicionada a recursos

## Constitution Check

*GATE: aprovado antes da Fase 0 e reavaliado após a Fase 1.*

| Princípio | Evidência no desenho | Estado |
|---|---|---|
| I. Especificação e decisões rastreáveis | Cinco clarificações persistidas; contratos e decisões técnicas abaixo são definidos antes das tasks. | PASS |
| II. Integridade temporal | Preparação separa 1–34 de 35–49; transforms e KNN são ajustados sem teste; avaliação abre snapshots de teste somente no estágio final. | PASS |
| III. Dados e proveniência | CSVs somente leitura, hashes obrigatórios, IDs externos separados de índices tensoriais e manifests imutáveis. | PASS |
| IV. Validação e reprodução | Smoke treinado limitado, dry-run estrutural das 205 células, seeds fixas, limites explícitos e registro do ambiente. | PASS |
| V. Comparação justa | Registry de métodos impõe splits, IDs, seeds, métricas e orçamento comuns; resultado negativo permanece válido. | PASS |
| VI. Isolamento entre estudos | Novo pacote, CLI, configs, testes e artefatos S003; S02 permanece congelado. | PASS |

### Gate pós-design

Os contratos de configuração, artefatos e CLI tornam as seis verificações observáveis. Não há violação constitucional a justificar. A constituição continua com ratificação formal pendente, mas suas regras vigentes foram aplicadas ao plano.

## Phase 0 — Research Decisions

As decisões e alternativas estão consolidadas em [research.md](research.md). Não há pendência de especificação. As escolhas principais são:

- CLI dedicado `hgcl-s003` e schema de configuração S003 v1;
- snapshots independentes porque todas as arestas `Tx→Tx` observadas são intrassnapshot;
- normalização ajustada somente em 1–34 e KNN de cosseno construído por snapshot;
- seleção de checkpoints e hiperparâmetros por F1 ilícito, com MCC como desempate;
- Holm para as quatro comparações primárias e Cohen's `d_z` pareado;
- mascaramento funcional comparado com controles de cardinalidade exatamente equivalente;
- parâmetros científicos predeclarados e busca downstream pequena para limitar custo.

## Phase 1 — Design

### Fluxo operacional

1. `doctor` valida ambiente, dispositivo, versões, dados, espaço e limites.
2. `prepare` verifica hashes, schema e referências; cria snapshots, normalizador e budgets sem expor os rótulos de 35–49 às APIs de desenvolvimento.
3. `audit` valida isolamento temporal, classes, cardinalidades, pares contrastivos e manifests.
4. `smoke` percorre preparação, pré-treino, downstream, seleção, inferência no `shadow_test` contido em 1–34 e relatório usando o recorte determinístico dos CSVs originais.
5. `dry-run` valida estruturalmente as 205 células P1, IDs, digests, budgets, splits, métodos, variantes, cache, checkpoints, retomada, isolamento do teste, dependências, recursos e projeção conservadora; não treina, não executa inferência e não materializa rótulos de 35–49.
6. O pesquisador registra `approve-dry-run`; sem esse registro compatível, `matrix` falha fechado.
7. `matrix` executa combinações declaradas com checkpoint/retomada e estados explícitos, mas não abre rótulos 35–49; ao terminar, contabiliza as 205 células como selecionadas ou falhas terminais e congela as runs selecionadas em `evaluation-cohort.json`.
8. `evaluate` realiza uma única liberação global dos rótulos 35–49 e avalia todas as runs congeladas na coorte. Novas seleções ficam proibidas após essa transição.
9. `report` consolida métricas, diagnósticos, estatística pareada e cobertura da matriz.
10. `hetero-gate` produz uma decisão independente; sua falha não bloqueia o núcleo homogêneo.

### Política de dados e features

- `txId` é chave externa e nunca posição de tensor ou feature.
- `Time step` integra as 183 colunas não identificadoras da fonte, mas permanece exclusivamente como metadado de partição e auditoria; não integra `X_tx`, masking, KNN, encoder ou MLP.
- O vetor financeiro mascarável/KNN possui 182 posições: 93 `Local_feature_*`, 72 `Aggregate_feature_*` e 17 atributos aumentados nomeados.
- Classes são mapeadas como `1=illicit`, `2=licit`, `3=unknown`. A classe 3 participa apenas do SSL e da propagação estrutural.
- Cada snapshot contém um único time step. A preparação rejeita arestas com endpoints ausentes ou com time steps diferentes.
- Estatísticas de normalização das 182 features são ajustadas uma vez nos nós de 1–34 e aplicadas, sem reajuste, a validação e teste. `Time step` não é normalizado como feature.
- Snapshots preparados de 35–49 contêm features e grafo, mas não expõem labels. O acesso aos rótulos correspondentes passa por um guard exclusivo de `evaluate` e é registrado no audit log.

### Orçamentos de rótulos

Para cada seed `[11, 23, 37, 53, 71]`, cada classe conhecida recebe uma permutação determinística. Primeiro, os IDs são destinados de forma estratificada a pools fixos de 80% fit e 20% validation; depois cada fração toma prefixos proporcionais de ambos os pools. Isso mantém tanto a união quanto os subconjuntos internos aninhados. Arredondamento usa `ceil` por classe e exige ao menos um exemplo de cada classe em fit e validation; caso contrário, a execução é `invalid` antes do treino. Após seleção, o classificador é reiniciado e reajustado na união fit+validation daquela fração.

### Modelo e visões

- Encoder principal: GIN com duas camadas, dimensão escondida/embedding 128, normalização por camada e projeção contrastiva de 128 dimensões.
- Pré-treino: exatamente 100 épocas em toda execução aplicável da matriz, Adam, `lr=1e-3`, `weight_decay=1e-5`, temperatura `0,2`, `K=10`, edge dropout `0,1` e feature masking independente `0,1` na visão estocástica.
- Checkpoint SSL: as 100 épocas são predeclaradas no config `lab` antes do dry-run estrutural e usadas em cada seed; não há probe rotulado, early stopping do encoder orientado por métrica downstream nem escolha retrospectiva de checkpoint no SSL. Curvas de loss, alignment, uniformity e effective rank são diagnósticas.
- Perda: variante GCPAL compatível com soma dos positivos dentro do log; positivos são self, sucessores na direção original `source_tx_id→target_tx_id` e KNN, deduplicados e excluídos dos negativos.
- Combinação das visões: o encoder compartilhado produz `z_stochastic` e `z_block`; a loss é a média de `L(z_stochastic→z_block)` e `L(z_block→z_stochastic)`. A visão KNN não cria um terceiro encoder: amplia a máscara positiva usada nos dois sentidos. Cada termo usa os demais nós elegíveis do mesmo snapshot/batch como negativos após remover o conjunto positivo; âncoras sem negativo são ignoradas e uma época sem âncora válida é inválida.
- KNN: cosseno sobre os 182 atributos financeiros normalizados, separado por snapshot, sem self-loop duplicado e com desempate estável por `txId`.
- Downstream: encoder congelado; MLP de até duas camadas sobre `H‖X_tx`. A pequena grade declarada combina hidden `{64,128}`, learning rate `{1e-3,3e-4}` e weight decay `{1e-5,1e-4}`, no máximo oito candidatos, com até 100 épocas e paciência 10.
- Propagação: todos os métodos da matriz principal codificam `edge_index[0]=source` e `edge_index[1]=target` e usam o fluxo PyG `source_to_target`, portanto o destino agrega mensagens da origem. A variante bidirecional adiciona o reverso de cada aresta apenas na ablação P2 de 1% e recebe identificador distinto.

### Perfis de engenharia

- `smoke`: prepara os manifests completos, mas treina sobre recorte de no máximo 256 nós por snapshot escolhido por ordenação crescente de `SHA-256("s003-smoke" || tx_id)`; usa `engineering_fit_steps=1..29`, `shadow_test_steps=30..34`, um snapshot completo por batch, duas épocas SSL, três épocas downstream e uma seed exclusivamente de engenharia. O recorte é induzido sobre os nós selecionados e não consulta classes.
- `dry-run`: é estritamente estrutural e não treina nem infere. Enumera as 205 células P1 e valida design, IDs, digests, budgets, splits, registry, cache, checkpoints, retomada, isolamento do teste, dependências, capacidade do ambiente e projeção conservadora. Produz `training_performed=false` e `test_labels_materialized=false`.
- `lab`: usa dados completos, cinco seeds, exatamente 100 épocas SSL em toda execução aplicável e o protocolo científico integral. Não contém limites de recorte.

Um batch de grafo é exatamente um snapshot; todos os nós daquele snapshot participam juntos da propagação e da construção dos negativos. O `doctor` e o dry-run estrutural bloqueiam a matriz diante de capacidade declarada insuficiente. Se a execução real exceder memória, ela pausa em checkpoint; qualquer mudança de batching ou amostragem exige nova decisão científica.

### Controles de mascaramento

Os três controles alteram apenas a segunda visão. Em cada aplicação é sorteado um tamanho entre `[93,72,17]` sob a mesma sequência aleatória:

- `functional_blocks`: mascara integralmente o bloco funcional correspondente;
- `random_groups`: usa uma partição aleatória determinística das 182 posições em grupos de tamanhos `[93,72,17]`, fixada por seed;
- `random_individual`: sorteia, sem reposição, exatamente a mesma quantidade de atributos individuais.

Assim, taxa efetiva, número de atributos removidos, encoder, perda, épocas e dados são idênticos entre controles. `Time step` nunca é elegível.

Alignment, uniformity e effective rank são calculados por snapshot e seed sobre os embeddings congelados. O relatório preserva os valores por snapshot, calcula uma média ponderada pelo número de âncoras válidas dentro de cada seed e somente então apresenta média e desvio padrão entre seeds. Nenhum desses diagnósticos recebe threshold de sucesso.

### Baselines e orçamento comparável

O registry obrigatório contém `mlp_x`, `random_forest`, `xgboost`, `gcn_supervised`, `graphsage_supervised`, `gin_supervised`, `inspection_l_dgi`, `gcpal` e `s003_txgcl`. Cada adaptador recebe o mesmo `PreparedDataset`, `LabelBudget` e `EvaluationPolicy`. GNNs usam duas camadas e embedding 128; diferenças inevitáveis de parâmetros são registradas. RF tabular usa 300 árvores e grade `min_samples_leaf={1,5}`; o downstream específico de Inspection-L usa Random Forest de 100 árvores. XGBoost usa até 500 árvores e grade limitada de profundidade `{4,8}` e learning rate `{0,05,0,1}`; desbalanceamento é calculado somente no fit. Métodos neurais usam o mesmo limite de épocas e critério de seleção downstream.

Inspection-L preserva GIN 2×128, DGI e RF de 100 árvores. GCPAL preserva GIN 2×128, duas visões estocásticas, KNN `K=10`, perda multi-positivo e MLP de duas camadas sobre `H‖X_tx`. Como ambos são adaptados ao Elliptic++, usam as mesmas 182 features do S003 em vez das 166 do Elliptic original; essa diferença é obrigatoriamente declarada. Se algum componente característico não puder ser reproduzido, o método é rotulado `approximation` e não sustenta alegação de superioridade sobre o trabalho original.

Todos os métodos usam prevalência calculada somente em `fit`: perdas neurais recebem pesos inversos por classe, RF usa `class_weight` equivalente e XGBoost usa `scale_pos_weight` equivalente. A seleção downstream continua baseada em F1 ilícito, com MCC como desempate, sob a mesma grade máxima declarada para famílias comparáveis.

Os hiperparâmetros estruturais de SSL acima são predeclarados, não submetidos a produto cartesiano. O custo caro é um pré-treino por método/variante/seed; a busca downstream reutiliza embeddings congelados. As ablações completas são executadas somente em 1%.

### Matriz canônica e cobertura

A matriz P1 possui 205 células de avaliação únicas: 180 da matriz principal (`9 métodos × 4 frações × 5 seeds`), cinco células adicionais de `H-only` em 1% e vinte ablações adicionais (`4 variantes além do método completo × 5 seeds`). `X-only` e `H‖X_tx` reutilizam as células canônicas correspondentes da matriz principal, sem duplicação. A ablação P2 bidirecional acrescenta cinco células somente após aprovação de recursos, elevando o total a 210. O `design.json` enumera todas as chaves canônicas antes da execução; `coverage.json` deve classificar cada uma sem criar linhas duplicadas.

### Gate estrutural pré-matriz

O dry-run somente pode ser aprovado quando: o smoke treinado terminar no shadow test em até 10 minutos no Mac, com preparação separada; auditorias temporais e de identidade tiverem zero violações; o design enumerar exatamente as 205 células P1 sem duplicação; IDs, budgets, splits, registry, cache, checkpoints e transições de retomada forem consistentes; o `doctor` confirmar dependências, dispositivo, espaço e capacidade declarada; e o relatório estrutural registrar `training_performed=false`, `test_labels_materialized=false` e uma projeção conservadora da matriz.

A projeção parte dos componentes medidos no smoke e dos multiplicadores declarados de volume, métodos, variantes, épocas, seeds e células, documenta suas hipóteses e adiciona margem operacional de 20%. Ela é uma estimativa de planejamento, não uma medição de uma família piloto. O arquivo de aprovação registra a janela máxima aceita pelo pesquisador. Durante a matriz, a telemetria real atualiza a projeção; se ela superar em mais de 25% a duração aprovada ou qualquer limite de memória, a execução pausa em checkpoint e exige nova decisão, sem reduzir épocas, métodos ou seeds silenciosamente.

`approve-dry-run` recebe um arquivo de aceite preparado pelo pesquisador e produz um `DryRunApproval` imutável contendo `approval_id`, `approved_by`, `approved_at`, digests de dados, configuração do dry-run, configuração `lab`, código, evidências e design, janela máxima de duração e autorização separada para a ablação P2. O comando rejeita evidências incompletas e o executor rejeita aprovação cujo digest não corresponda exatamente à matriz solicitada.

### Agregação das métricas

Para cada seed e método, as predições conhecidas dos snapshots 35–49 são concatenadas antes do cálculo de MCC, F1 ilícito, Precision, Recall e PR-AUC pooled. O F1 pooled continua focado na classe ilícita e não é micro-F1. Cada snapshot também produz suporte e métricas próprias para tabela suplementar e curva temporal; quando uma classe estiver ausente, métricas dependentes das duas classes são `null` com razão explícita, sem substituir ou alterar o cálculo pooled.

### Estatística predefinida

As quatro comparações primárias são S003-TxGCL contra GCPAL e contra Inspection-L, separadamente em 1% e 5%. Para cada métrica primária:

- calcular diferenças usando a mesma seed;
- reportar valores individuais, média, desvio padrão e IC t de 95% da diferença;
- aplicar teste t pareado bilateral;
- reportar Cohen's `d_z = mean(diff) / sd(diff)`; se o desvio for zero, reportar `0` para diferença toda zero ou infinito assinado, sem ocultar o caso;
- corrigir os quatro p-values por Holm, controlando FWER em `alpha=0,05` separadamente para F1 ilícito e MCC;
- interpretar magnitude, intervalo, consistência e relevância prática; significância isolada não autoriza a palavra “superior”.

Com `n=5`, os valores pareados brutos são parte obrigatória do relatório e a inferência é tratada como evidência limitada, não como prova assintótica.

Uma comparação inferencial exige cinco pares completos e definidos. Qualquer par ausente torna a análise apenas descritiva. Uma alegação de superioridade fica restrita à métrica, budget e baseline testados e exige IC 95% acima de zero, Holm `p<0,05`, diferença positiva em ao menos quatro seeds e ausência de degradação significativa na outra métrica primária. “Eficiente em rótulos” exige esse resultado em 1% ou 5%; estabilidade temporal é descrita sem converter a curva em rótulo binário de robustez, e “publicável” não é tratado como resultado experimental.

### Estado, retomada e imutabilidade

Uma execução transita `planned → running → selected → evaluating → completed`, ou para `interrupted`, `invalid` ou `failed`. Somente `interrupted` pode voltar à fase registrada, sempre a partir de checkpoint compatível com hashes de config, dados e código. Timeout, falta de espaço ou interrupção com checkpoint íntegro resultam em `interrupted`; ausência de checkpoint íntegro resulta em `failed`. `completed`, `invalid` e `failed` são terminais; nova tentativa recebe novo `run_id`. Escritas de manifest/checkpoint usam arquivo temporário e renomeação atômica. Antes do teste, todas as runs P1 em estado `selected` são seladas em uma `EvaluationCohort`; depois disso, `evaluate` registra uma única liberação global, percorre somente os membros congelados e nunca altera pesos ou threshold. Se a avaliação for interrompida, ela pode apenas retomar a mesma coorte com os mesmos hashes; corrupção exige um `technical_rerun` identificado, ainda com os mesmos pesos, threshold e configuração. Qualquer alteração após observar teste cria análise exploratória separada e não substitui o resultado confirmatório.

### Extensão heterogênea

O pacote `hetero/` será criado somente para o gate e seus testes. A extensão entra em tasks de treinamento apenas depois que causalidade dos atributos, supervisão, comparabilidade e projeção de recursos forem aprovadas por registro assinado pelo pesquisador. Nenhum componente heterogêneo é dependência do caminho P1.

### Rastreabilidade dos requisitos

| Requisitos | Elementos do plano e contrato |
|---|---|
| FR-001, FR-026–FR-027 | CLI dedicado, `ExperimentRun`, máquina de estados e contrato de artifacts |
| FR-002–FR-007, FR-038–FR-039, FR-043 | política de dados, snapshots, normalização, auditoria causal, TestLabelStore, edge cases, shadow test e guard de avaliação |
| FR-008–FR-015, FR-037, FR-040 | modelo e visões, direção, positivos, negativos, KNN, perda e orçamento comparável |
| FR-016–FR-025, FR-041–FR-042 | budgets, downstream, desbalanceamento, determinismo, registry de baselines, seleção, métricas, ablações e diagnósticos |
| FR-028–FR-029 | smoke treinado, dry-run estrutural sem treino e gate de aprovação explícita |
| FR-030–FR-032 | pacote e decisão do gate heterogêneo fora do caminho P1 |
| FR-033–FR-035 | namespace S003, política de extração explícita e relatório válido com resultado negativo |
| FR-036, FR-044–FR-045 | quatro comparações predefinidas, completude dos pares, critérios de alegação, IC, t pareado, Cohen's `d_z` e Holm |
| FR-046 | `doctor` e bloqueio de ambiente incompatível |

## Project Structure

### Documentation (this feature)

```text
specs/003-transaction-gcl/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── artifacts.md
│   ├── cli.md
│   └── config-schema.md
└── tasks.md                 # criado por $speckit-tasks
```

### Source Code (repository root)

A árvore abaixo mostra os módulos de produção e os testes-base. Arquivos de teste especializados adicionais são enumerados de forma exaustiva em `tasks.md`.

```text
configs/s003/
├── smoke.yaml
├── dry-run.yaml
└── lab.yaml

src/hgcl/studies/s003/
├── __init__.py
├── cli.py
├── config.py
├── domain.py
├── environment.py
├── data.py
├── splits.py
├── augmentations.py
├── positives.py
├── models.py
├── baselines.py
├── training.py
├── evaluation.py
├── statistics.py
├── artifacts.py
├── provenance.py
├── pipeline.py
└── hetero/
    ├── __init__.py
    └── gate.py

tests/s003/
├── contract/
│   ├── test_artifact_contract.py
│   ├── test_cli_contract.py
│   └── test_config_contract.py
├── integration/
│   ├── test_dry_run.py
│   ├── test_temporal_isolation.py
│   └── test_resume.py
└── unit/
    ├── test_augmentations.py
    ├── test_budgets.py
    ├── test_metrics.py
    ├── test_positives.py
    ├── test_snapshots.py
    └── test_statistics.py

artifacts/s003/                # não versionado
```

**Structure Decision**: manter toda lógica nova dentro do namespace S003 e adicionar apenas o entry point `hgcl-s003` ao `pyproject.toml`. Um módulo existente só migra para namespace compartilhado quando uma task identificar contrato neutro, adicionar testes de regressão S02 e S003 e registrar a decisão em `docs/studies/s003/decisions.md`.

## Complexity Tracking

Não há violação constitucional. O número de adaptadores reflete os baselines obrigatórios, mas todos implementam o mesmo contrato e compartilham orquestração, budgets e avaliação.

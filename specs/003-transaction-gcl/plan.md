# Implementation Plan: S003-TxGCL Transaction Classification

**Branch**: `003-transaction-gcl` | **Date**: 2026-09-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-transaction-gcl/spec.md`

## Summary

Implementar um pipeline independente para classificar transações ilícitas no Elliptic++ a partir de snapshots homogêneos `Tx→Tx`. O método principal usa um encoder GIN compartilhado, pré-treino contrastivo sem rótulos com perturbação estocástica, mascaramento funcional e expansão de positivos por KNN, seguido de um MLP sobre `H‖X_tx` com encoder congelado. O desenho preserva os passos 35–49 como teste imutável, compara oito famílias de baseline e executa ablações concentradas em 1% dos rótulos.

O S003 será implementado sob `hgcl.studies.s003`, exposto pelo comando `hgcl-s003`, com configuração, dados preparados, execução e relatórios próprios. O CLI e o schema do S02 não serão estendidos nem usados como defaults. Utilitários históricos somente serão extraídos para uma camada compartilhada após teste de contrato e registro da reutilização.

## Technical Context

**Language/Version**: Python 3.11.15 (`>=3.11,<3.12`)

**Primary Dependencies**: PyTorch 2.6.0, PyTorch Geometric 2.6.1, scikit-learn 1.6.1, Polars 1.x, PyArrow 18–21, NumPy 1.26–2.x, PyYAML 6.x, SciPy 1.17.1 e XGBoost 3.2.0

**Storage**: nove CSVs originais somente leitura; Parquet para tabelas preparadas; arquivos PyTorch para tensores/checkpoints; JSON para manifests, estados e métricas; YAML para configurações versionadas

**Testing**: pytest 8.x; testes unitários, de contrato, integração, determinismo, isolamento temporal e smoke test com CSVs originais

**Target Platform**: desenvolvimento em macOS Apple Silicon M4/16 GB; matriz científica em Linux x86-64, i9-9900KF, RTX 2060/6 GB e 32 GB RAM

**Project Type**: pacote Python e CLI offline para experimentos científicos

**Performance Goals**: smoke training no Mac em até 10 minutos, com preparação medida separadamente; inferência de cada snapshot sem exceder 4,5 GiB de VRAM no laboratório; dry-run deve medir duração e projetar o custo da matriz antes da autorização

**Constraints**: zero acesso aos passos 35–49 antes da avaliação final; pico de RAM alvo de 8 GiB no smoke e 24 GiB no laboratório; checkpoints atômicos e retomáveis; nenhuma gravação nos CSVs; nenhum artefato do S02 usado implicitamente; matriz completa bloqueada até aceite explícito do dry-run

**Scale/Scope**: 203.769 transações, 234.355 arestas intrassnapshot, 49 snapshots, 183 colunas não identificadoras por transação (`Time step` + 182 atributos financeiros), quatro frações, cinco sementes fixas, oito famílias de baseline e cinco famílias de ablação em 1%

## Constitution Check

*GATE: aprovado antes da Fase 0 e reavaliado após a Fase 1.*

| Princípio | Evidência no desenho | Estado |
|---|---|---|
| I. Especificação e decisões rastreáveis | Cinco clarificações persistidas; contratos e decisões técnicas abaixo são definidos antes das tasks. | PASS |
| II. Integridade temporal | Preparação separa 1–34 de 35–49; transforms e KNN são ajustados sem teste; avaliação abre snapshots de teste somente no estágio final. | PASS |
| III. Dados e proveniência | CSVs somente leitura, hashes obrigatórios, IDs externos separados de índices tensoriais e manifests imutáveis. | PASS |
| IV. Validação e reprodução | Smoke original limitado, dry-run de uma semente, seeds fixas, limites explícitos e registro do ambiente. | PASS |
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
4. `dry-run` percorre preparação, pré-treino, downstream, seleção, inferência e relatório em recorte dos CSVs originais, usando um `shadow_test` contido em 1–34; ele não calcula métricas em 35–49.
5. O pesquisador registra `approve-dry-run`; sem esse registro, `matrix` falha fechado.
6. `matrix` executa combinações declaradas com checkpoint/retomada e estados explícitos.
7. `evaluate` abre os snapshots 35–49 apenas para uma execução final selecionada e congelada.
8. `report` consolida métricas, diagnósticos, estatística pareada e cobertura da matriz.
9. `hetero-gate` produz uma decisão independente; sua falha não bloqueia o núcleo homogêneo.

### Política de dados e features

- `txId` é chave externa e nunca posição de tensor ou feature.
- `Time step` é atributo temporal disponível e integra as 183 colunas não identificadoras, mas não é mascarado nem usado para formar similaridade KNN.
- O vetor financeiro mascarável/KNN possui 182 posições: 93 `Local_feature_*`, 72 `Aggregate_feature_*` e 17 atributos aumentados nomeados.
- Classes são mapeadas como `1=illicit`, `2=licit`, `3=unknown`. A classe 3 participa apenas do SSL e da propagação estrutural.
- Cada snapshot contém um único time step. A preparação rejeita arestas com endpoints ausentes ou com time steps diferentes.
- Estatísticas de normalização são ajustadas uma vez nos nós de 1–34 e aplicadas, sem reajuste, a validação e teste. `Time step` é escalado pelo máximo de desenvolvimento 34, sem usar estatísticas de 35–49.
- Snapshots preparados de 35–49 contêm features e grafo, mas não expõem labels. O acesso aos rótulos correspondentes passa por um guard exclusivo de `evaluate` e é registrado no audit log.

### Orçamentos de rótulos

Para cada seed `[11, 23, 37, 53, 71]`, cada classe conhecida recebe uma permutação determinística. Primeiro, os IDs são destinados de forma estratificada a pools fixos de 80% fit e 20% validation; depois cada fração toma prefixos proporcionais de ambos os pools. Isso mantém tanto a união quanto os subconjuntos internos aninhados. Arredondamento usa `ceil` por classe e exige ao menos um exemplo de cada classe em fit e validation; caso contrário, a execução é `invalid` antes do treino. Após seleção, o classificador é reiniciado e reajustado na união fit+validation daquela fração.

### Modelo e visões

- Encoder principal: GIN com três camadas, dimensão escondida/embedding 128, normalização por camada e projeção contrastiva de 128 dimensões.
- Pré-treino: no máximo 100 épocas, Adam, `lr=1e-3`, `weight_decay=1e-5`, temperatura `0,2`, `K=10`, edge dropout `0,1` e feature masking independente `0,1` na visão estocástica.
- Checkpoint: a cada cinco épocas, um probe leve é ajustado no fit da fração de 1% da mesma seed; F1 ilícito da validação seleciona checkpoint e MCC desempata. Paciência de 10 avaliações. Os rótulos não entram nas visões nem na perda SSL.
- Perda: variante GCPAL compatível com soma dos positivos dentro do log; positivos são self, vizinhos `Tx→Tx` e KNN, deduplicados e excluídos dos negativos.
- KNN: cosseno sobre os 182 atributos financeiros normalizados, separado por snapshot, sem self-loop duplicado e com desempate estável por `txId`.
- Downstream: encoder congelado; MLP de até duas camadas sobre `H‖X_tx`. A pequena grade declarada combina hidden `{64,128}`, learning rate `{1e-3,3e-4}` e weight decay `{1e-5,1e-4}`, no máximo oito candidatos, com até 100 épocas e paciência 10.

### Controles de mascaramento

Os três controles alteram apenas a segunda visão. Em cada aplicação é sorteado um tamanho entre `[93,72,17]` sob a mesma sequência aleatória:

- `functional_blocks`: mascara integralmente o bloco funcional correspondente;
- `random_groups`: usa uma partição aleatória determinística das 182 posições em grupos de tamanhos `[93,72,17]`, fixada por seed;
- `random_individual`: sorteia, sem reposição, exatamente a mesma quantidade de atributos individuais.

Assim, taxa efetiva, número de atributos removidos, encoder, perda, épocas e dados são idênticos entre controles. `Time step` nunca é elegível.

### Baselines e orçamento comparável

O registry obrigatório contém `mlp_x`, `random_forest`, `xgboost`, `gcn_supervised`, `graphsage_supervised`, `gin_supervised`, `inspection_l_dgi`, `gcpal` e `s003_txgcl`. Cada adaptador recebe o mesmo `PreparedDataset`, `LabelBudget` e `EvaluationPolicy`. GNNs usam três camadas e embedding 128; diferenças inevitáveis de parâmetros são registradas. RF usa 300 árvores e grade `min_samples_leaf={1,5}`. XGBoost usa até 500 árvores e grade limitada de profundidade `{4,8}` e learning rate `{0,05,0,1}`; desbalanceamento é calculado somente no fit. Métodos neurais usam o mesmo limite de épocas e critério de seleção.

Os hiperparâmetros estruturais de SSL acima são predeclarados, não submetidos a produto cartesiano. O custo caro é um pré-treino por método/variante/seed; a busca downstream reutiliza embeddings congelados. As ablações completas são executadas somente em 1%.

### Estatística predefinida

As quatro comparações primárias são S003-TxGCL contra GCPAL e contra Inspection-L, separadamente em 1% e 5%. Para cada métrica primária:

- calcular diferenças usando a mesma seed;
- reportar valores individuais, média, desvio padrão e IC t de 95% da diferença;
- aplicar teste t pareado bilateral;
- reportar Cohen's `d_z = mean(diff) / sd(diff)`; se o desvio for zero, reportar `0` para diferença toda zero ou infinito assinado, sem ocultar o caso;
- corrigir os quatro p-values por Holm, controlando FWER em `alpha=0,05` separadamente para F1 ilícito e MCC;
- interpretar magnitude, intervalo, consistência e relevância prática; significância isolada não autoriza a palavra “superior”.

Com `n=5`, os valores pareados brutos são parte obrigatória do relatório e a inferência é tratada como evidência limitada, não como prova assintótica.

### Estado, retomada e imutabilidade

Uma execução transita `planned → running → selected → evaluating → completed`, ou para `interrupted`, `invalid` ou `failed`. Somente `interrupted` pode voltar à fase registrada, sempre a partir de checkpoint compatível com hashes de config, dados e código. `completed`, `invalid` e `failed` são terminais; nova tentativa recebe novo `run_id`. Escritas de manifest/checkpoint usam arquivo temporário e renomeação atômica. `evaluate` aceita somente `selected`, registra a abertura do teste e nunca altera pesos ou threshold congelados.

### Extensão heterogênea

O pacote `hetero/` será criado somente para o gate e seus testes. A extensão entra em tasks de treinamento apenas depois que causalidade dos atributos, supervisão, comparabilidade e projeção de recursos forem aprovadas por registro assinado pelo pesquisador. Nenhum componente heterogêneo é dependência do caminho P1.

### Rastreabilidade dos requisitos

| Requisitos | Elementos do plano e contrato |
|---|---|
| FR-001, FR-026–FR-027 | CLI dedicado, `ExperimentRun`, máquina de estados e contrato de artifacts |
| FR-002–FR-007 | política de dados, snapshots, normalização, TestLabelStore e guard de avaliação |
| FR-008–FR-015 | modelo e visões, positivos, KNN, perda e orçamento comparável |
| FR-016–FR-025 | budgets, downstream, registry de baselines, seleção, métricas, ablações e diagnósticos |
| FR-028–FR-029 | smoke, dry-run com shadow test e gate de aprovação explícita |
| FR-030–FR-032 | pacote e decisão do gate heterogêneo fora do caminho P1 |
| FR-033–FR-035 | namespace S003, política de extração explícita e relatório válido com resultado negativo |
| FR-036 | quatro comparações predefinidas, IC, t pareado, Cohen's `d_z` e Holm |

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

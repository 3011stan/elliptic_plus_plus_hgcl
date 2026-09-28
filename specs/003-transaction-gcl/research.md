# Phase 0 Research — S003-TxGCL

## R1 — Isolamento técnico do estudo

**Decision**: criar o CLI `hgcl-s003`, schema de configuração e orquestração dentro de `src/hgcl/studies/s003/`.

**Rationale**: o CLI e o config atuais descrevem explicitamente o S02 e contêm `native_wallet`, `fusion.alphas` e `hgcl`. Um dispatcher próprio elimina defaults ambíguos e permite congelar o comportamento histórico.

**Alternatives considered**: acrescentar condicionais `study_id` ao CLI atual foi rejeitado por aumentar acoplamento; criar outro repositório foi rejeitado porque dados, ambiente e utilitários neutros podem permanecer no mesmo projeto com namespaces e contratos explícitos.

## R2 — Fronteira de reutilização

**Decision**: implementar primeiro no namespace S003; extrair apenas primitivas realmente neutras, uma por task, após testes S02/S003 e registro em `decisions.md`.

**Rationale**: módulos atuais têm contratos úteis, mas vários assumem carteiras, regimes do S02 ou sua matriz. Copiar silenciosamente transferiria erros de especificação.

**Alternatives considered**: importar diretamente os módulos atuais foi rejeitado; duplicação permanente também foi rejeitada, pois dificulta manutenção. Extração incremental preserva rastreabilidade.

## R3 — Unidade temporal do grafo

**Decision**: materializar 49 snapshots independentes, um por `Time step`.

**Rationale**: auditoria local confirmou 203.769 transações e 234.355 arestas, todas com endpoints no mesmo time step. Logo, o particionamento preserva 100% das arestas `Tx→Tx` e garante que inferência em `t` não acesse `t+1`.

**Alternatives considered**: grafo cumulativo foi rejeitado por custo e por mudar o protocolo; remover arestas entre períodos é desnecessário porque elas não existem nos dados observados.

## R4 — Dimensionalidade e papéis das colunas

**Decision**: modelar `txId` como chave; `Time step` exclusivamente como metadado temporal; e usar como entrada 182 features financeiras, divididas em 93 locais, 72 agregadas e 17 aumentadas.

**Rationale**: o CSV possui 184 colunas totais: uma chave e 183 colunas não identificadoras. A distinção impede que ID ou tempo sejam tratados como atributos financeiros e torna os blocos verificáveis.

**Alternatives considered**: chamar todas as 183 colunas de features financeiras foi rejeitado; incluir `Time step` em `X_tx` foi rejeitado porque as fontes o usam para fatiamento, não como atributo contínuo do nó. A coluna permanece preservada e auditável.

## R5 — Normalização e KNN

**Decision**: ajustar o normalizador somente sobre 1–34; construir KNN de cosseno sobre os 182 atributos financeiros, separadamente em cada snapshot, com `K=10` e desempate por `txId`.

**Rationale**: cosseno reduz dependência de escala após normalização; o índice por snapshot não cria conexões temporais; o desempate torna a vizinhança reproduzível.

**Alternatives considered**: KNN global foi rejeitado por vazamento temporal; distância euclidiana foi rejeitada como principal por maior sensibilidade residual à escala; K variável fica reservado a análise posterior, pois multiplicaria pré-treinos.

## R6 — Comparabilidade do mascaramento

**Decision**: comparar blocos funcionais `[93,72,17]` com uma partição aleatória dos mesmos tamanhos e com amostras individuais de cardinalidade idêntica em cada aplicação.

**Rationale**: controla simultaneamente quantidade removida e granularidade de grupo. O contraste funcional versus grupo aleatório passa a medir semântica do agrupamento, não apenas block dropout.

**Alternatives considered**: Bernoulli independente com taxa aproximada foi rejeitado porque não iguala cardinalidade; grupos aleatórios de tamanhos diferentes confundiriam estrutura e intensidade.

## R7 — Seleção com custo controlado

**Decision**: predeclarar hiperparâmetros estruturais do SSL e congelar após o dry-run um número fixo de épocas comum às seeds, sem probe rotulado ou seleção retrospectiva de checkpoint; limitar a busca do classificador downstream a oito combinações baratas que reutilizam embeddings.

**Rationale**: Inspection-L fornece precedente para épocas fixas; a decisão preserva SSL independente de rótulos, evita viés para 1% e elimina o custo de probes recorrentes. F1 ilícito/MCC continuam controlando somente seleção, early stopping e threshold supervisionados.

**Alternatives considered**: busca completa do SSL foi rejeitada pelo custo; menor loss SSL em holdout não rotulado foi rejeitada porque não tem precedente no corpus e pode não se alinhar ao downstream; probe de 1% foi rejeitado por tornar a seleção do encoder dependente de rótulos e da menor fração.

## R7A — Direção da passagem de mensagens

**Decision**: usar somente a direção original `Tx→Tx` em toda a matriz principal e permitir uma variante P2 com arestas reversas em 1%, condicionada ao gate de recursos.

**Rationale**: a direção do fluxo é explícita no dataset, enquanto Inspection-L e GCPAL não documentam se suas implementações inverteram ou simetrizaram arestas. Um padrão dirigido é semanticamente defensável; a ablação mede o efeito da escolha sem contaminar o núcleo.

**Alternatives considered**: converter silenciosamente para não dirigido foi rejeitado; tornar a variante reversa obrigatória P1 foi rejeitado pelo custo e pela ausência de precedente explícito.

## R8 — Inferência estatística

**Decision**: teste t pareado bilateral, IC t de 95% da diferença, Cohen's `d_z` e correção de Holm para quatro comparações, separadamente por métrica primária.

**Rationale**: o pareamento aproveita as mesmas seeds e budgets; Holm controla FWER sem ser tão conservador quanto Bonferroni; `d_z` mede magnitude padronizada da diferença pareada. Com cinco pares, valores brutos e incerteza serão sempre expostos.

**Alternatives considered**: quatro testes sem correção foram rejeitados; Bonferroni simples perde potência; Wilcoxon com apenas cinco pares tem resolução muito baixa; bootstrap não remove a limitação de cinco seeds.

## R9 — XGBoost e compatibilidade

**Decision**: adicionar XGBoost 3.2.0 como dependência direta do S003, mantendo Python 3.11.

**Rationale**: essa versão declara Python `>=3.10` e fornece wheels para macOS ARM64 e Linux x86-64, cobrindo os dois ambientes do estudo. A release 3.4.1 exige Python `>=3.12` e é incompatível com o projeto. Fonte de compatibilidade: [PyPI — xgboost 3.2.0](https://pypi.org/project/xgboost/3.2.0/).

**Alternatives considered**: atualizar todo o projeto para Python 3.12 foi rejeitado por ampliar escopo e arriscar PyTorch/PyG; substituir por HistGradientBoosting foi rejeitado porque o baseline aprovado é XGBoost.

## R10 — Persistência e retomada

**Decision**: manifests JSON imutáveis, tabelas Parquet, tensores/checkpoints PyTorch, escrita atômica e máquina de estados com retomada exclusiva de `interrupted`.

**Rationale**: formatos simples e locais atendem ao pipeline offline; hashes impedem retomar sob dados ou configs diferentes; estados terminais evitam sobrescrever evidência científica.

**Alternatives considered**: banco relacional foi rejeitado por complexidade sem necessidade concorrente; reusar diretórios por nome humano foi rejeitado por colisão e baixa proveniência.

## R11 — Gate heterogêneo

**Decision**: implementar apenas auditoria e registro do gate em P1; encoder `Addr↔Tx` só será planejado para execução após aprovação explícita.

**Rationale**: atributos globais de endereço podem conter futuro, e a extensão não deve consumir o orçamento do núcleo antes de provar causalidade e viabilidade.

**Alternatives considered**: construir ambos os grafos desde o início foi rejeitado por risco de leakage, custo e desvio do caminho crítico.

# Decisões — Estudo 003

## D003-001 — Topologia principal

O experimento principal usa o grafo homogêneo `Tx→Tx`. O grafo heterogêneo
`Addr↔Tx` é uma extensão condicionada, não o núcleo obrigatório.

## D003-002 — Orçamento de rótulos

Avaliar 1%, 5%, 10% e 100% dos nós rotulados de treino, com amostragem
estratificada, subconjuntos aninhados e cinco sementes.

## D003-003 — Downstream

Congelar o encoder e treinar um MLP sobre `H‖X_tx`. No regime de 1%, comparar
`X-only`, `H-only` e `H‖X_tx`. Fine-tuning é somente análise secundária.

## D003-004 — Protocolo temporal

Usar 1–34 para pré-treino, validação interna e treino. Manter 35–49 intocados e
inferir cada snapshot separadamente.

## D003-005 — Mecanismo contrastivo

Usar duas visões aumentadas sem rótulos: perturbação estocástica e mascaramento
por blocos funcionais. KNN não produz uma terceira representação: apenas expande
os positivos. A perda multi-positivo considera o próprio nó, sucessores na direção
original `source_tx_id→target_tx_id` e vizinhos KNN como positivos.

## D003-006 — Nomenclatura

Usar `S003-TxGCL` como identificador interno. Reservar `H-GCL` para o Estudo 002;
o grafo principal do S003 não é heterogêneo.

## D003-007 — Direção da propagação

A matriz principal codifica a aresta original como `edge_index[0]=source` e
`edge_index[1]=target`, com propagação PyG `source_to_target`: o destino agrega
mensagens da origem. Uma variante com arestas reversas é ablação P2 em 1% e
depende do gate de recursos.

## D003-008 — Encoder e dimensão

O encoder principal e as adaptações GNN comparáveis usam duas camadas e dimensão
128, alinhadas a Inspection-L e GCPAL.

## D003-009 — Features do modelo

Os CSVs preservam 183 colunas não identificadoras, mas `X_tx` usa somente as 182
features financeiras. `Time step` permanece metadado de partição e auditoria.

## D003-010 — Seleção do SSL

Substituída por D003-016. A decisão anterior previa definir o número fixo de
épocas após um dry-run treinado.

## D003-011 — Equivalência dos baselines

Inspection-L preserva GIN 2×128, DGI e RF de 100 árvores; GCPAL preserva GIN
2×128, duas visões estocásticas, KNN `K=10`, perda multi-positivo e MLP 2 camadas
sobre `H‖X_tx`. Ambos usam os dados, splits, budgets e seeds comuns do S003, com
adaptações de dataset declaradas.

## D003-012 — Agregação de métricas

As métricas principais são calculadas sobre as predições conhecidas concatenadas
dos passos 35–49. O F1 é da classe ilícita pooled, não micro-F1. Resultados por
snapshot permanecem obrigatórios para auditoria temporal.

## D003-013 — Blindagem e single-unblinding

O smoke usa shadow test dentro de 1–34; o dry-run estrutural não executa
inferência nem materializa rótulos. Antes de abrir rótulos 35–49,
todas as 205 células P1 devem ter seleção, pesos e threshold congelados em um
manifesto de coorte. Uma única liberação global avalia a coorte sem permitir nova
seleção; cada acesso físico é auditado por coorte e run. Retomada ou rerun técnico
mantêm os mesmos hashes, pesos e threshold e ficam registrados.

## D003-014 — Perfis de engenharia

O smoke usa no máximo 256 nós por snapshot por seleção de hash estável,
`engineering_fit_steps=1..29`, `shadow_test_steps=30..34`, um snapshot completo
por batch, duas épocas SSL e três downstream. A definição anterior do dry-run
treinado foi substituída por D003-016. O smoke é evidência de engenharia, nunca
resultado científico.

## D003-015 — Aprovação do dry-run

O aceite é materializado por `DryRunApproval`, vinculado aos digests de dados,
configuração, código e pacote de evidências, à janela máxima de custo aceita, à
identidade declarada do pesquisador e ao instante UTC. Aprovação incompatível,
alterada ou reutilizada para outro design falha fechado.

## D003-016 — Dry-run estrutural e 100 épocas SSL

Decisão aceita pelo pesquisador em 2026-09-28. O smoke permanece como única
validação treinada pré-matriz. O dry-run é estritamente estrutural: não treina,
não executa inferência e registra `training_performed=false` e
`test_labels_materialized=false`. Ele valida as 205 células P1, identidades,
budgets, splits, registry, cache, checkpoints, retomada, isolamento do teste,
dependências, capacidade e projeção conservadora antes do aceite humano.

Toda execução aplicável da matriz usa exatamente 100 épocas SSL em cada seed,
predeclaradas antes do dry-run estrutural e nunca escolhidas por probe, rótulo,
convergência observada ou métrica downstream. A matriz científica e suas 205
células não foram reduzidas.

## D003-017 — Tratamento de valores ausentes por zeros estruturais (0.0)

Decisão aceita pelo pesquisador em 2026-09-28 após consulta ao notebook NotebookLM
autorizado (`66fb9e95-d225-4eaf-b45b-f8d232053677`). No dataset Elliptic++
(Elmougy & Liu, 2023), 965 transações (~0,47% do total) não foram desanonimizadas
na raspagem da blockchain e possuem valores vazios exclusivamente nas 17 features
aumentadas. Conforme a literatura especializada em redes financeiras UTXO e o design
do artigo, a ausência de registro representa inexistência da atividade (zero estrutural)
e deve ser imputada como constante `0.0` antes da normalização causal fit-only (1–34).
A imputação por média ou mediana é expressamente rejeitada por inflacionar
artificialmente volumes e graus e distorcer a geometria das representações.

## D003-018 — Execução estritamente temporal por snapshot no pré-treino contrastivo (FR-008, FR-012, FR-040)

Decisão aceita pelo pesquisador em 2026-10-02 após constatação de estouro de memória no cluster (tentativa de alocação contínua de 74,3 GB de RAM ao processar grafo achatado de 136.265 nós) e consulta formal ao notebook NotebookLM autorizado (`66fb9e95-d225-4eaf-b45b-f8d232053677`).

A literatura seminal presente no acervo (*Inspection-L - Loa et al., 2022; GCPAL - Lu & Wang, 2024; Elliptic++ - Elmougy & Liu, 2023; Weber et al., 2019; HeteroGCL - Chen et al., 2026*) estabelece que os datasets Elliptic e Elliptic++ são grafos temporais discretos compostos por subgrafos/snapshots independentes. Conectar ou fundir todos os snapshots em um único grafo plano monolítico cria atalhos temporais espúrios, viola a causalidade do fluxo UTXO e gera custo de memória $O(N_{\text{total}}^2)$ inviável.

Fica deliberado:
1. **Pré-treino contrastivo por snapshot**: Os encoders de métodos baseados em grafos contrastivos (`S003-TxGCL` e `GCPAL`) MUST executar o treinamento auto-supervisionado iterando snapshot por snapshot sobre os time steps de desenvolvimento (1–34), compartilhando os parâmetros do encoder GIN entre os passos temporais, conforme especificado em FR-008.
2. **Escopo local do KNN e negativos**: Conforme FR-012 e FR-040, o grafo de similaridade KNN ($k=10$) e os conjuntos de negativos na perda contrastiva multi-positivo MUST ser construídos exclusivamente no interior de cada snapshot temporal ($N_t \approx 1.000$ a $7.000$ nós), garantindo isolamento temporal sem conexões entre passos diferentes e pico de memória inferior a 200 MB.
3. **Representação downstream tabular**: Após o pré-treino SSL de cada semente, os embeddings congelados $H$ de todos os snapshots de treino (1–34) são extraídos e concatenados ordenadamente para compor o espaço de representação tabular ($H$, $H \parallel X$ ou $X$) sobre o qual o classificador downstream MLP é ajustado e avaliado estritamente nos subconjuntos de fit, validação e refit do orçamento de rótulos.
4. **Observabilidade da matriz**: Falhas em células da matriz não podem ser silenciadas; `execute_matrix_cell` deve logar tracebacks completos e `progress.json` deve incluir `failure_kind` e `failure_message`.

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

O encoder SSL usa número fixo de épocas definido após o dry-run e congelado antes
do teste. Não há probe rotulado nem seleção downstream de checkpoint do encoder.

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

Smoke e dry-run usam shadow test dentro de 1–34. Antes de abrir rótulos 35–49,
todas as 205 células P1 devem ter seleção, pesos e threshold congelados em um
manifesto de coorte. Uma única liberação global avalia a coorte sem permitir nova
seleção; cada acesso físico é auditado por coorte e run. Retomada ou rerun técnico
mantêm os mesmos hashes, pesos e threshold e ficam registrados.

## D003-014 — Perfis de engenharia

O smoke usa no máximo 256 nós por snapshot por seleção de hash estável,
`engineering_fit_steps=1..29`, `shadow_test_steps=30..34`, um snapshot completo
por batch, duas épocas SSL e três downstream. O dry-run de laboratório usa todos
os nós de 1–34, a mesma divisão shadow, um snapshot completo por batch, seed 11,
fração 1%, dez épocas SSL e vinte downstream com paciência cinco. Ambos são
evidência de engenharia, nunca resultado científico.

## D003-015 — Aprovação do dry-run

O aceite é materializado por `DryRunApproval`, vinculado aos digests de dados,
configuração, código e pacote de evidências, à janela máxima de custo aceita, à
identidade declarada do pesquisador e ao instante UTC. Aprovação incompatível,
alterada ou reutilizada para outro design falha fechado.

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

Usar três visões sem rótulos: perturbação estocástica, mascaramento por blocos
funcionais e KNN. A perda multi-positivo considera o próprio nó, vizinhos `Tx→Tx`
e vizinhos KNN como positivos, seguindo o GCPAL.

## D003-006 — Nomenclatura

Usar `S003-TxGCL` como identificador interno. Reservar `H-GCL` para o Estudo 002;
o grafo principal do S003 não é heterogêneo.

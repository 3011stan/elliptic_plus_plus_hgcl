# Configurações do Estudo 003

Este diretório é exclusivo do `S003-TxGCL`.

Não copiar `configs/lab.yaml`: ele codifica o protocolo do Estudo 002, incluindo
`native_wallet`, `fusion.alphas`, taxas fixas e uma divisão de validação própria.
Os schemas e perfis do S003 serão definidos pelo SDD.

Todo config deverá declarar ao menos:

- `study_id: s003`
- `task: transaction_classification`
- `graph_schema: tx_tx`
- `method_id: s003_txgcl`
- `target_node_type: transaction`

# Contract — `hgcl-s003` CLI

Todos os comandos imprimem JSON em stdout; erros estruturados em stderr. Códigos: `0=success`, `2=input/config`, `3=runtime/resource`, `4=gate/incomplete`, `130=interrupted`.

```text
hgcl-s003 doctor --config PATH
hgcl-s003 prepare --config PATH
hgcl-s003 audit --config PATH --prepared PATH
hgcl-s003 dry-run --config PATH [--prepared PATH] --run-id s003-...
hgcl-s003 approve-dry-run --run PATH --approval-file PATH
hgcl-s003 matrix --config PATH --prepared PATH --matrix-id s003-...
hgcl-s003 resume --run PATH
hgcl-s003 evaluate --run PATH
hgcl-s003 report --matrix PATH
hgcl-s003 hetero-gate --config PATH --prepared PATH
```

## Behavioral guarantees

- Todo identificador fornecido deve começar com `s003-`.
- `prepare` nunca grava no data root e nunca calcula métricas de 35–49.
- `matrix` falha com código 4 sem aprovação de dry-run cujo config/data/code seja compatível.
- `dry-run` usa apenas um shadow test contido em 1–34 e não abre rótulos de 35–49.
- `evaluate` exige estado `selected`, pesos e threshold congelados e abre os rótulos de 35–49 uma única vez por run final.
- `resume` aceita somente estado `interrupted` e verifica todos os digests.
- `report` inclui falhas e ausências; nunca preenche combinação inexistente.
- `hetero-gate` não inicia treino heterogêneo nem altera a matriz homogênea.
- Nenhum comando resolve `configs/lab.yaml` ou `artifacts/` do S02 como fallback.

Cada resposta de sucesso contém `study_id`, `command`, `status` e caminhos relativos dos artefatos produzidos.

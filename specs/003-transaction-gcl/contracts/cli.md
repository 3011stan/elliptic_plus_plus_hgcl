# Contract — `hgcl-s003` CLI

Todos os comandos imprimem JSON em stdout; erros estruturados em stderr. Códigos: `0=success`, `2=input/config`, `3=runtime/resource`, `4=gate/incomplete`, `130=interrupted`.

```text
hgcl-s003 doctor --config PATH
hgcl-s003 prepare --config PATH
hgcl-s003 audit --config PATH --prepared PATH
hgcl-s003 smoke --config PATH [--prepared PATH] --run-id s003-...
hgcl-s003 dry-run --config PATH --prepared PATH --run-id s003-...
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
- `approve-dry-run` valida que o pacote de evidências está completo, vincula os digests de dados/config/código/evidências/design e registra identidade, instante UTC, janela máxima aceita e autorização P2 separada; não aceita aprovação incompatível ou reutilizada.
- `matrix` não abre rótulos 35–49 e, após contabilizar as 205 células P1 como selecionadas ou falhas terminais explícitas, sela as runs selecionadas em `evaluation-cohort.json` antes de permitir `evaluate`.
- `smoke` treina no recorte determinístico e usa apenas um shadow test contido em 1–34, sem abrir rótulos de 35–49.
- `dry-run` não treina nem infere; valida o design exato das 205 células P1 e produz evidências com `training_performed=false` e `test_labels_materialized=false`.
- `evaluate` exige uma coorte completa em estado `sealed`, realiza uma única liberação global do teste e avalia somente runs, pesos e thresholds congelados; cada acesso físico é auditado por coorte e run.
- `resume` aceita somente estado `interrupted` e verifica todos os digests. Se a interrupção ocorreu após a abertura do teste, nenhuma seleção ou configuração pode ser alterada.
- Um rerun causado por corrupção técnica após abertura do teste recebe vínculo `technical_rerun_of`, mantém config, dados, pesos e threshold idênticos e não pode substituir silenciosamente o resultado original.
- `report` inclui falhas e ausências; nunca preenche combinação inexistente.
- `hetero-gate` não inicia treino heterogêneo nem altera a matriz homogênea.
- Nenhum comando resolve `configs/lab.yaml` ou `artifacts/` do S02 como fallback.

Cada resposta de sucesso contém `study_id`, `command`, `status` e caminhos relativos dos artefatos produzidos.

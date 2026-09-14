# T031 — apoio operacional local entregue

Data: 2026-09-14. D049. Matriz preservada: 100 avaliações, 25 grupos, 10 caches SSL.

## Alterações e verificação

- `scripts/lab/run.py audit` usa o recibo da preparação, valida o conteúdo completo
  e gera `artifacts/environment/lab-audit.json` com hashes, contagens por passo,
  dimensões, máximos, ausências, orçamentos sem IDs e recorrência agregada.
- `scripts/lab/run.py dry-run` exige auditoria/recibos compatíveis e gera
  `artifacts/environment/lab-dry-run.json` com o plano. Status planned retorna zero;
  nenhuma chamada a execute/fit/evaluate é feita por essa etapa.
- Aceite humano não é inferido de PASS nem de saída zero. O comando matrix continua
  iniciando treinamento, portanto só deve ser chamado depois do aceite explícito.
- Testes novos: resumo sem IDs, propagação de falha da validação, caminho dry-run
  sem execute e rejeição de auditoria com payload incompatível. São fixtures locais;
  não evidenciam preparação integral ou CUDA.
- `.venv/bin/python -m pytest -q`: 65 aprovados, 2 opcionais com dados originais
  ignorados. Nenhum teste original adicional, CUDA ou experimento integral executado.

## Próximos comandos do pesquisador

O agente não fez commit, push, pull remoto ou acesso ao laboratório. Depois de
versionar/subir esta entrega e puxá-la no laboratório, na raiz do clone:

```bash
git status --short --branch
git rev-parse HEAD
nix develop --no-update-lock-file
```

Exigir árvore limpa. Como src/scripts/tests/AGENTS.md mudaram, os recibos anteriores
pertencem a outra identidade. Antes de repetir as etapas, preservar os recibos antigos
num diretório de arquivo novo (este comando falha se o diretório já existir):

```bash
.venv-lab/bin/python - <<'PY'
from pathlib import Path
import shutil
folder = Path('artifacts/environment')
archive = folder / 'before-t031-d049'
archive.mkdir(parents=True, exist_ok=False)
for receipt in folder.glob('lab-*.json'):
    shutil.copy2(receipt, archive / receipt.name)
print(archive)
PY
.venv-lab/bin/python scripts/lab/run.py doctor
```

Conferir PASS, dados, CUDA e recursos. Depois usar um ID de smoke ainda inexistente;
`gpu-smoke-003` abaixo é sugestão, não deve substituir uma tentativa existente:

```bash
.venv-lab/bin/python scripts/lab/run.py smoke --id gpu-smoke-003
```

Se qualquer comando falhar, parar e enviar o diagnóstico antes de continuar.
Trazer `lab-doctor.json` e `lab-smoke.json` atualizados para conferir a nova identidade.
Preservar também o retorno de `git rev-parse HEAD` e os recursos disponíveis.

Após pré-requisitos conferidos, conforme protocolo:

```bash
.venv-lab/bin/python scripts/lab/run.py prepare
.venv-lab/bin/python scripts/lab/run.py audit
```

Executar cada comando somente após sucesso do anterior. Enviar `lab-prepare.json`
e `lab-audit.json`; aguardar revisão da auditoria antes do dry-run.

```bash
.venv-lab/bin/python scripts/lab/run.py dry-run
```

Enviar `lab-dry-run.json` e aguardar aceite antes de matrix. Roteiro completo e
critérios de conclusão: [protocolo T031](../../specs/001-hgcl-experiment/t031-execution-protocol.md).
T031 permanece pendente; relatório experimental só será criado com resultados reais.

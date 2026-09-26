# T031 — Início rápido da matriz no laboratório

**Data:** 2026-09-14  
**Estado:** Dry-run aceito (D051), matriz pronta para começar.  
**Tempo estimado:** Copiar e colar 5 minutos; treinamento desconhecido.

## Passo a passo

### 1. Entrar no clone

```bash
cd ~/elliptic_plus_plus_hgcl
```

### 2. Abrir o ambiente Nix

```bash
nix develop --no-update-lock-file
```

Sinal de sucesso: prompt muda, não há erros.

### 3. Verificação rápida (5 linhas)

```bash
echo "$HGCL_NIX_SYSTEM"
.venv-lab/bin/python -c 'import sys; print(sys.executable)'
nvidia-smi
ps -fu "$USER" | grep '[r]un.py matrix' || echo "Nenhum matrix em execução"
tmux ls
```

**Esperado:**
- Primeira linha: `x86_64-linux`
- Segunda linha: `/.../.venv-lab/bin/python`
- `nvidia-smi`: mostra RTX 2060, sem erros
- Terceira linha: "Nenhum matrix em execução" ou lista de processos
- Quarta linha: lista de sessões `tmux` (pode estar vazia)

Se tudo OK, prosseguir. Senão, parar e trazer a saída aqui.

### 4. Criar sessão persistente

```bash
tmux new -s hgcl
```

Você entra numa nova janela do tmux. Prompt pode parecer igual.

### 5. Iniciar o treinamento (dentro da janela tmux)

```bash
.venv-lab/bin/python scripts/lab/run.py matrix --id s02-001
```

O treinamento começa. Mensagens de progresso aparecem no terminal.

### 6. Desconectar (dentro da janela tmux)

Quando quiser sair **sem interromper**:

```
Ctrl+B
D
```

(Segurar Ctrl e B, soltar, depois apertar D. A sessão continua.)

Você volta ao prompt anterior.

### 7. Verificar progresso (depois de desconectar)

```bash
jq '[.[] | select(.state == "complete")] | length' artifacts/matrices/s02-001/groups.json
```

Mostra quantos grupos dos 25 já terminaram.

### 8. Reconectar (se quiser ver em tempo real)

```bash
nix develop --no-update-lock-file
tmux attach -t hgcl
```

De volta à janela com o treinamento em andamento (ou concluído, se não houver mais mensagens).

---

## Se algo der errado durante o treinamento

**Processo visível mas erro no terminal:**
1. Note a mensagem de erro e qual grupo estava rodando
2. Desconecte com `Ctrl+B, D`
3. Trazer a captura de tela ou log aqui

**Processo não aparece, mas grupos estão em `complete` em `groups.json`:**
- Treinamento terminou. Rodou:

```bash
nix develop --no-update-lock-file
.venv-lab/bin/python scripts/lab/run.py report --id s02-001
```

**SSH desconectou sem intenção:**
1. Reconecte
2. Rode (sem entrar no nix develop ainda):

```bash
ps -fu "$USER" | grep '[r]un.py matrix'
```

Se houver processo, reconecte ao tmux:

```bash
nix develop --no-update-lock-file
tmux attach -t hgcl
```

Se não houver processo, comece do passo 3 (verificação rápida).

---

## Fim esperado

Depois de 25 grupos completos, rodou:

```bash
nix develop --no-update-lock-file
.venv-lab/bin/python scripts/lab/run.py report --id s02-001
```

Confira que `artifacts/matrices/s02-001/report.json` exibe:

```json
{
  "status": "complete",
  "expected_evaluations": 100,
  "available_evaluations": 100,
  "missing_evaluations": []
}
```

Se tudo está em 100, trazer `artifacts/matrices/s02-001/report.json` aqui.

---

## Resumo

| Passo | Comando | Sinal de sucesso |
|-------|---------|------------------|
| 1 | `cd ~/elliptic_plus_plus_hgcl` | Sem erro, está no diretório |
| 2 | `nix develop --no-update-lock-file` | Prompt muda, sem erros |
| 3 | 5 linhas de verificação | Todos os valores conferem |
| 4 | `tmux new -s hgcl` | Entrada numa nova janela |
| 5 | `.venv-lab/bin/python scripts/lab/run.py matrix --id s02-001` | Mensagens de progresso começam |
| 6 | `Ctrl+B, D` | Desconecta e volta ao prompt anterior |
| 7 | `jq ...` | Número entre 0 e 25 |
| 8 | `tmux attach -t hgcl` | Volta à janela do treinamento |
| Fim | `scripts/lab/run.py report --id s02-001` | `status: complete`, 100/100 |

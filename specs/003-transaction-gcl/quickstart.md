# Quickstart Validation — S003-TxGCL

Este guia descreve validação esperada após a implementação. Não autoriza a matriz científica.

## Prerequisites

- branch `003-transaction-gcl`;
- Python 3.11 e ambiente instalado a partir do projeto;
- nove CSVs Elliptic++ disponíveis em modo somente leitura;
- espaço separado para `artifacts/s003/`;
- no laboratório, driver CUDA compatível com PyTorch 2.6.0 e limite configurado de 4,5 GiB.

## 1. Validate software contracts

```bash
.venv/bin/python -m pytest tests/s003/contract tests/s003/unit
```

Esperado: config rejeita chaves do S02, blocos cobrem 182 features, budgets são aninhados, positivos e negativos são disjuntos e estatística reproduz os casos de referência.

## 2. Inspect environment

```bash
.venv/bin/hgcl-s003 doctor --config configs/s003/smoke.yaml
```

Esperado: JSON `status=ready`, hashes dos nove CSVs válidos, Python/dependências registrados e nenhum path sobreposto.

## 3. Prepare original data

```bash
.venv/bin/hgcl-s003 prepare --config configs/s003/smoke.yaml
```

Use o path `prepared` retornado:

```bash
.venv/bin/hgcl-s003 audit --config configs/s003/smoke.yaml --prepared ARTIFACT_PATH
```

Esperado: 49 manifests de snapshot; 203.769 transações e 234.355 arestas antes do recorte; zero aresta entre time steps; 183 atributos não identificadores e 182 mascaráveis; transforms ajustados somente em 1–34.

## 4. Run the engineering smoke test

```bash
.venv/bin/hgcl-s003 dry-run \
  --config configs/s003/smoke.yaml \
  --prepared ARTIFACT_PATH \
  --run-id s003-smoke-local
```

Esperado: fluxo completo termina em até 10 minutos de treinamento no Mac, preparação reportada separadamente, métricas do shadow test em 1–34 marcadas `engineering_only=true`, zero abertura dos rótulos 35–49 e nenhum artefato científico liberado.

## 5. Exercise interruption and resume

Interrompa uma execução de teste após checkpoint e execute:

```bash
.venv/bin/hgcl-s003 resume --run artifacts/s003/runs/s003-smoke-local
```

Esperado: retomada produz os mesmos IDs e digests; alteração de config, dados ou revisão é rejeitada.

## 6. Run the one-seed dry-run

```bash
.venv/bin/hgcl-s003 dry-run \
  --config configs/s003/dry-run.yaml \
  --prepared ARTIFACT_PATH \
  --run-id s003-dry-run-001
```

Revise `run.json`, `selection.json`, uso de RAM/VRAM, duração projetada e relatório do shadow test. Confirme que o audit log registra zero abertura dos rótulos 35–49. A aprovação é uma ação separada e explícita do pesquisador. Sem ela, o comando `matrix` deve terminar com código 4.

## 7. Release gate

Antes de pedir autorização para a matriz, execute:

```bash
.venv/bin/python -m pytest tests/s003
git diff --check
```

O pacote de evidências deve demonstrar:

- nenhum acesso a 35–49 antes de `evaluate`;
- igualdade de budgets entre métodos;
- cardinalidade idêntica nos três controles de masking;
- cobertura e estados explícitos;
- custo medido e projeção da matriz;
- arquivos congelados do S02 sem modificação.

Somente depois do aceite do pesquisador será criado o arquivo de aprovação compatível e a matriz `lab` poderá ser iniciada.

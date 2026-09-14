# T031 — protocolo da primeira rodada experimental

Data: 2026-09-14. Formalização sob D048. Estado: roteiro preparado; nenhuma etapa
integral executada nesta entrega. Operador: pesquisador no laboratório NixOS/CUDA.
Este documento rege a execução da T031 sem modificar as escolhas científicas.

## Escopo e pontos de revisão

Matriz: 80 avaliações principais + 20 nativas; quatro métodos, seeds 11/23/37/53/71,
frações 1/5/10/100% no principal e apenas 100% no nativo. São 25 grupos,
10 caches SSL e 150 configurações candidatas RF/supervisionadas; estas últimas
não contam épocas, opções de limiar/alpha ou treinamentos SSL como candidatos extras.
Treino 1–28; validação 29–34; teste 35–49. Não retreinar em treino + validação.
[Extensões adiadas](deferred-extensions.md) não bloqueiam esta rodada.

1. Congelar a versão e conferir pré-requisitos antes da preparação integral.
2. Preparar e validar os dados; apresentar auditoria ao pesquisador antes do dry-run.
3. Executar dry-run com dados preparados; apresentar plano e aguardar aceite explícito
   do pesquisador antes de iniciar treinamento.
4. Executar/repor a matriz compatível e gerar relatório portátil; verificar completude.

Estes são pontos de revisão operacionais, não etapas automatizadas pelo wrapper.
O comando `scripts/lab/run.py matrix` começa treinamento após seu dry-run interno:
**não usá-lo para apenas inspecionar o plano**. Nenhum prazo transcorrido vale como aceite.

## 1. Congelamento e pré-requisitos

No clone do laboratório, raiz do projeto, com os documentos desta entrega versionados
pelo pesquisador antes da execução, registrar commit e árvore limpa:

```bash
git status --short --branch
git rev-parse HEAD
nix develop --no-update-lock-file
```

Conferir que `git status --porcelain --untracked-files=all` não retorna alterações.
Resolver pendências com o pesquisador; não excluir nem commitar arquivos automaticamente.
Conferir `configs/lab.yaml`, nove hashes no manifesto de origem, `flake.lock` e
`requirements/lab-cuda.lock`. Usar caminhos padrão do perfil; overrides precisam
ser explicitados e verificados em todos os comandos, nunca aplicados parcialmente.

Inspecionar `artifacts/environment/lab-doctor.json` e `lab-smoke.json`: PASS/complete,
CUDA/amostrador/backward, hashes e identidade de fonte compatíveis. A identidade
inclui src/configs/tests/requirements/scripts, AGENTS.md e flake; docs/specs não
entram nesse hash, por isso também registrar o commit documental.
Se a identidade mudou, repetir doctor e smoke com ID novo seguindo
[roteiro do laboratório](../../docs/laboratorio-nixos.md); preservar recibos anteriores
antes de repetir, pois o wrapper reutiliza seus nomes. Não repetir smoke por simples
mudança documental sem mudança de identidade relevante.

Registrar RAM, espaço livre e GPU no início (`free -h`, `df -h .`, `nvidia-smi`).
Orçamentos do perfil: RSS 24 GiB, GPU alocada 4,5 GiB; recorte desativado.
Espaço suficiente deve considerar derivados, checkpoints e cópia de preservação;
não há estimativa integral validada nesta entrega. Diante de insuficiência, parar.
Nenhum fallback CPU ou redução de dados/parâmetros é autorizado implicitamente.

## 2. Preparação, validação e auditoria

Somente após os pré-requisitos, executar no laboratório:

```bash
.venv-lab/bin/python scripts/lab/run.py prepare
```

Exigir código de saída zero e `artifacts/environment/lab-prepare.json` com PASS.
O recibo fornece `prepared`, `preparation_hash`, `payload_hash`, `attempt`,
`elapsed_seconds`, `peak_rss_gib` e `source_sha256`. Preservar tentativa e logs.
Copiar o valor exato de `prepared` para a variável abaixo (substituir o exemplo):

```bash
HGCL_PREPARED='/caminho/exato/retornado/no/recibo'
export HGCL_PREPARED
.venv-lab/bin/python -m hgcl.cli validate --config configs/lab.yaml --prepared "$HGCL_PREPARED" --json > artifacts/environment/t031-validate.json
```

Não continuar após saída não zero. Esperado: `status=PASS`, identidades iguais às do
recibo e `budgets_verified=20`. A validação completa lê alvos para verificar sua
semântica; não calcula métricas nem seleciona modelos. Isso é distinto de fit/dry-run,
que não materializam rótulos de teste para aprendizagem/seleção.

Apresentar auditoria compacta, sem listas integrais de endereços:

| Evidência | Verificação para aceite |
| --- | --- |
| `manifest.json` do preparado | Configuração, fontes, código, payload e arquivos verificados; recibo da tentativa preservado |
| `diagnostics.json`, campo snapshots | 49 passos por regime (98 representações), IDs/arestas válidos; dimensões 123/32 principal e 110/32 nativo; contar nós e arestas por passo e registrar máximos |
| `canonical/` e validação | Identidade endereço–passo deduplicada; cobertura dos extremos; nenhuma ligação entre passos; desconhecidos fora da supervisão |
| Preprocessadores e `missingness.json` | Allowlist e ausências preservadas; estatísticas ajustadas apenas nos passos de treino; sem substituir ausência por observação real |
| `targets/budgets.json` | 20 registros seed/fração; contagens disponíveis/selecionadas das classes 1 e 2 em treino e validação; ambas presentes, floor por classe, seleção aninhada e hashes |
| Máscaras e `targets/recurrence.parquet` | Mesmas máscaras entre métodos/regimes aplicáveis, orçamento separado de validação; recorrência e exposição ao rótulo registradas; visto significa observado no desenvolvimento 1–34 |
| Recibo e sistema | Tempo de preparação, pico RSS, tamanho dos derivados e espaço restante; não confundir com custo de treinamento |

`validate` reconstrói e compara máscaras/orçamentos/recorrência e confere dimensões,
IDs, índices e reversas. Ele não produz sozinho todas as tabelas desta auditoria;
o agente consolida os artefatos retornados pelo operador, sem nova decisão científica.
Não enviar CSVs originais ou grandes listas de IDs ao chat: apresentar resumos/JSONs.
Ponto de revisão 1: apresentar recibo, validação e auditoria; resolver qualquer
divergência antes de avançar. Não inflar orçamentos com classes insuficientes.

## 3. Dry-run explícito e aceite para treinar

Com a mesma variável `HGCL_PREPARED` e a auditoria aceita:

```bash
.venv-lab/bin/python -m hgcl.cli matrix --config configs/lab.yaml --prepared "$HGCL_PREPARED" --matrix-id s02-001 --dry-run --json > artifacts/environment/t031-dry-run.json
```

Conferir saída zero, `status=planned`, `data_integrity_checked=true`,
`training_performed=false`, `test_labels_materialized=false`,
`expected_evaluations=100`, `expected_ssl_caches=10`,
`expected_candidate_configurations=150`, 25 grupos e 100 IDs únicos.
`preparation` deve ter PASS e hashes iguais aos da validação anterior.
O dry-run usa validação estrutural: `budgets_verified=0` nesse resultado é esperado,
não substitui os 20 orçamentos já conferidos pela validação completa.

Registrar o aceite do pesquisador com data, commit, source/config/preparation/payload
e ID da matriz no registro operacional antes do treinamento. Ele deve aprovar o
plano concreto sobre esses dados. Não executar o próximo comando automaticamente.

## 4. Matriz serial e retomada

Após o aceite, usar sessão persistente no ambiente Nix:

```bash
tmux new -s hgcl
.venv-lab/bin/python scripts/lab/run.py matrix --id s02-001
```

Conferir que a nova sessão mantém o ambiente; quando necessário entrar nela com
`nix develop --no-update-lock-file`. Para desconectar: Ctrl+B, D; para retornar:
`tmux attach -t hgcl`. Não iniciar segunda execução concorrente.
Em primeira execução, confirmar que o ID não contém tentativa anterior não revisada.
Os recibos, `artifacts/matrices/s02-001/manifest.json`, `groups.json`, caches `ssl/`
e runs referenciados devem ser preservados. Grupos completos não são retreinados.

O mesmo comando/ID retoma uma matriz compatível **após inspeção da interrupção e
decisão do pesquisador**. Nunca apagar checkpoints, mudar seeds, hashes, dependências,
fanouts, frações, limiares ou dispositivos para conseguir concluir silenciosamente.
Falha: registrar comando, grupo, diagnóstico, status, recursos e impacto; parar e
consultar. Não atualizar código/locks durante a rodada. Teste não orienta ajustes.

## 5. Relatório e critérios de conclusão

```bash
.venv-lab/bin/python scripts/lab/run.py report --id s02-001
```

O relatório gera `artifacts/matrices/s02-001/report.json` e `report.md`.
Exigir status complete, `expected_evaluations=available_evaluations=100` e
`missing_evaluations=[]`; conferir também 25 estados complete em `groups.json`,
80 chaves principais, 20 nativas, quatro métodos e cinco seeds esperadas por condição.
Conferir caches e proveniências, scores/checkpoints/seleção congelada e suporte alinhado.
Métricas indefinidas em estratos sem suporte são reportadas como tais, nunca zero
inventado; distinguir completude das execuções de disponibilidade da métrica.

O agente produzirá **`docs/validation/experiment.md`** a partir dos artefatos reais,
após recebê-los do pesquisador. O wrapper não cria esse arquivo. Conteúdo obrigatório:

- Data, commit, configuração, locks, identidade dos dados/preparação e ambiente.
- Resultado dos pontos de revisão, contagem das avaliações e tentativas/falhas.
- F1 ilícito, precisão, recall, AP e MCC por regime/fração/método; média, desvio-padrão
  amostral, seeds definidas e suporte; visto/inédito e exposição prévia ao rótulo.
- Diferenças pareadas fusion–RF, fusion–H-GCL, fusion–controle e H-GCL–controle;
  preservar valores negativos, sem inferir significância apenas da média.
- Tempos observados separados por preparação, SSL, ajuste e avaliação quando
  disponíveis; declarar indisponível o que não foi medido, sem extrapolações.
- Limites: dataset único, classes globais sem data de crime, complemento nativo
  sem interpretação causal de vazamento, sem alegação de SOTA ou robustez adversária.
- Localização e hashes dos relatórios; inventário e destino da cópia de preservação
  dos artefatos e runs. Um resumo Markdown não substitui checkpoints/predições.

Não criar relatório com resultados fictícios nem marcar T031 concluída por dry-run.
Somente após todas as evidências acima, atualizar tasks/status; prosseguir com T032
e auditoria T033. Resultados parciais podem ser apresentados aos orientadores, desde
que identifiquem explicitamente condições/seeds faltantes e mantenham T031 aberta.

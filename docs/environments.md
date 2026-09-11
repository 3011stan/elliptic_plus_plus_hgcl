# Ambientes e armazenamento

## Decisões recebidas em 2026-09-05

- Dataset fornecido: `/Users/stan/Downloads/Elliptic++ Dataset`.
- Teste inicial: máquina pessoal do pesquisador.
- Experimento completo: mesma máquina de laboratório documentada no S00.
- Meta do teste pequeno: até 10 minutos de treinamento; preparação medida separadamente.

## Máquina pessoal — verificada localmente

- MacBook Air, Apple M4, CPU de 10 núcleos e GPU integrada de 8 núcleos.
- 16 GB de memória unificada.
- Aproximadamente 52 GiB livres antes da cópia do dataset.
- CPU é a proposta inicial de execução; aceleração deve passar por verificação de
  compatibilidade das operações escolhidas. O plano ainda definirá versões e dispositivo.
- O limite de 10 minutos é uma meta de aceite a dimensionar com amostragem e medições,
  não uma duração já demonstrada.

## Laboratório — hardware informado, sem inspeção remota nesta etapa

- Intel Core i9-9900KF, 3,60 GHz.
- NVIDIA GeForce RTX 2060, 6 GB de VRAM.
- 32 GB de RAM; HDD de 1 TB e SSD de 260 GB conforme registro anterior.
- Pendências: sistema operacional, driver NVIDIA, CUDA, espaço livre e acesso operacional.
- Versões de Python e bibliotecas do S00 não são automaticamente as versões do S02.
- Treinamento completo deve prever limites de memória, persistência de checkpoints e
  retomada, dimensionados no plano. Acesso ao laboratório ainda não foi exercido.

## Armazenamento

Raiz local dos dados, escolhida pelo pesquisador dentro do projeto:
`/Users/stan/Projects/masters-degree/hgcl-elliptic/elliptic-plus-plus`.

- `raw/`: nove CSVs, sem alterações; 2.206.089.537 bytes de conteúdo.
- Recortes e caches futuros: diretórios separados, sem modificar `raw/`.
- Código: `/Users/stan/Projects/masters-degree/hgcl-elliptic`.
- Artefatos futuros: `artifacts/`, ignorado pelo Git, ou caminho externo configurado.

Primeiro foi feita uma cópia verificada para um diretório compartilhado, pois o sistema
bloqueou a renomeação de Downloads e o acesso ao Finder não estava habilitado. Depois o
pesquisador colocou `elliptic-plus-plus` dentro do projeto. A localização atual foi
conferida novamente contra os nove hashes. As cópias anteriores não foram removidas.
[Histórico da cópia](data/relocation.json); [localização atual](data/current-location.json).

O diretório inteiro `elliptic-plus-plus/` é ignorado pelo Git. A configuração local usa
`ELLIPTIC_DATA_ROOT=./elliptic-plus-plus/raw`, relativo à raiz do projeto.

No laboratório, o caminho será configurado localmente por `ELLIPTIC_DATA_ROOT`, apontando
para os mesmos arquivos conferidos contra [o manifesto](data/source-manifest.json).
Nenhum caminho absoluto do Mac deve ser necessário dentro do algoritmo de treinamento.

## Milestone 1 environment checkpoint — 2026-09-11

Created project-local `.venv` from `/opt/homebrew/bin/python3.11` (3.11.15).
The installed system Python was not changed. Downloading a managed Python failed by
DNS, so the already installed compatible interpreter was used. PyPI dependency
resolution also fails by DNS; no validated lock, installed training stack or passed
PyG operation probe exists yet. Cache for these attempts: `/private/tmp/hgcl-uv-cache`.

### Atualização após instalação manual do pesquisador

O bloqueio acima é histórico: requirements/mac-cpu.lock foi gerado e os pacotes foram
instalados pelo pesquisador. Relatório em artifacts/environment/mac-cpu-probe.json.
Conferência do agente: uv pip check aprovou 42 pacotes; GIN bipartite em CPU executou
forward/backward com gradientes finitos. Núcleo: torch 2.6.0, PyG 2.6.1, sklearn 1.6.1,
Python 3.11.15. O doctor reutilizável continua pendente; laboratório não verificado.

D035: laboratory lock/CUDA/sampler work moved entirely to T030. T002 completed: reusable CPU doctor passed; report artifacts/environment/mac-cpu-doctor.json includes the Mac lock hash.

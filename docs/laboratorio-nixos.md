# Execução no laboratório NixOS — roteiro do pesquisador

A implementação está preparada para a **primeira verificação no laboratório**.
Não há ainda evidência de execução deste Flake, instalação Linux ou treinamento CUDA.
T030 continua aberta até a verificação real. Não é necessário ter Codex no laboratório.

## O que foi escolhido

- NixOS x86_64: ambiente FHS de usuário definido no Flake, sem Docker e sem sudo.
  Ele fornece o layout de bibliotecas esperado pelos wheels Linux; usa o driver NVIDIA
  já instalado em `/run/opengl-driver/lib`. Não instala driver nem altera o sistema.
- Python 3.11, torch 2.6.0+cu124, PyG 2.6.1 e pyg-lib 0.4.0+pt26cu124.
  Os dois wheels CUDA vêm dos índices oficiais PyTorch/PyG. O restante é resolvido
  pelo uv e fixado com hashes em requirements/lab-cuda.lock.
- Mac ARM: o mesmo Flake oferece Python 3.11/CPU. A `.venv` já validada continua
  disponível; o caminho opcional Nix usa `.venv-nix`, sem substituir a `.venv`.
- A RTX 2060 tem 6 GiB. Permanecem os limites de 4,5 GiB de alocação GPU e 24 GiB
  de RSS do protocolo, mesmo com os 62 GiB de RAM informados agora.
- O projeto não baixa dados nem começa treinamento ao entrar no Flake.

Base: PDF PPComp fornecido pelo pesquisador (páginas impressas 3–6), que orienta
Flakes de usuário e reserva configuration.nix/driver ao administrador. Exemplo de
Python 3.12 no PDF não altera o Python 3.11 deste experimento.
Referências: https://nixos.org/manual/nixpkgs/stable/#sec-fhs-environments,
https://docs.pytorch.org/get-started/previous-versions/ e
https://data.pyg.org/whl/torch-2.6.0+cu124.html.

## 1. Atualizar o clone e gerar o lock Nix

Depois que as alterações locais forem enviadas ao GitHub, entre na pasta do clone
**no terminal remoto**. Execute um comando de cada vez. Se qualquer um falhar,
pare e traga a saída; não instale pacotes globais nem use sudo.

```bash
git status --short
git pull --ff-only
nix flake lock
git add flake.lock
nix develop --no-update-lock-file
```

Se git status mostrar alterações anteriores que você não reconhece, consulte antes
do pull. O flake.lock será criado no laboratório, como combinado. Adicioná-lo ao
índice permite que Nix use o arquivo no checkout Git; isso não faz commit nem push.
Não execute `nix flake update` durante o experimento.

## 2. Gerar o lock Python/CUDA e instalar

Dentro do ambiente aberto pelo último comando:

```bash
bash scripts/lab/environment.sh lock
bash scripts/lab/environment.sh install
```

A primeira etapa gera requirements/lab-cuda.lock com dependências transitivas e
hashes. Recusa substituir um lock existente. Em uma instalação futura com os dois
locks já recebidos pelo Git, pule a geração e execute apenas install.

A instalação pode baixar vários GB, mas não compila torch/PyG a partir do código-fonte.
As versões científicas não são atualizadas automaticamente. O ambiente fica em
`.venv-lab`, ignorado pelo Git. As bibliotecas CUDA desse ambiente são 12.4: o 13.2
mostrado por nvidia-smi representa o suporte do driver, não o runtime do PyTorch.

## 3. Baixar e conferir os originais

```bash
.venv-lab/bin/python scripts/lab/run.py download
```

O comando lista a pasta pública dos autores, seleciona apenas os nove nomes do
manifesto e baixa para artifacts/downloads. Cada arquivo só é publicado em
elliptic-plus-plus/raw depois de conferir tamanho e SHA-256; originais existentes
não são substituídos. Se já estiverem presentes e corretos, nenhuma consulta ao Drive
é necessária. IDs remotos não são fixados sem inspecionar a listagem atual.

O download público depende das permissões/cotas do Google Drive. Ausência, duplicação
ou hash diferente é motivo de parada, sem regenerar o manifesto nem aceitar outra versão.
Os dados continuam fora do Git. Resultado: artifacts/environment/lab-download.json.

## 4. Executar o diagnóstico e parar para revisão

```bash
.venv-lab/bin/python scripts/lab/run.py doctor
```

O diagnóstico confere Linux/Python/wheels, hashes, memória e espaço; executa o
NeighborLoader real, mantém relações reversas pareadas e verifica forward/backward
com atualização de pesos CUDA para âncoras dos dois tipos. Não usa rótulos nem inicia
a matriz. Não faz fallback para CPU. Os 10 GiB mínimos de disco são um piso inicial,
não uma previsão do armazenamento total do experimento.

**Traga o conteúdo de artifacts/environment/lab-doctor.json.** Se a instalação falhar
antes disso, traga o erro do terminal. Ainda não inicie a matriz.

Guarde os locks gerados para devolver ao GitHub após revisão:

```bash
git add flake.lock requirements/lab-cuda.lock
```

Podem ser enviados pelo seu fluxo Git habitual ou copiados de volta pelo VS Code.
Não compartilhe credenciais. Não é necessário copiar a `.venv-lab` ou os dados.

## 5. Após revisão: smoke com dados originais na GPU

```bash
.venv-lab/bin/python scripts/lab/run.py smoke --id gpu-smoke-001
```

Usa configs/smoke-gpu.yaml: mesmos passos 1,2/29/35 e 256 transações por passo do
smoke CPU, quatro métodos, seed 11, 100% dos rótulos do recorte, até 600 segundos
para treino/seleção. Usa CUDA e o amostrador do laboratório. É teste de engenharia,
não resultado científico. Preparação e avaliação final têm medição separada.
Use um ID novo se houver uma execução anterior. Traga lab-smoke.json para revisão.

## 6. Somente após aceite da T030: preparação integral e matriz

```bash
.venv-lab/bin/python scripts/lab/run.py prepare
```

Isso prepara os 49 passos e o complemento nativo. Se os dados não atenderem aos
contratos, pare e traga o diagnóstico. Preparação integral ainda não foi executada.
Depois, para T031, dentro de uma sessão persistente:

```bash
tmux new -s hgcl
.venv-lab/bin/python scripts/lab/run.py matrix --id s02-001
```

Use Ctrl+B, depois D para desconectar da sessão sem interromper o processo. Ao
voltar, entre novamente no ambiente Nix e use `tmux attach -t hgcl`. Não inicie uma
segunda matriz enquanto a primeira estiver rodando. Se o processo tiver parado,
primeiro inspecione a falha; o mesmo comando/ID faz a retomada compatível e preserva
grupos concluídos. Não atualize código ou locks com treinamento em andamento.

O tmux precisa ser iniciado dentro do ambiente Nix; se já houver um servidor tmux
antigo com outro ambiente, abra `nix develop --no-update-lock-file` dentro da nova
janela antes de executar Python.

```bash
.venv-lab/bin/python scripts/lab/run.py report --id s02-001
```

Relatórios ficam em artifacts/matrices/s02-001/report.json e report.md. Artefatos de
execução são ignorados pelo Git: mantenha cópia dos checkpoints/resultados. Nenhum
script faz push ou upload. Cada etapa grava recibo em artifacts/environment/lab-*.json;
um erro deixa status FAIL e não aciona automaticamente a etapa seguinte.

## Uso opcional no Mac com Nix já instalado

Só depois de receber flake.lock gerado no laboratório:

```bash
nix develop --no-update-lock-file
bash scripts/lab/environment.sh install
.venv-nix/bin/python -m hgcl.cli doctor --config configs/smoke.yaml
```

O caminho Mac continua usando requirements/mac-cpu.lock e o procedimento anterior
de build isolado do pacote local. O caminho Nix/macOS ainda requer verificação própria.
Não é necessário instalar Nix no Mac para continuar usando `.venv/bin/python`.

# T030 — implementação local entregue; aceite remoto pendente

Autorização D042; geração dos locks no laboratório aceita em D043. Checklist de
requisitos: 16 aprovados, 0 pendentes. Sem hooks de extensão. Nenhum contêiner foi
iniciado nem Nix instalado no Mac. O ponto de parada local anterior foi resolvido
pela escolha de executar Nix e gerar os locks no laboratório.

## Verificação local

- 12 testes específicos: download por manifesto, hashes incorretos, ausência na
  listagem, preservação de originais, perfil GPU e forward/backward CPU do diagnóstico.
- Regressão geral: 61 aprovados, 2 opcionais ignorados, 5,20 s.
- Smoke original separado: 1 aprovado, 9,75 s; run smoke-003. Treino + seleção:
  2,162 s; avaliação/reload: 0,101 s. É execução CPU, não CUDA.
- Sintaxe Bash e git diff --check aprovados.
- Evidência estruturada: lab-local-checks.json.

## O que ainda precisa ser executado no NixOS

1. Gerar flake.lock e avaliar o Flake de usuário.
2. Gerar requirements/lab-cuda.lock e instalar os wheels com hashes.
3. Testar o download público do Drive e verificar os nove CSVs.
4. Executar CUDA/pyg-lib/NeighborLoader/backward e conferir memória/espaço.
5. Após revisão, executar smoke-gpu com dados originais.

Não foram produzidos locks fictícios. Nem a avaliação do Flake em Linux/macOS nem
os testes CUDA são substituídos pelos testes CPU. Os testes de download usam cliente
simulado; os hashes do smoke usam os originais reais. Compatibilidade do ambiente
FHS de usuário com esta instalação NixOS e acesso atual ao Drive permanecem verificações
remotas explícitas. Não há instalação global, alteração de driver ou fallback CPU.

Roteiro operacional: [laboratorio-nixos.md](../laboratorio-nixos.md).
T030 permanece incompleta; T031 não foi iniciada. O pesquisador fará o push e
executará a primeira verificação remota, conforme combinado.


## Primeira tentativa remota e correção D044

O pesquisador executou nix flake lock no laboratório. A API GitHub retornou
HTTP 403 / API rate limit exceeded ao resolver nixos-26.05. Essa tentativa
não fornece evidência de avaliação do ambiente ou de funcionamento CUDA.
Sob autorização D044, flake.nix passou a usar o tarball oficial do canal NixOS
26.05. Alteração local conferida por diff; nova tentativa remota pendente.
As evidências CPU acima antecedem esta troca de origem e não validam o novo
conteúdo Nixpkgs ainda não resolvido. Não houve mudança no código Python.

## Diagnóstico remoto aprovado e primeiro smoke GPU

O doctor remoto passou no NixOS: Python 3.11.16, torch 2.6.0+cu124, PyG 2.6.1,
pyg-lib 0.4.0+pt26cu124 e RTX 2060. Conferiu nove originais, executou NeighborLoader
e forward/backward CUDA para âncoras address/transaction. Pico alocado no probe:
22.899.200 bytes; memória GPU livre informada: 5.746.065.408 bytes.

`gpu-smoke-001` chegou ao fim do treinamento dos quatro métodos e falhou antes da
junção com rótulos de teste, na repetição da inferência do checkpoint. O código
comparava duas cargas do mesmo checkpoint com rtol 1e-7/atol 1e-8. A inferência
CUDA usa `index_add_`, cuja ordem de acumulação pode produzir variações float32.
A mensagem antiga não registrou magnitude nem método, portanto a causa numérica
é a explicação técnica sustentada pelo caminho do código, ainda a ser confirmada
pelos novos diagnósticos no NixOS.

Correção autorizada: tolerância alinhada ao contrato de inferência existente
(rtol 1e-5, atol 2e-6), zero divergências de decisão no limiar como condição
adicional e relatório por método com diferenças máxima/média. Uma diferença fora
da tolerância ou qualquer decisão divergente continua falhando. Verificação local:
63 testes aprovados, 2 opcionais ignorados; smoke original `smoke-004` aprovado.
Nova execução remota: `gpu-smoke-002`.

## Aceite final da T030

`gpu-smoke-002` terminou com status `complete`, escopo de engenharia, quatro métodos
e 314 observações de teste por método. Treino + seleção: 5,820331847 s; avaliação:
0,187829161 s; recarga aprovada. Foram comparadas 1.300 pontuações por método, com
zero mudanças de decisão. Maior diferença absoluta: 5,960464478e-8 (H-GCL), abaixo
de atol 2e-6/rtol 1e-5. RF foi idêntica.

Os locks gerados no NixOS foram versionados no commit `96455c1` e sincronizados:
`flake.lock` SHA-256 `0c3d6450f6083ef28ac229bdaa3a2385f1fa886444534e7918933c76486233a0`;
`requirements/lab-cuda.lock` SHA-256 `13b9004ff793f65ccad445b4378712118c3b84c2409070cd1388275d0b406fe0`.
Esses hashes coincidem com o doctor remoto. T030 concluída; a matriz científica
continua não executada. Próximo estágio: preparação integral T031.

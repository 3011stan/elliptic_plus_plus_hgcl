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

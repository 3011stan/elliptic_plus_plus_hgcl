# Instruções permanentes para agentes — H-GCL Elliptic++

## Interrupção obrigatória diante de impedimentos ou desvio de escopo

Instrução explícita do pesquisador, registrada em 2026-09-11, válida para toda a
implementação do projeto:

> Se tiver algum impedimento ou algo que saia do escopo, interrompa e me consulte.

Ao encontrar um impedimento ou uma ação fora do escopo autorizado, interrompa a
execução e consulte o pesquisador antes de continuar. Informe a tarefa afetada,
a evidência do problema, seu impacto e a decisão necessária. Não contorne o
impedimento nem altere escopo, protocolo, dependências ou critérios de aceite
silenciosamente. Não marque tarefas incompletas como concluídas.

Esta regra prevalece sobre orientações locais de continuidade autônoma. Uma autorização
anterior para executar tarefas não é autorização para contornar novos impedimentos.
Registre o ponto de parada para permitir retomada após a resposta do pesquisador.

## Escopo e rastreabilidade

- Escopo autorizado nesta etapa (D049): implementar e verificar localmente o apoio
  operacional à T031, preservando a matriz de 100 avaliações. O pesquisador faz
  commit/push, atualiza o clone e executa os comandos no laboratório. Validação
  CUDA/NixOS e preparação integral dependem das evidências do operador; treinamento
  da matriz somente após revisão e aceite explícito do dry-run com dados preparados.
- Preserve as decisões científicas aceitas e os CSVs originais.
- Estado operacional em docs/status.md; decisões em docs/decisions.md.
- Verifique tarefas antes de marcar conclusão. O contexto atual do usuário determina
  autorizações posteriores; esta lista não impede uma ampliação explícita futura.

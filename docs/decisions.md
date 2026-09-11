# Decisões e pendências — S02

| ID | Estado | Registro | Origem |
| --- | --- | --- | --- |
| D001 | Adotada operacionalmente | Código em repositório independente sob `Projects/masters-degree/hgcl-elliptic`; pesquisa permanece no StanOS. | Organização proposta e continuidade do pedido. |
| D002 | Confirmada | Usar os CSVs originais fornecidos em Downloads. | Pesquisador, 2026-09-05. |
| D003 | Executada com limitação | Cópia verificada em `datasets/elliptic-plus-plus/raw`; origem retida porque o movimento foi bloqueado pelo sistema. | Comprovante em `data/relocation.json`. |
| D004 | Confirmada | Smoke no Mac; experimento completo na máquina do laboratório do S00. | Pesquisador, 2026-09-05. |
| D005 | Confirmada | Meta de até 10 minutos de treinamento no smoke, excluindo preparação inicial, que será medida à parte. | Resposta do pesquisador, 2026-09-05. |
| D006 | Proposta | CPU como dispositivo inicial do smoke; avaliar aceleração quando as operações forem definidas. | Compatibilidade de operações ainda não verificada. |
| D007 | Redação aceita | Preservar a pergunta principal revisada sobre fusão versus métodos de referência sob diferentes disponibilidades de rótulos; comparadores principais definidos posteriormente em D024. | Pesquisador aprovou a revisão e pediu acréscimo sem mudança, 2026-09-07. |
| D008 | Resolvida no escopo especificado | Unidade e alvo em D015; deduplicação no contrato de entradas; recorrência em D028. Incertezas históricas dos atributos permanecem limitações explícitas da política D018; investigação em D020. | Discussão com o pesquisador, consolidada em 2026-09-07. |
| D009 | Resolvida pelas decisões D023 e D026 | Partições, orçamento de validação, métricas, limiar e número de seeds aceitos. Detalhes operacionais seguem no plano técnico. | Aprovações do pesquisador, 2026-09-07. |
| D010 | Em revisão | Constituição inicial 0.1.0, com ratificação pendente. | Rascunho derivado do pedido e princípios propostos. |
| D011 | Confirmada e verificada | Dataset dentro do projeto em `elliptic-plus-plus/raw`; essa localização substitui a raiz operacional compartilhada da D003. Diretório ignorado pelo Git. | Pesquisador informou a movimentação; nove hashes conferidos novamente. |
| D012 | Redação em D007; matriz em D024 | Redação ampliada aceita; manter ramos isolados para analisar contribuição e selecionar referências externas tabulares e de grafos. Matriz principal posteriormente aceita em D024. | Consulta a `elmougy2023demystifying` e discussão, 2026-09-07. |
| D013 | Confirmada | Inspection-L é a inspiração informada pelo pesquisador para a etapa autossupervisionada. Fonte registrada como `lo2022inspectionl`. | Pesquisador, 2026-09-07. |
| D014 | Pergunta e controle aceitos | RQ2 sobre benefício do pré-treinamento sob poucos rótulos aceita. Controle com o mesmo encoder aceito em D024; fine-tuning fixado em D027. Ineditismo não estabelecido; `lu2024gcpal` é um precedente adicional. | Pesquisador aceitou o acréscimo após a proposta; persistido em 2026-09-07. |
| D015 | Confirmada | Predizer a classe global do endereço ativo por `(address, Time step)`. Não interpretar como trajetória temporal de licitude, data de crime ou identidade de pessoa. Linhas brutas repetidas exigem consolidação explícita. | Concordância sobre o alvo e discussão posterior, 2026-09-07. |
| D016 | Confirmada | Grafos endereço–transação independentes por passo no cenário principal, predição ao encerramento da fatia e encoder com pesos compartilhados; sem grafo cumulativo ou 49 modelos separados. | Concordância explícita às regras 1 e 2, 2026-09-07. |
| D017 | Confirmada | Teste excluído de todo ajuste, inclusive SSL sem rótulos. Classes separadas das entradas e vistas; usadas apenas nos estágios autorizados de supervisão e avaliação. | Regra 3 aceita e regra 4 esclarecida na discussão, 2026-09-07. |
| D018 | Confirmada — opção A, esclarecida por D021 | Cenário principal com atributos por passo nos dois ramos, incluindo propriedades financeiras locais das transações; cenário complementar limitado com atributos nativos globais e limitações temporais explícitas. Catálogo em input-contract.md; complemento em D029. | Opção A aceita e conteúdo financeiro exigido pelo pesquisador, 2026-09-07. |
| D019 | Confirmada | Orçamento de escassez contado por endereços únicos rotulados do treino; mesmos subconjuntos entre métodos; orçamento de validação separado. Percentuais fixados em D025; arredondamento será especificado tecnicamente. | Concordância explícita à regra 5, 2026-09-07. |
| D020 | Investigação concluída; política escolhida em D018 | INV-001 concluída no escopo limitado. Evidências, exemplos, exceção de contagem, scripts e verificação independente persistidos. Último bloco permanece indeterminado; contrato posterior registrado em D030. | Execução autorizada e opção A posteriormente aceita, 2026-09-07; relatório em `data/inv-001/README.md`. |
| D021 | Confirmada | BTC, taxas e tamanho de transação são fundamentais no cenário principal. Usar propriedades locais das transações do passo e resumos por endereço; checagens de colunas e semântica integram o trabalho técnico. A opção A não exige atributos exclusivamente topológicos. | Orientação explícita do pesquisador, 2026-09-07. |
| D022 | Confirmada — condução do trabalho | Reutilizar a literatura; interações concisas focadas em decisões científicas abertas. O agente resolve detalhes técnicos e apresenta apenas desvios materiais de escopo/interpretação para discussão. | Pesquisador solicitou mais objetividade e velocidade no SDD, 2026-09-07. |
| D023 | Confirmada | Treino nos passos 1–28, validação 29–34 e teste 35–49, inclusive, para os métodos comparados. A reserva interna de validação é uma adaptação do S02, sem precedente exato confirmado; avaliação detalhada posteriormente em D026. | Pesquisador aceitou a divisão proposta, 2026-09-07. |

| D024 | Confirmada | Matriz principal: RF tabular; encoder supervisionado sem SSL; mesmo encoder com pré-treinamento H-GCL; fusão tardia RF + H-GCL. Fine-tuning definido em D027 e complemento em D029. | Pesquisador aprovou a proposta, 2026-09-07. |
| D025 | Confirmada | Avaliar cada método com 1%, 5%, 10% e 100% dos endereços únicos rotulados do treino, com os mesmos subconjuntos entre métodos: 16 configurações principais por seed. Orçamento da validação separado, definido posteriormente em D026. | Pesquisador aprovou a proposta, 2026-09-07. |

| D026 | Confirmada | F1 ilícito por endereço–tempo como métrica principal; precisão, recall, AP e MCC complementares. Cinco seeds, média/desvio-padrão e comparações pareadas (80 avaliações). Limiar maximiza F1 na validação e é congelado antes do teste. Validação usa o mesmo percentual do treino sobre seus próprios endereços únicos rotulados, com subconjuntos iguais entre métodos; conferir contagens antes de treinar. | Pesquisador aprovou o conjunto de definições, 2026-09-07. |

| D027 | Confirmada | Após SSL, ajustar encoder e cabeça com rótulos permitidos; controle usa a mesma arquitetura/cabeça treinada do zero e o mesmo regime supervisionado. | Resposta explícita do pesquisador, 2026-09-07. |
| D028 | Confirmada | Manter endereços recorrentes no teste e reportar resultados de já vistos e inéditos separadamente, além do resultado geral. | Resposta explícita do pesquisador, 2026-09-07. |
| D029 | Confirmada | Complemento nativo: quatro métodos, 100% dos rótulos, cinco seeds; 20 avaliações adicionais, total de 100. | Resposta explícita do pesquisador, 2026-09-07. |
| D030 | Definida tecnicamente sob D022 | Contrato de entradas e desenho inicial de treinamento/fusão persistidos; campos financeiros explícitos, deduplicação por chave, ausências preservadas, fine-tuning pareado e média ponderada na validação. Complemento troca descritores de endereço pelos 55 atributos nativos nos dois ramos, mantendo transações explícitas. | Consolidação técnica do agente; documentos em `specs/001-hgcl-experiment/`. |

| D031 | Planejamento técnico registrado | Plano, modelo de dados, contratos CLI/configuração, quickstart e 33 tasks ordenadas; Python 3.11 isolado, CPU smoke e CUDA após probe; cenários científicos preservados. | Continuidade autorizada pelo pesquisador, 2026-09-07. |

## Consolidação de 2026-09-11

- D032 — Esclarecimento de D015/D028: o modelo pode inferir sobre endereços inéditos,
  sem receber seus rótulos. IDs são chaves, nunca atributos ou embeddings de identidade.
  O rótulo global supervisiona e avalia a associação à ilicitude; não define a data do evento.
  A avaliação de inéditos permanece a já aceita. Não houve mudança de tarefa ou matriz.
- D033 — Pesquisador autorizou consolidar o estado documental e executar o marco 1,
  T001–T012. Treinamento e execução da matriz pertencem aos marcos seguintes.

Plano, entradas, arquitetura, aumentações, fusão e recursos iniciais estão especificados.
Compatibilidade, tempo e memória ainda exigem verificações práticas. A constituição
continua em revisão. Estado atual: [status.md](status.md); execução: [tasks.md](../specs/001-hgcl-experiment/tasks.md).

- D034 — Instrução permanente explícita do pesquisador: durante toda a implementação,
  diante de qualquer impedimento ou algo fora do escopo, interromper e consultá-lo antes
  de continuar. Registrada em AGENTS.md na raiz do repositório, em 2026-09-11.

- D035 — Autorizado pelo pesquisador: concentrar integralmente a preparação, lock e
  validação do ambiente do laboratório na T030. T002 cobre Mac e doctor CPU reutilizável;
  T003–T012 podem prosseguir após seu aceite local. Matriz científica inalterada.

- D036 — Pesquisador autorizou iniciar T013–T025, implementação dos modelos e smoke
  no Mac. Regra D034 de interrupção/consulta preservada; matriz científica não iniciada.

- D037 — Pesquisador autorizou corrigir a divisão dos lotes na T016 e retomar as
  tarefas; reafirmou que a consulta obrigatória deve continuar diante de impedimentos.

- D038 — Pesquisador autorizou corrigir a expectativa do teste de limiar na T022
  e seguir. Regra de desempate científico preservada; consulta obrigatória continua.

- D039 — Pesquisador autorizou concluir T026–T029 antes da transferência ao laboratório.
  Escopo: controles de acesso, matriz, retomada e relatórios; treinamento científico
  integral e preparação do laboratório continuam fora desta execução. D034 vigente.

- D040 — Pesquisador autorizou separar validação estrutural/hash da validação de
  rótulos para corrigir o acesso inicial de fit e retomar T026–T029.

- D041 — Pesquisador autorizou corrigir o isolamento da fixture de retomada, remover
  somente seus artefatos simulados identificados e retomar as verificações T026–T029.

- D042 — Pesquisador autorizou a preparação local da T030: flake.nix/flake.lock,
  dependências Linux/CUDA, diagnóstico real do amostrador, download do Drive com
  hashes e roteiro em português. O pesquisador enviará as alterações ao GitHub e
  executará a validação no laboratório quando avisado. Não há autorização para
  modificar a configuração global do laboratório ou iniciar a matriz nesta etapa.
  PDF do PPComp informa uso de Flakes por projeto, sem sudo. Máquina palmito:
  NixOS 26.05.5591.fd1462031fde, Nix 2.34.8, x86_64, RTX 2060 6 GiB, driver
  595.71.05, aproximadamente 62 GiB de RAM e 123 GB livres no volume consultado.

- D043 — Pesquisador escolheu gerar flake.lock no NixOS, executando comandos
  preparados aqui; Docker e instalação de Nix no Mac descartados. A entrega local
  inclui também o comando de resolução Linux para requirements/lab-cuda.lock,
  que será gerado e verificado no ambiente de destino antes de congelar a execução.
  Smoke GPU é um teste de engenharia pequeno com a política do smoke CPU, usando
  CUDA/NeighborLoader; não amplia nem altera a matriz científica.

- D044 — Pesquisador autorizou substituir a origem github:NixOS/nixpkgs/nixos-26.05
  pelo arquivo oficial https://channels.nixos.org/nixos-26.05/nixexprs.tar.xz,
  após HTTP 403 (limite da API GitHub no IP do laboratório) na geração do lock.
  Série 26.05 preservada; revisão exata ainda depende do lock gerado no NixOS.
  Roteiro atualizado. Nenhum token, commit ou push automático foi utilizado.

# Feature Specification: Experimento H-GCL e fusão tardia no Elliptic++

**Feature Branch**: `main` (especificação ativa em `specs/001-hgcl-experiment`)

**Created**: 2026-09-05

**Status**: Especificação e planejamento consolidados; implementação pendente.

**Updated**: 2026-09-07 — entradas, avaliação, fine-tuning, recorrência e complemento consolidados.

**Input**: Especificar em conjunto um experimento no Elliptic++, usando SDD e Spec Kit,
com teste pequeno nos CSVs originais ao final da implementação. O pesquisador confirmou
smoke na máquina pessoal, até 10 minutos de treinamento, e experimento completo no laboratório.

**Contrato metodológico acordado**

- Predizer a classe global disponibilizada pelo Elliptic++ para cada endereço ativo em
  um passo, usando `(address, Time step)` como unidade de predição. O endereço não equivale
  necessariamente a uma pessoa ou carteira com múltiplos endereços. Os rótulos não
  descrevem mudanças temporais de licitude ou datas de descoberta de atividade ilícita.
- Usar grafos endereço–transação independentes por passo, com transações daquele passo
  e endereços incidentes. A predição corresponde ao encerramento da fatia observada.
  O encoder compartilha pesos entre os grafos; não haverá um modelo separado por passo.
- Restringir ajuste de pré-processamento, SSL, modelos, calibração, fusão e limiar às
  respectivas partições permitidas de treino/validação. O teste não participa do ajuste,
  inclusive como dado não rotulado de pré-treinamento; inferência no grafo de teste é permitida.
- Fixar treino nos passos 1–28, validação nos passos 29–34 e teste nos passos 35–49,
  com limites inclusivos. A reserva interna de validação é uma adaptação do S02;
  não foi confirmada como reprodução de uma divisão exata da literatura.
- Separar classes de entradas e vistas contrastivas. Classes supervisionam apenas os
  estágios autorizados; classes reservadas de avaliação são consultadas após as predições.
- Medir escassez por endereços únicos rotulados de treino, com a mesma seleção para os
  métodos comparados. Repetições temporais não constituem novas anotações. O orçamento de
  validação é separado; o recorte de smoke não substitui o orçamento científico de rótulos.
- Adotar a opção A: cenário principal com atributos calculados exclusivamente a partir
  do grafo de cada passo nos dois ramos e cenário complementar limitado com atributos
  nativos. O complemento tem limitações temporais explícitas e quatro métodos a 100%
  dos rótulos, cinco seeds: 20 avaliações adicionais. A decisão abrange entradas de endereços,
  transações e classificadores downstream; não basta retirar três colunas do ramo H-GCL.
- BTC, taxas e tamanho de transação são requisitos do cenário principal, conforme
  esclarecimento posterior do pesquisador. O grafo do passo inclui propriedades locais
  das suas transações; os atributos de endereço são derivados das transações incidentes
  nesse passo. A definição anterior não deve ser interpretada como restrição à topologia.

A presença de uma linha no passo `t` não comprova que seus atributos foram calculados
somente com informação disponível até `t`. O [contrato de entradas](input-contract.md)
define catálogo e deduplicação. Recorrentes permanecem no teste, com resultados
separados para endereços já vistos e inéditos (D028).
Não se afirma antecipação de crimes, ausência de vazamento em atributos ou generalização
para endereços inéditos apenas pela separação dos grafos.

**Próxima atividade**: implementar as [tasks](tasks.md) na ordem do [plano](plan.md), usando o
[contrato de entradas](input-contract.md), o [desenho de treinamento](training-design.md)
e a [pesquisa](research.md). A INV-001 permanece preservada; a definição exata da unidade
publicada de tamanho será verificada no plano, sem retirar esse atributo nem assumir fee/byte.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Conhecer os dados efetivamente utilizados (Priority: P1)

Como pesquisador, quero um inventário verificável dos arquivos e de suas unidades de
observação para que as decisões do experimento usem o dataset real.

**Why this priority**: Chaves, repetições e atributos afetam tanto a construção do grafo
quanto a validade da avaliação.

**Independent Test**: Inspecionar os CSVs, conferir contagens e referências e reproduzir
seus identificadores de conteúdo, sem treinar modelos.

**Acceptance Scenarios**:

1. **Given** os nove CSVs originais, **When** a inspeção for executada, **Then** o relatório
   informa arquivo, hash, tamanho, cabeçalho, número de linhas, chaves e classes observadas.
2. **Given** endereços repetidos, **When** o relatório for gerado, **Then** distingue
   endereços únicos, pares endereço-tempo e linhas, sem deduplicar os originais.
3. **Given** um arquivo necessário ausente ou uma referência inválida, **When** a validação
   terminar, **Then** a falha é identificada por arquivo e o treinamento não é iniciado.
4. **Given** um atributo candidato e um endereço recorrente, **When** sua disponibilidade
   temporal for investigada, **Then** o registro distingue tempo da ocorrência, horizonte
   do cálculo, valores publicados, evidência verificável e limites da conclusão.

### User Story 2 - Validar o pipeline completo em uma amostra real (Priority: P1)

Como pesquisador, quero executar um treinamento pequeno na minha máquina e inspecionar
as saídas de todas as etapas antes de comprometer recursos do laboratório.

**Why this priority**: O smoke demonstra que as partes do pipeline funcionam em conjunto.

**Independent Test**: Executar o perfil smoke após preparar os dados e verificar seus
artefatos, integridade temporal e duração, sem executar a matriz experimental completa.

**Acceptance Scenarios**:

1. **Given** arquivos conferidos e um protocolo definido, **When** o smoke for executado,
   **Then** um recorte real percorre aprendizagem de representações, treinamento dos dois
   ramos, ajuste da fusão na validação e avaliação reservada.
2. **Given** a mesma versão dos arquivos e configuração de amostragem, **When** o recorte
   for reconstruído, **Then** os IDs e as partições selecionados são os mesmos.
3. **Given** preparação concluída no Mac de referência, **When** o treino smoke executar,
   **Then** termina em até 600 segundos ou é marcado como falha do critério de tempo;
   o relatório separa preparação, treinamento e avaliação.
4. **Given** uma execução concluída, **When** seus artefatos forem recarregados, **Then**
   é possível associar previsões às unidades avaliadas e recuperar a configuração usada.

### User Story 3 - Comparar os métodos sob um protocolo explícito (Priority: P2)

Como pesquisador, quero comparar os ramos e a fusão no laboratório, sob as mesmas
condições de informação, para avaliar a hipótese a ser estabelecida neste estudo.

**Why this priority**: É a finalidade científica do projeto, dependente do protocolo.

**Independent Test**: Para uma configuração experimental definida, verificar que todos
os métodos produzem previsões sobre as mesmas unidades e exportam as métricas acordadas.

**Acceptance Scenarios**:

1. **Given** treino, validação e teste definidos, **When** os modelos forem comparados,
   **Then** usam a mesma partição de avaliação e o orçamento de rótulos acordado.
2. **Given** modelos selecionados, **When** o teste final for avaliado, **Then** seus
   rótulos não influenciam treinamento, seleção, calibração, fusão ou limiar.
3. **Given** uma execução interrompida e um checkpoint válido, **When** ela for retomada,
   **Then** preserva a configuração e registra a retomada e o progresso recuperado.
4. **Given** resultados sem melhoria da fusão, **When** o relatório for gerado, **Then**
   registra esse resultado sem confundi-lo com falha de implementação.

### Edge Cases

- Ausência de uma das classes conhecidas no recorte: diagnóstico explícito e smoke inválido,
  sem inventar exemplos ou usar rótulos desconhecidos como negativos.
- Rótulos desconhecidos: participação conforme o protocolo de representações, sem tratá-los
  como rótulos supervisionados.
- Carteiras recorrentes ou linhas repetidas: aplicação da política explícita de identidade;
  nenhuma multiplicação silenciosa de exemplos.
- Nós isolados ou referências inexistentes após recorte: política documentada e relatório
  de contagens, sem alterar os CSVs originais.
- Poucas unidades para formar pares contrastivos: falha identificada antes da otimização.
- Memória insuficiente ou estouro de tempo: execução incompleta identificada como tal;
  parâmetros não são alterados silenciosamente para produzir um resultado aparentemente válido.
- Dados ou configuração diferentes ao retomar: incompatibilidade identificada antes da retomada.
- Probabilidades com classes em ordem diferente: alinhamento explícito pelo significado da classe.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O projeto DEVE preservar os CSVs de origem e produzir dados derivados separadamente.
- **FR-002**: A inspeção DEVE informar hashes, tamanhos, cabeçalhos, contagens, classes,
  tempos, duplicatas das chaves candidatas e integridade das referências necessárias ao pipeline.
- **FR-003**: A representação e os relatórios DEVEM distinguir ID externo, unidade de
  observação e posição interna dos dados.
- **FR-004**: O experimento DEVE responder à pergunta principal sobre fusão registrada
  em Research Questions, cuja redação foi aceita pelo pesquisador. A matriz principal de métodos está definida abaixo; detalhes de avaliação permanecem em FR-016.
- **FR-005**: A unidade de predição DEVE ser o par `(address, Time step)`, com alvo global
  por endereço. Linhas repetidas NÃO DEVEM gerar predições independentes para o mesmo par.
  Deduplicar chaves endereço–tempo e arestas por relação; derivar as entradas do cenário
  principal do grafo original do passo. O contrato normativo de nomes, agregações,
  ausências e entradas complementares está em [input-contract.md](input-contract.md).
- **FR-006**: Todo uso de dados no aprendizado de representações, pré-processamento,
  treinamento, seleção, calibração e fusão DEVE obedecer ao contrato temporal escolhido.
  Dados do teste NÃO DEVEM participar de ajuste, nem de pré-treinamento sem rótulos.
- **FR-007**: O perfil smoke DEVE usar um recorte reproduzível dos CSVs originais, com
  IDs e limites registrados e referências válidas entre nós e arestas.
- **FR-008**: O smoke DEVE executar todas as etapas principais propostas: aprendizagem
  contrastiva de representações, ramo tabular, classificador sobre representações, fusão
  e avaliação. As escolhas arquiteturais do anexo são candidatas para detalhamento no plano.
- **FR-009**: O smoke DEVE produzir perdas finitas, previsões e artefatos recarregáveis;
  os estágios treináveis devem atualizar seus parâmetros e checkpoints SSL reutilizáveis
  devem ser preservados. O encoder pré-treinado também é ajustado na fase supervisionada.
- **FR-010**: O relatório DEVE separar duração da preparação, treinamento e avaliação;
  a meta de treinamento no Mac é de até 600 segundos.
- **FR-011**: Smoke e execução completa DEVEM compartilhar o pipeline, diferenciando-se
  por configurações explícitas de volume, recursos e duração.
- **FR-012**: Toda execução DEVE registrar revisão do código, hashes de dados, configuração
  resolvida, seeds, hardware, versões do ambiente, IDs, previsões, métricas e status.
- **FR-013**: A matriz principal DEVE comparar Random Forest tabular, encoder de grafos
  supervisionado sem SSL, o mesmo encoder com pré-treinamento H-GCL e fusão tardia
  RF + H-GCL, nas mesmas unidades de avaliação. O complemento repete os quatro métodos
  a 100% dos rótulos e cinco seeds (20 avaliações), totalizando 100 com as 80 principais.
- **FR-014**: A redução de dados para smoke e o orçamento de rótulos supervisionados
  DEVEM ser controles distintos. Amostragem de escassez só pode consumir rótulos permitidos
  de endereços únicos do treino, usando a mesma seleção para os métodos comparados;
  ocorrências repetidas não ampliam a contagem de anotações. O orçamento usado na validação
  deve ser registrado separadamente e usar o mesmo percentual aplicado ao treino,
  calculado sobre endereços únicos com classe conhecida da própria validação. Os
  subconjuntos de validação DEVEM ser iguais entre métodos. Cada método principal DEVE ser avaliado com
  1%, 5%, 10% e 100% dos endereços únicos rotulados do treino (16 configurações por seed).
- **FR-015**: Classes desconhecidas NÃO DEVEM entrar como alvos supervisionados nem
  como verdade de referência nas métricas binárias.
- **FR-016**: A política de avaliação DEVE ser fixada antes de consultar o teste final:
  treino nos passos 1–28, validação nos passos 29–34 e teste nos passos 35–49, inclusive.
  A métrica principal DEVE ser F1 da classe ilícita sobre os pares endereço–tempo
  rotulados do teste; métricas complementares: precisão, recall, Average Precision
  (AP) e MCC, com ilícito como classe positiva. O limiar de cada modelo DEVE maximizar
  F1 apenas na validação permitida e ser congelado antes do teste. Cada configuração
  DEVE usar cinco seeds distintas, comuns entre métodos, com média e desvio-padrão
  das métricas e comparações pareadas por seed. A matriz principal totaliza 80
  avaliações; isso não implica 80 pré-treinamentos independentes. As contagens de
  classes disponíveis em cada subconjunto de validação DEVEM ser verificadas antes
  do treinamento; insuficiência não autoriza ampliar silenciosamente o orçamento.
- **FR-017**: A fusão DEVE alinhar explicitamente o significado das classes de ambos os
  ramos e ajustar pesos ou outros parâmetros somente com a informação permitida.
  O desenho inicial usa média ponderada dos scores e seleção de peso/limiar na validação,
  conforme [training-design.md](training-design.md), sem calibrador adicional.
- **FR-018**: Caminhos de dados e recursos computacionais DEVEM ser configuráveis por
  ambiente, preservando a identidade dos arquivos ao transferir o experimento ao laboratório.
- **FR-019**: Uma execução incompleta NÃO DEVE ser apresentada como resultado final.
  O modo completo deve salvar progresso que permita retomada com verificação de compatibilidade.
- **FR-020**: Os relatórios DEVEM distinguir conclusão de engenharia, resultado científico
  e limitações da disponibilidade temporal dos dados.
- **FR-021**: Cada grafo DEVE conter apenas transações do seu passo e endereços/arestas
  incidentes permitidos. Não haverá adjacência entre passos ou acumulação de histórico no
  cenário principal; um encoder com pesos compartilhados será aplicado aos diferentes grafos.
- **FR-022**: Classes de endereços e transações NÃO DEVEM integrar atributos de entrada,
  seleção de arestas ou construção das vistas contrastivas. A seleção de exemplos rotulados
  para supervisão e o cálculo das métricas permanecem usos autorizados pelo protocolo.
- **FR-023**: Antes de fixar a política de atributos, a investigação INV-001 DEVE produzir
  evidências rastreáveis, incertezas e alternativas. Conclusões sobre uma coluna ou amostra
  NÃO DEVEM ser generalizadas automaticamente às demais. A decisão resultante DEVE ser
  incorporada a FR-005 e ao registro de decisões antes das tasks que dependem dela.
- **FR-024**: O cenário principal DEVE utilizar atributos derivados apenas do grafo do
  passo observado, incluindo BTC, taxas e tamanho das transações desse passo, nos dois
  ramos. Definições, unidades e disponibilidade dos campos são verificações do contrato
  técnico, não motivos para excluí-los silenciosamente. Isso inclui os nós de transação e os classificadores
  downstream. O cenário complementar DEVE usar uma política explícita de atributos
  nativos, identificada nos resultados e acompanhada de suas limitações temporais.
  Ambos DEVEM respeitar as mesmas regras de isolamento de ajuste e acesso a rótulos;
  o uso complementar de atributos nativos não autoriza ajuste com o teste.

### Key Entities *(include if feature involves data)*

- **Arquivo de origem**: CSV fornecido, identificado por conteúdo, cabeçalho e contagens.
- **Endereço**: identificador com classe global; pode ter várias ocorrências nos dados e
  não identifica necessariamente uma pessoa ou uma carteira completa.
- **Ocorrência temporal**: unidade de predição endereço–tempo; pode corresponder a várias
  linhas brutas, cuja política de consolidação será documentada.
- **Transação**: identificador com tempo e atributos, conectado a endereços de entrada e saída.
- **Protocolo**: definição da unidade, informação permitida, partições, orçamento e avaliação.
- **Recorte**: conjunto reproduzível de unidades e relações selecionadas dos dados originais.
- **Execução**: configuração, proveniência, modelos, checkpoints, previsões, métricas e estado.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Os nove CSVs têm identidade verificável e relatório de estrutura; todas
  as referências necessárias ao pipeline são conferidas antes de aceitar uma execução.
- **SC-002**: Duas construções do recorte com os mesmos dados, seed e configuração
  produzem IDs e partições idênticos.
- **SC-003**: O smoke conclui todas as etapas principais e seus treinamentos em até
  600 segundos no Mac de referência, após preparação; saídas não finitas invalidam a execução.
- **SC-004**: Cem por cento das execuções aceitas possuem registro de dados, configuração,
  código, ambiente, unidades avaliadas e estado final.
- **SC-005**: Os quatro métodos principais produzem previsões alinhadas para todas as mesmas
  unidades de teste, permitindo uma comparação sob as métricas acordadas.
- **SC-006**: Verificações de isolamento não detectam uso de rótulos do teste nas etapas
  de ajuste; a política de acesso a atributos e topologia é auditável.
- **SC-007**: É possível recarregar um checkpoint e recuperar os artefatos de uma execução
  pequena, verificando a compatibilidade de dados e configuração.
- **SC-008**: Cada atributo do escopo inicial de INV-001 possui um registro de definição,
  evidência, horizonte temporal avaliado, classificação da conclusão e limitações; a
  decisão do pesquisador sobre as entradas é rastreável a esse registro.
- **SC-009**: Verificações dos grafos aceitos confirmam que todas as transações pertencem
  ao passo declarado e não há conexões entre passos; as entradas e vistas não contêm classes.

Esses critérios medem conclusão do sistema. A confirmação da hipótese será julgada pelo
protocolo definido em FR-004 e FR-016; não se exige melhoria de métricas para declarar
que o software funciona.

## Assumptions

- O projeto tem como usuário principal o pesquisador do mestrado; execução local e no
  laboratório são suficientes para a primeira versão.
- A unidade endereço–tempo com rótulo global e os grafos independentes por passo foram
  acordados; atributos e protocolo estão detalhados nos contratos vinculados.
- As partições 1–28/29–34/35–49 foram aceitas. Os percentuais de treino 1%, 5%, 10% e 100%
  foram aceitos; a validação usa o mesmo percentual sobre seu próprio conjunto de
  endereços únicos rotulados (D026).
- Testes de engenharia podem usar fixtures pequenas em complemento; o smoke solicitado
  obrigatoriamente utiliza um recorte dos CSVs originais.
- Não há promessa de igualdade numérica entre dispositivos; amostragem deve ser reproduzível
  e eventuais fontes de não determinismo do treinamento devem ser documentadas.
- Dependências: decisão do protocolo, investigação da disponibilidade histórica de features
  e verificação do ambiente de software do laboratório.
- O número de nós e épocas do smoke será dimensionado para a meta aceita; não foi escolhido
  um tamanho arbitrário de amostra antes de medir o pipeline.

## Research Questions

### RQ1 — Pergunta principal aceita

> A fusão tardia entre um classificador tabular e um classificador baseado em H-GCL
> melhora a detecção de carteiras ilícitas no Elliptic++, em comparação com métodos
> de referência, sob diferentes níveis de disponibilidade de rótulos?

Redação preservada conforme pedido do pesquisador. A matriz principal foi aceita em D024.

### RQ2 — Pergunta complementar aceita

> O pré-treinamento autossupervisionado no grafo heterogêneo endereço–transação do
> Elliptic++ melhora a detecção de carteiras ilícitas em comparação ao mesmo encoder
> treinado apenas de forma supervisionada, especialmente quando há poucos rótulos?

A primeira pergunta avalia a fusão final; a segunda investiga o benefício do
pré-treinamento. O classificador downstream continua exigindo rótulos.
O pesquisador aceitou RQ2 e o controle com o mesmo encoder, com e sem pré-treinamento
SSL, na matriz principal D024. D027 fixa fine-tuning de encoder e cabeça após SSL.

Para isolar o efeito do pré-treinamento, a comparação aceita mantém arquitetura,
cabeça de classificação, partições e rótulos e usa o mesmo protocolo de fine-tuning
supervisionado, com inicialização autossupervisionada versus inicialização do zero.
O ramo pré-treinado congelado previsto no anexo constitui um regime separado. Compará-lo
ao treinamento supervisionado fim a fim envolve também a diferença de congelamento.
A implementação do controle, o orçamento computacional e ablações adicionais serão
definidos no plano; a matriz principal aceita está abaixo.

Inspection-L e GCPAL são referências candidatas de SSL; qualquer adaptação deve explicitar
a mudança de tarefa e representação. A inspiração em Inspection-L não significa que
H-GCL seja a arquitetura daquele artigo. A lacuna exata no Elliptic++ ainda exige revisão.

## Clarifications

### Session 2026-09-05

- Dados originais fornecidos pelo pesquisador em Downloads e depois colocados dentro
  do projeto em `elliptic-plus-plus/raw`.
- Teste inicial no Mac pessoal; experimento completo no laboratório do S00.
- Treinamento pequeno com meta de até 10 minutos, medindo preparação separadamente.
- Pergunta principal enviada nesta sessão; sua aceitação posterior consta em Research Questions.
- Inspeção factual: [relatório dos CSVs](../../docs/data/README.md).
- Contexto de ambientes: [ambientes e armazenamento](../../docs/environments.md).

### Session 2026-09-07

- O pesquisador considerou adequada a pergunta sobre fusão, mas questionou se limitar
  a redação aos ramos isolados contemplaria os métodos estudados no artigo do Elliptic++.
- A consulta ao artigo foi registrada na nota de leitura de `elmougy2023demystifying`
  no StanOS, com localizadores das seções experimentais.
- Redação proposta nesta sessão e posteriormente aceita:

  > A fusão tardia entre um classificador tabular e um classificador baseado em H-GCL
  > melhora a detecção de carteiras ilícitas no Elliptic++, em comparação com métodos
  > de referência, sob diferentes níveis de disponibilidade de rótulos?

- Proposta para o protocolo: manter os ramos isolados como análise da contribuição da
  fusão; selecionar referências externas tabulares e de grafos para avaliar desempenho
  comparativo. RF e XGBoost são candidatos tabulares; uma GNN supervisionada será escolhida
  por adequação à tarefa. Essa seleção ainda não é um requisito fechado de implementação.
- RF pode exercer simultaneamente o papel de ramo isolado e de referência tabular quando
  as configurações e as condições de treinamento coincidirem, sem duplicar execuções iguais.
- Comparações controladas devem usar a mesma unidade de predição, partições, política
  temporal, orçamento de rótulos e avaliação. Números publicados sob outro protocolo
  servem como contexto; uma execução adaptada não será chamada de reprodução exata.
- Um controle com o mesmo encoder sem pré-treinamento contrastivo é candidato a ablação
  para distinguir o efeito do SSL do efeito da fusão. A matriz final permanece em aberto.

### Session 2026-09-07 — Inspiração e pergunta complementar

- O pesquisador aceitou a redação revisada da pergunta principal e pediu que fosse
  preservada, considerando o acréscimo de uma pergunta sobre autossupervisão.
- Fonte de inspiração informada: Inspection-L, arXiv `2203.10465`.
- Fontes registradas no StanOS: `lo2022inspectionl` e `lu2024gcpal`; análise em
  `s02-ssl-research-question.md`, no framing do S02.
- RQ2 foi acrescentada como proposta. Nenhuma ausência de trabalhos relacionados foi
  assumida; a busca realizada foi preliminar e encontrou um precedente posterior.

### Session 2026-09-07 — Contrato temporal e investigação dos dados

- Q: Qual é a unidade e o alvo da predição? → A: Endereço ativo por passo, identificado
  pelo par `(address, Time step)`, com a classe global publicada, sem trajetória de licitude.
- Q: Qual contexto de grafo e compartilhamento de modelo serão usados? → A: Grafos
  independentes por passo, predição ao final da fatia e um encoder com pesos compartilhados.
- Q: Como proteger o teste e separar alvos de entradas? → A: Nenhum ajuste usa o teste,
  nem no SSL; classes ficam fora de atributos e vistas, sendo usadas na supervisão e avaliação permitidas.
- Q: Como contar o orçamento de rótulos? → A: Endereços únicos rotulados do treino,
  mesmos subconjuntos entre métodos e orçamento de validação registrado separadamente.
- Q: Como resolver a disponibilidade histórica dos atributos? → A: Investigar primeiro
  quantidade de passos com atividade, total de transações e último bloco; discutir a política
  de entradas com base nas evidências antes de fechar o plano e as tasks dependentes.
- RQ2 aceita pelo pesquisador após a proposta complementar; redação de RQ1 preservada.
- A autorização atual é persistir os acordos e preparar a investigação no SDD. Esta
  atualização não executa a auditoria dirigida nem implementa treinamento.
- Inspection-L permanece a inspiração temporal/SSL. O mesmo protocolo temporal não será
  atribuído ao GCPAL sem evidência específica; números de transações e endereços não são
  diretamente comparáveis. Fontes e limites estão em [research.md](research.md).

### Session 2026-09-07 — Execução autorizada da INV-001

- O pesquisador autorizou a execução de ponta a ponta da investigação. Foram criados
  scripts de análise e verificação, sem treinamento de modelos ou consulta a arquivos de classes.
- Investigação limitada concluída: [resultados e limites](../../docs/data/inv-001/README.md).
  Todos os seis critérios de entrega da investigação foram atendidos; a verificação
  independente confirmou os exemplos e a exceção, sem repetir a auditoria populacional inteira.
- A decisão D018 permanece aberta. A recomendação de entradas derivadas do passo nos
  dois ramos e atributos nativos em cenário complementar não foi adotada automaticamente.
- O protocolo completo, as tasks científicas e a arquitetura continuam em definição.

### Session 2026-09-07 — Escolha da política de atributos

- Q: Qual política de atributos adotar após a INV-001? → A: Opção A, aceita explicitamente:
  atributos por passo no cenário principal, nos dois ramos, e atributos nativos em um
  cenário complementar limitado. A matriz exata será definida posteriormente.
- D018 passa a confirmada. Catálogo de entradas, fórmulas, deduplicação, relação entre
  entradas dos ramos e abrangência das execuções complementares permanecem em definição.
- O alvo global por endereço e RQ1/RQ2 são preservados. Não há mudança automática de
  arquitetura, implementação de treinamento ou alegação de disponibilidade temporal dos rótulos.

### Session 2026-09-07 — Conteúdo financeiro e ritmo do SDD

- O pesquisador considera BTC, taxas e tamanho fundamentais para os modelos. Incorporar
  propriedades locais das transações do passo e resumos dessas propriedades nos endereços;
  não atribuir a cada endereço todo o valor financeiro de uma transação com múltiplos participantes.
- Reutilizar definições e métodos das referências, com adaptações explícitas à tarefa
  endereço–tempo. A proposta de atributos exclusivamente estruturais não foi aceita.
- Rodadas seguintes devem trazer perguntas curtas sobre escolhas científicas ainda abertas.
  Mapeamento de colunas, fórmulas usuais, validações e implementação ficam sob responsabilidade
  técnica do agente; retornar ao pesquisador apenas se alterarem escopo ou interpretação.

### Session 2026-09-07 — Partições temporais

- Q: Adotar treino 1–28, validação 29–34 e teste 35–49? → A: Sim, divisão aceita.
- Decisão D023: limites inclusivos e comuns aos métodos comparados. Preservar as
  restrições de ajuste já acordadas; orçamento da validação e demais itens de FR-016
  seguem em aberto. Não há autorização implícita para retreinar em treino + validação.
- O corte interno em 28/29 é uma escolha operacional do estudo, não um precedente
  exato confirmado. Manter a janela de teste não equivale a reproduzir todo o protocolo
  nem torna os resultados publicados diretamente comparáveis.

### Session 2026-09-07 — Matriz principal e percentuais aceitos

- O pesquisador aprovou os quatro métodos: Random Forest tabular; encoder de grafos
  supervisionado sem pré-treinamento SSL; mesmo encoder com pré-treinamento H-GCL;
  fusão tardia do tabular com H-GCL.
- Cada método será avaliado com 1%, 5%, 10% e 100% dos endereços únicos rotulados
  do treino, usando os mesmos subconjuntos entre métodos. O denominador exclui
  endereços de classe desconhecida; 100% é a referência de disponibilidade completa
  dos rótulos de treino, não de validação ou teste.
- São 16 configurações principais por seed. O smoke permanece um recorte de engenharia,
  sem obrigação de executar toda a matriz. O cenário complementar com atributos nativos
  continua limitado e terá sua seleção de execuções definida separadamente.
- Métricas, seeds e orçamento de validação permanecem abertos. A aprovação não fixa
  automaticamente arquitetura, classificador downstream, congelamento ou fine-tuning.

### Session 2026-09-07 — Avaliação e orçamento de validação aceitos

- O pesquisador aprovou F1 da classe ilícita como métrica principal, calculada sobre
  os pares endereço–tempo do teste; precisão, recall, AP e MCC como complementares.
- Cinco seeds, média e desvio-padrão e comparação pareada entre métodos: 80 avaliações
  principais. Pré-treinamento pode ser reutilizado onde entradas e configuração forem
  idênticas e não houver dependência dos rótulos amostrados.
- Limiar escolhido para maximizar F1 na validação e congelado antes do teste.
- Validação com o mesmo percentual do treino, aplicado aos endereços únicos rotulados
  da própria partição, e os mesmos subconjuntos entre métodos. Esse é o orçamento
  compartilhado pelas etapas de seleção, calibração, fusão e limiar; não um novo
  orçamento por etapa. Verificar contagens de classes antes de treinar.
- D026 consolida essas escolhas. Procedimento interno de ajuste, tratamento de
  endereços recorrentes e escopo complementar ainda precisam ser especificados.

### Session 2026-09-07 — Contrato e desenho de treinamento

- Q: Ajustar o encoder após SSL ou congelá-lo com RF? → A: Ajustar após SSL, com a
  mesma cabeça de classificação e protocolo supervisionado do encoder treinado do zero (D027).
- Q: Manter recorrentes no teste ou avaliar apenas inéditos? → A: Manter recorrentes
  e separar resultados de endereços já vistos e inéditos (D028).
- Q: Qual tamanho do complemento nativo? → A: Quatro métodos, 100% dos rótulos,
  cinco seeds: 20 avaliações adicionais (D029).
- Sob D022, o agente detalhou deduplicação, allowlists, resumos financeiros contextuais,
  preenchimento de ausências, arquitetura inicial e média ponderada em documentos vinculados.
  Esses detalhes são decisões técnicas rastreáveis, não aprovações adicionais atribuídas ao pesquisador.
- Registrar em cada avaliação a população geral e os estratos de recorrência; não mudar
  modelos ou limiares ao consultar os resultados de teste. A população principal é preservada.

## Clarification — unseen addresses (2026-09-11)

Prediction accepts a previously unseen address with the required snapshot features and
relations. A global label is a supervision/evaluation target, never an inference input.
IDs are opaque mapping keys, not learned identity vectors. The existing unseen-test
stratum evaluates this capability within Elliptic++; generalization to external data
remains unmeasured. Unknown labels do not prevent inference. Global labels do not locate
the prediction relative to crime onset. This clarifies D015/D028 without changing scope.

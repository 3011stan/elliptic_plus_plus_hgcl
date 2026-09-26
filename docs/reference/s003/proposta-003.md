S003-TxGCL: Classificação de Transações Ilícitas sob Escassez de Rótulos no Bitcoin

0. Decisões Consolidadas

* [DECIDIDO em 2026-09-26] O alvo primário do Estudo 003 é a classificação de transações ilícitas.
* [DECIDIDO em 2026-09-26] O experimento principal usará o grafo homogêneo direcionado de fluxo de valor entre transações (Tx→Tx). Essa escolha preserva a comparabilidade direta com Inspection-L, GCPAL e baselines supervisionados no Elliptic/Elliptic++.
* [DECIDIDO em 2026-09-26] O grafo heterogêneo bidirecional Addr↔Tx não será o núcleo obrigatório do estudo. Ele será tratado como extensão/ablação para medir o ganho marginal da representação explícita de endereços, condicionado à viabilidade computacional e à construção de atributos temporalmente causais.
* [DECIDIDO em 2026-09-26] A curva principal de eficiência de rótulos usará 1%, 5%, 10% e 100% dos nós rotulados disponíveis no conjunto de treino, com amostragem estratificada por classe e cinco sementes fixas. Todos os nós não rotulados dos time steps autorizados permanecem disponíveis no pré-treino auto-supervisionado. As ablações não precisam repetir toda a curva: poderão usar um regime principal definido no SDD.
* [DECIDIDO em 2026-09-26] Após o pré-treino, o encoder principal será congelado. O classificador downstream será um MLP leve alimentado pela concatenação H‖X, em que H é o embedding e X contém exclusivamente os 183 atributos da transação. No regime de 1%, serão comparadas as variantes X-only, H-only e H‖X. Fine-tuning será, no máximo, uma análise secundária explicitamente separada do resultado principal.
* [DECIDIDO em 2026-09-26] O pré-treino, o ajuste e o treinamento downstream usarão somente os time steps 1–34. A seleção de hiperparâmetros será feita por validação estratificada interna nesse intervalo, sem consultar 35–49. Após a seleção, o modelo final poderá usar todo o orçamento rotulado de 1–34. O teste permanecerá intocado em 35–49 e a inferência será executada separadamente por snapshot temporal.
* [DECIDIDO em 2026-09-26] O mecanismo contrastivo principal será estritamente auto-supervisionado e terá três visões: perturbação estocástica controlada, mascaramento por blocos funcionais sem rótulos e grafo de similaridade KNN. A perda multi-positivo seguirá o GCPAL: o próprio nó, vizinhos Tx→Tx e vizinhos KNN serão positivos e ficarão fora do conjunto de negativos. Taxas de perturbação, K e temperatura serão escolhidos somente na validação interna.
* [CONDICIONAL] A extensão Addr↔Tx somente será executada se passar pelo gate de causalidade e viabilidade da Seção 5.4; atributos globais de endereço não serão incorporados ao experimento principal.

1. Contextualização e Problema Operacional no Protocolo UTXO

A contínua integração dos ecossistemas de criptoativos aos mercados financeiros globais aumenta a demanda por métodos de inteligência forense e monitoramento para Prevenção à Lavagem de Dinheiro (AML, Anti-Money Laundering) e Combate ao Financiamento do Terrorismo (CFT). Na rede Bitcoin, a classificação automatizada de transações pode apoiar a priorização de investigações. Métodos tabulares constituem baselines fortes, enquanto Redes Neurais em Grafos (GNNs, Graph Neural Networks) permitem incorporar explicitamente a topologia e a dinâmica temporal dos fluxos. O alvo deste estudo é a classificação de transações; endereços serão avaliados apenas como contexto representacional na extensão condicionada.

1.1 A Dinâmica do Protocolo UTXO e os Datasets Elliptic / Elliptic++

O protocolo do Bitcoin baseia-se no modelo de Saídas de Transações Não Gastas (UTXO - Unspent Transaction Output). Diferente de arquiteturas baseadas em contas (como o Ethereum), uma única transação no Bitcoin agrega múltiplos endereços de entrada (inputs) e distribui fundos para múltiplos endereços de saída (outputs). Abordagens pioneiras de detecção de anomalias concentraram-se em grafos homogêneos transação-para-transação. A camada de endereços oferece uma visão complementar das relações de entrada e saída, mas não identifica automaticamente usuários reais nem estabelece responsabilização, pois um ator pode controlar múltiplos endereços e as heurísticas de agrupamento introduzem incerteza.

Para superar essa limitação, o dataset Elliptic++ estende o dataset original Elliptic, mapeando explicitamente os atores (endereços de carteiras) e suas interações temporais junto ao fluxo de capital. A Tabela 1 sintetiza a transição metodológica e as especificações estruturais de ambos os conjuntos de dados.

Tabela 1: Comparação Estrutural entre os Datasets Elliptic e Elliptic++

Dimensão / Característica	Dataset Elliptic (Original)	Dataset Elliptic++ (Estendido)
Número de Transações (Nós de Transação)	203.769 transações	203.769 transações
Número de Carteiras / Atores (Endereços)	Ausente (não mapeado)	822.942 endereços de carteira (actors)
Interações Temporais	49 time steps discretos	1.268.260 ocorrências temporais de endereços ao longo dos 49 time steps
Atributos por Nó	166 atributos de transação (93 locais, 72 agregados, 1 time step)	183 atributos de transação (166 originais + 17 aumentados) e 56 atributos por carteira/endereço
Tipologias de Grafos Suportadas	Money Flow Transaction Graph (Homogêneo transação-para-transação)	1. Money Flow Transaction Graph<br>2. Actor Interaction Graph<br>3. Address-Transaction Graph<br>4. User Entity Graph

A adição de 17 atributos aumentados nas transações do Elliptic++ inclui métricas agregadas fundamentais: volumes de entrada e saída (BTCin, BTCout), contagem de transações associadas (Txsin, Txsout), contagem de endereços envolvidos (Addrin, Addrout), volume total transacionado (BTCtotal), taxas de rede (Fees) e tamanho da transação em bytes (Size). As métricas BTCin e BTCout desdobram-se em 5 estatísticas descritivas (total, mínimo, máximo, média e mediana), consolidando a caracterização do nó.

Na camada de atores, os 56 atributos por carteira englobam variáveis financeiras e temporais desdobradas estatisticamente (total, mínimo, máximo, média e mediana), tais como volume transacionado (BTCtransacted), volume enviado (BTCsent), volume recebido (BTCreceived), taxas pagas (Fees), proporção das taxas (Feesshare), além de intervalos de blocos entre transações (Blockstxs, Blocksinput, Blocksoutput) e contagem de interações com outros endereços (Addr_interactions). Variáveis escalares adicionais detalham a vida útil da carteira (Lifetime), blocos iniciais e finais (Blockfirst, Blocklast, Blockfirst_sent, Blockfirst_receive), total de transações (Txstotal, Txsinput, Txsoutput), número de janelas ativas (Timesteps) e interações repetidas com os mesmos atores (Repeat_interactions).

As quatro tipologias topológicas suportadas no Elliptic++ oferecem visões complementares do protocolo:

* Money Flow Transaction Graph: Grafo direcionado mapeando a transferência de valor diretamente entre transações.
* Actor Interaction Graph: Mapeia interações par a par entre endereços de entrada e saída.
* Address-Transaction Graph: Grafo heterogêneo bi-direcional representando o fluxo efetivo do BTC entre endereços e transações (das carteiras de entrada para as transações e destas para as carteiras de saída).
* User Entity Graph: Agrupamento por heurísticas de clustering para identificar endereços controlados pela mesma entidade.

1.2 O Gargalo da Escassez de Rótulos e a Ineficiência do Descarte Supervisionado

A obtenção de rótulos confiáveis no ecossistema de criptomoedas representa um gargalo operacional crítico, exigindo auditorias periciais complexas e cruzamento de informações fora da rede (off-chain). Consequentemente, a imensa maioria dos dados do blockchain permanece não rotulada. No dataset Elliptic/Elliptic++, essa assimetria traduz-se na seguinte distribuição de classes:

* Classe 1 (Ilícita): ~2% das transações (4.545 nós) e ~2% dos endereços (14.266 carteiras).
* Classe 2 (Lícita): ~21% das transações (42.019 nós) e ~31% dos endereços (251.088 carteiras).
* Classe 3 (Desconhecida / Não Rotulada): ~77% das transações (157.205 nós) e ~67% dos endereços (557.588 carteiras). Ao considerar as 1.268.260 ocorrências temporais de endereços, a Classe 3 atinge 71% (900.788 ocorrências).

Em cenários operacionais caracterizados por extrema escassez de supervisão — tais como regimes com apenas 1%, 5% ou 10% de rótulos disponíveis —, métodos tabulares puramente supervisionados normalmente excluem as amostras da Classe 3 da otimização. GNNs supervisionadas, por outro lado, podem manter nós e arestas não rotulados durante a propagação de mensagens e restringir a função de perda aos nós rotulados.

Mesmo quando a topologia não rotulada é preservada, a ausência de sinal supervisionado para ~77% das transações limita o aprendizado discriminativo. O pré-treino auto-supervisionado busca explorar explicitamente atributos e estrutura desses nós antes da classificação downstream. No contexto regulatório AML, um Falso Negativo (deixar de identificar uma transação ilícita) acarreta custos relevantes, tornando Recall e MCC (Matthews Correlation Coefficient) métricas prioritárias em conjunto com Precision e F1.

A ineficiência demonstrada pelos modelos puramente supervisionados diante do descarte de dados brutos exige uma revisão crítica das abordagens propostas pela literatura recente e de suas limitações estruturais.

2. Análise Crítica da Literatura e Lacunas Metodológicas

O desenvolvimento do Aprendizado Auto-Supervisionado (SSL, Self-Supervised Learning) e do Aprendizado de Representações em Grafos Contraste (GCL, Graph Contrastive Learning) busca mitigar a dependência de dados rotulados ao construir representações latentes ricas a partir da própria topologia e dos atributos não rotulados. Contudo, a aplicação dessas técnicas ao protocolo UTXO revela limitações metodológicas na literatura corrente.

2.1 Sintonização do Estado da Arte (SSL e GNNs Financeiras)

A Tabela 2 compara as principais arquiteturas da literatura voltadas à detecção de anomalias no ecossistema do Bitcoin.

Tabela 2: Quadro Comparativo de Abordagens do Estado da Arte

Método / Estudo	Arquitetura Base	Estratégia SSL/GCL	Modelagem de Atores/Endereços	Limitações Estruturais
Inspection-L (Loa et al., 2022)	GNN (GIN / DGI)	Aprendizado Contraste via Deep Graph Infomax (DGI)	Ausente (opera exclusivamente sobre o grafo homogêneo de transações)	Restrito à topologia de transações; incapacidade estrutural de rastrear o endereço de origem e destino dos fundos.
GCPAL (2024)	GNN com visões estocásticas	Aprendizado Contraste com visões estocásticas e vizinhança KNN	Parcial / Indireta (sem nós de carteira explícitos)	Não incorpora os 56 atributos de atores nem reflete a topologia bi-direcional do protocolo UTXO.
HeteroGCL (2026)	GNN heterogênea com nós de transação e clusters temporais	Aprendizado contrastivo heterogêneo com aumentações guiadas por domínio/topologia	Ausente (endereços foram explicitamente excluídos por ruído e escalabilidade)	Não avalia a camada Addr↔Tx; sua heterogeneidade temporal não é diretamente comparável a uma modelagem explícita de endereços.
MIE-HetGRL (Zhang et al., 2025)	GNN heterogênea multiplex	Contraste local/global e fusão por atenção semântica	Não se aplica a Bitcoin; modela usuários, avaliações e itens	Estabelece precedente para SSL heterogêneo em fraude, mas não classifica transações nem usa Elliptic/Elliptic++.

2.2 Diagnóstico de Vulnerabilidades da Literatura

A desconstrução crítica dos métodos vigentes expõe três gargalos estruturais principais:

1. Restrição Homogênea do Inspection-L (Loa et al., 2022): A formulação baseada em DGI sobre grafos de Isomorfismo de Grafos (GIN) limita-se ao fluxo transação-para-transação. Sem a representação de nós de endereço, a arquitetura é incapaz de mapear a reputação das carteiras operantes. Sob o ponto de vista da perícia forense, mesmo que uma transação seja sinalizada como anômala, o modelo não identifica a entidade responsável pela injeção ou retirada do capital.
2. Aproximação do GCPAL (2024): A geração de visões sintéticas, a expansão de positivos por KNN e as perturbações estocásticas buscam suprir a falta de rotulagem, mas não modelam endereços como nós explícitos. Atributos agregados de carteira podem fornecer sinais comportamentais complementares, porém não demonstram isoladamente padrões multissalto como peeling chains ou smurfing.
3. Escopo do HeteroGCL (2026): O HeteroGCL modela transações e clusters temporais, excluindo explicitamente nós de endereço por preocupações de ruído e escalabilidade. Assim, a lacuna relevante para este estudo não é uma suposta rigidez de metacaminhos Addr↔Tx, mas a ausência de uma avaliação controlada do ganho marginal da camada explícita de endereços para classificar transações.

A identificação dessas lacunas direciona a concepção do Estudo 003: estabelecer primeiro um resultado comparável no grafo Tx→Tx e, depois, isolar por ablação o efeito de acrescentar a topologia Addr↔Tx.

3. Arquitetura Proposta: Pipeline Contrastivo com Núcleo Tx→Tx

A arquitetura será projetada para atuar em cenários de escassez de supervisão. O experimento principal aplicará pré-treinamento auto-supervisionado contrastivo ao grafo Tx→Tx dos time steps autorizados, utilizando os nós não rotulados sem empregar seus rótulos. Uma extensão controlada substituirá o encoder homogêneo por um encoder Addr↔Tx para medir se a camada explícita de endereços produz ganho sobre o mesmo alvo de classificação.

+-----------------------------------------------------------------------------------+
|                    EXPERIMENTO PRINCIPAL: GRAFO Tx -> Tx                          |
|                                                                                   |
|             ( Transação origem ) ----fluxo----> ( Transação destino )             |
|                   (183 atributos)                    (183 atributos)               |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|            PRÉ-TREINAMENTO CONTRASTIVO AUTO-SUPERVISIONADO (S003-TxGCL)           |
|                                                                                   |
|  Visão 1: perturbação estocástica controlada                                     |
|  Visão 2: mascaramento por blocos funcionais, sem rótulos                        |
|  Visão 3: grafo KNN para expansão de positivos                                   |
|                                  \       |       /                                |
|                                   v      v      v                                 |
|                          InfoNCE multi-positivo                                   |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                       TRANSFERÊNCIA E CLASSIFICAÇÃO DOWNSTREAM                    |
|                                                                                   |
|       Encoder congelado; H || X_tx  ===> Classificador Leve (MLP)                 |
|                                      (Treinado sob 1%, 5%, 10% ou 100% de Rótulos) |
+-----------------------------------------------------------------------------------+

Extensão/ablação: substituir o grafo Tx→Tx pelo grafo heterogêneo Addr↔Tx, mantendo o alvo em nós-transação e controlando rigorosamente custo, protocolo temporal e disponibilidade causal dos atributos de endereço.

3.1 Modelagem Principal em Grafo de Transações

O experimento principal estrutura a rede como um grafo homogêneo direcionado \mathcal{G}_{Tx} = (\mathcal{V}_{Tx}, \mathcal{E}_{Tx}), no qual cada nó representa uma transação e cada aresta representa fluxo de valor entre transações. Cada nó possui os 183 atributos de transação do Elliptic++. O experimento principal não incorpora atributos de endereço na concatenação downstream; esses atributos pertencem exclusivamente à extensão condicionada.

A passagem de mensagens do encoder principal agrega a vizinhança Tx→Tx por um GIN, mantendo a família de encoder usada por Inspection-L e GCPAL. Arquitetura, dimensão e orçamento de parâmetros deverão ser mantidos equivalentes nas comparações que isolam apenas a estratégia de pré-treino.

Na extensão heterogênea, serão adicionados nós Addr e relações Addr→Tx e Tx→Addr. Essa extensão não poderá usar atributos globais de endereço calculados com eventos futuros. Sua comparação com o núcleo Tx→Tx deverá manter o mesmo split, alvo, orçamento de rótulos, sementes e protocolo de seleção de hiperparâmetros.

3.2 Pré-Treino Contrastivo Auto-Supervisionado

O pré-treino será executado nos snapshots 1–34 com parâmetros compartilhados entre snapshots e sem empregar rótulos, direta ou indiretamente, na construção das visões. Cada transação será representada em três visões:

1. Visão estocástica de fluxo: aplica feature masking e edge dropout controlados ao snapshot Tx→Tx. Essa visão serve como referência comparável ao GCPAL. O edge dropout é uma perturbação de robustez; não será descrito como preservação formal das equações UTXO.
2. Visão de mascaramento por blocos funcionais: mascara grupos predefinidos exclusivamente a partir do dicionário de dados, sem usar classe ou informação mútua com rótulos. Os blocos distinguirão, no mínimo, atributos locais originais, atributos agregados de vizinhança e os 17 atributos aumentados do Elliptic++. O time step e identificadores não serão alvos de mascaramento. A relação completa de índices de cada bloco será um artefato versionado do SDD.
3. Visão de similaridade KNN: conecta transações próximas no espaço de atributos de entrada normalizados, seguindo o princípio de expansão de positivos do GCPAL. O normalizador será ajustado somente com dados de 1–34 e o índice KNN será construído separadamente dentro de cada snapshot, sem criar arestas entre passado e futuro.

Para a âncora i, o conjunto P(i) seguirá o GCPAL e conterá a própria transação, seus vizinhos no grafo Tx→Tx e seus vizinhos na visão KNN. Esses nós serão positivos e serão excluídos do conjunto de negativos N(i). A perda principal manterá a formulação multi-positivo comparável ao GCPAL, com a soma dos positivos dentro do log:

\mathcal{L}_{i} =
-\log
\frac{\sum_{p \in P(i)}\exp(\operatorname{sim}(z_i,z_p)/\tau)}
{\sum_{k \in P(i)\cup N(i)}\exp(\operatorname{sim}(z_i,z_k)/\tau)}

Não será empregado hard-negative mining. As taxas de feature masking e edge dropout, o número K de vizinhos, a temperatura \tau e eventuais pesos entre visões serão hiperparâmetros selecionados na validação interna de 1–34. A proposta não fixa valores ou intervalos como resultados antecipados; o espaço de busca e seu orçamento serão definidos no SDD e registrados antes do teste final.

3.3 Classificador Downstream e Transferência de Aprendizado

Após a etapa de pré-treinamento contrastivo nos time steps de treino, as representações latentes H são extraídas com o encoder congelado. O classificador downstream principal será um Perceptron Multicamadas (MLP) leve, cuja entrada é a concatenação H‖X entre os embeddings e os 183 atributos brutos de transação. Nenhum atributo global de endereço integra essa entrada principal.

O classificador downstream será avaliado com 1%, 5%, 10% e 100% dos nós rotulados das Classes 1 e 2 disponíveis no conjunto de treino. Cada subconjunto será amostrado de forma estratificada por classe para cada uma das cinco sementes fixas; os subconjuntos serão aninhados dentro de cada semente. O orçamento de cada fração inclui os rótulos usados para ajuste e validação interna, evitando utilizar rótulos adicionais fora do regime declarado. Após a seleção, o classificador final poderá ser reajustado sobre todo o subconjunto da fração antes do teste. O regime de 100% atuará como teto de referência, enquanto 1%, 5% e 10% formarão a curva de escassez comparável ao GCPAL. Todos os nós não rotulados dos time steps de treino permanecerão disponíveis no pré-treino, sem contribuir com rótulos para a perda supervisionada.

Para isolar a contribuição do encoder, o regime de 1% incluirá as ablações X-only, H-only e H‖X usando o mesmo MLP, subconjunto rotulado e sementes. O resultado principal permanecerá baseado no encoder congelado. Eventual fine-tuning end-to-end será reportado apenas como análise secundária/limite prático e não substituirá essa comparação.

A validação da arquitetura S003-TxGCL exige um protocolo experimental rigoroso, imune a contaminações temporais e capaz de auditar a qualidade geométrica do espaço latente gerado.

4. Protocolo Experimental e Métricas de Auditoria Latente

A avaliação empírica de modelos aplicados a redes financeiras dinâmicas requer o isolamento rigoroso entre informações do passado e do futuro, evitando o vazamento de dados (data leakage) e simulando as condições reais enfrentadas por analistas forenses.

4.1 Configuração Indutiva e Split Temporal

O protocolo experimental adota uma divisão temporal estrita fundamentada na ordenação cronológica dos 49 time steps presentes no dataset:

* Pré-Treinamento, Validação Interna e Treino Downstream (Time Steps 1 a 34): Os primeiros 34 time steps (~70% do período temporal) são destinados ao pré-treino contrastivo auto-supervisionado (utilizando 100% dos nós existentes no intervalo), à validação estratificada interna e ao treinamento downstream sob regimes de 1%, 5%, 10% e 100% dos rótulos de treino. Nenhuma escolha de arquitetura, hiperparâmetro, época ou threshold poderá consultar os passos 35–49. Após a seleção, o modelo final poderá ser reajustado usando todo o orçamento rotulado de 1–34.
* Avaliação Indutiva Futura (Time Steps 35 a 49): Os 15 time steps finais (~30% do período temporal) permanecem intocados até a avaliação final. A inferência será executada separadamente em cada snapshot, sem disponibilizar ao passo t conexões ou atributos de passos posteriores. Endereços serão considerados apenas na extensão heterogênea e somente com informação disponível até o snapshot avaliado.

A avaliação sobre os time steps 35–49 será reportada tanto de forma agregada quanto por snapshot, permitindo observar degradação temporal e mudanças de regime sem disponibilizar informações futuras ao modelo. Normalizadores, imputadores, seleção de features, índice KNN e thresholds de decisão serão ajustados exclusivamente com dados de 1–34.

4.2 Métricas de Desempenho e Qualidade das Representações

As métricas primárias serão MCC e F1 da classe ilícita. Precision e Recall da classe ilícita serão reportados para explicitar o compromisso operacional entre falsos positivos e falsos negativos. PR-AUC será incluída por ser informativa sob forte desbalanceamento; ROC-AUC e Macro-F1 poderão ser apresentadas como métricas secundárias de comparabilidade. Todas serão reportadas como média ± desvio padrão nas cinco sementes, além dos resultados por snapshot nos passos 35–49.

A qualidade geométrica dos embeddings será auditada por alignment e uniformity, conforme a literatura contrastiva, e por effective rank via SVD como diagnóstico complementar:

p_i = \frac{\sigma_i}{\sum_{j=1}^{d}\sigma_j}

\operatorname{EffectiveRank}(Z) =
\exp\left(-\sum_{i=1}^{d}p_i\ln p_i\right)

Não será adotado um threshold universal como EffectiveRank > 0,85d, pois esse limite não é sustentado pelas fontes. Nenhuma dessas métricas substitui o desempenho downstream; elas servem para comparar colapso, alinhamento e dispersão entre variantes sob o mesmo protocolo.

5. Hipóteses, Baselines e Matriz Experimental

5.1 Hipóteses Científicas

* H1 — Eficiência de rótulos: o pré-treino contrastivo com encoder congelado melhora MCC e F1 ilícito sobre baselines supervisionados e sobre X-only, sobretudo em 1% e 5% de rótulos.
* H2 — Mascaramento funcional: substituir o feature masking independente por blocos funcionais melhora a robustez temporal sem utilizar classes durante o pré-treino.
* H3 — Expansão de positivos: a visão KNN reduz o efeito de falsos negativos da InfoNCE e melhora o downstream em relação à mesma arquitetura sem KNN.
* H4 — Informação de atores: quando a reconstrução causal de atributos de endereço for viável, a extensão Addr↔Tx produz ganho incremental na classificação das mesmas transações em relação ao núcleo Tx→Tx.

5.2 Baselines Obrigatórios

O estudo deverá incluir, sob o mesmo split temporal e as mesmas sementes:

* baselines tabulares X-only, incluindo o MLP usado nas ablações e ao menos um método de árvore comparável à literatura;
* GNNs supervisionadas Tx→Tx, como GCN/GraphSAGE/GIN, sem pré-treino;
* um baseline DGI/Inspection-L compatível com o protocolo;
* um baseline GCPAL ou reprodução funcional equivalente com feature/edge dropout e visão KNN;
* o método proposto completo e suas ablações;
* para a extensão heterogênea, um encoder supervisionado Addr↔Tx e a variante contrastiva proposta, ambos avaliados sobre os mesmos nós-transação.

5.3 Matriz de Avaliação com Custo Controlado

Tabela 3: Matriz consolidada

Prioridade	Comparação	Regimes de rótulos	Sementes	Resultado principal
P1	Método completo vs. baselines finais	1%, 5%, 10%, 100%	5	MCC e F1 ilícito no teste 35–49
P1	X-only vs. H-only vs. H‖X	1%	5	Ganho marginal da representação e da fusão
P1	Completo vs. sem KNN vs. masking aleatório vs. sem edge dropout	1%	5	Contribuição de cada aumento; alignment, uniformity e métricas downstream
P2	Fine-tuning end-to-end vs. encoder congelado	1% e 100%, somente se houver orçamento	5	Limite prático, reportado separadamente
P2	Addr↔Tx vs. Tx→Tx	1%, condicionado ao gate de viabilidade	5	Ganho downstream e custo incremental

As ablações não serão cruzadas com toda a curva de rótulos. Primeiro será feito um dry-run em uma semente para validar dados, ausência de leakage, consumo de memória e tempo. A matriz definitiva somente será executada depois desse aceite operacional.

5.4 Gate para a Extensão Addr↔Tx

A extensão heterogênea somente entrará na matriz definitiva se:

1. os atributos de endereço puderem ser reconstruídos usando apenas eventos disponíveis até cada snapshot, ou forem substituídos por atributos causais explicitamente documentados;
2. nenhum rótulo de endereço derivado de transações futuras for usado como feature ou supervisão da tarefa principal;
3. o dry-run couber no orçamento computacional definido no SDD;
4. a comparação preservar alvo, splits, subconjuntos rotulados, sementes, classificador e métricas do experimento Tx→Tx.

Se qualquer condição falhar, a extensão será registrada como trabalho futuro, sem impedir a conclusão do experimento principal. A contribuição central permanecerá sendo o mecanismo contrastivo auto-supervisionado por blocos funcionais, avaliado sob escassez de rótulos e protocolo temporal indutivo.

6. Contribuição Científica Pretendida

O Estudo 003 não reivindicará ser o primeiro uso de GCL em fraude nem o primeiro grafo heterogêneo para detecção de fraude. A contribuição pretendida é uma avaliação causal e reproduzível de pré-treino contrastivo para transações Bitcoin no Elliptic++, combinando: (i) mascaramento funcional de atributos sem rótulos; (ii) expansão de positivos por similaridade para mitigar falsos negativos; (iii) auditoria geométrica e downstream sob 1%, 5%, 10% e 100% dos rótulos; e (iv) uma extensão controlada Addr↔Tx, caso passe pelo gate de viabilidade.

O diferencial publicável dependerá de demonstrar ganho consistente sobre X-only, GNNs supervisionadas, DGI/Inspection-L e GCPAL, com cinco sementes, teste temporal intocado e ablações que atribuam o ganho aos componentes propostos. Resultados negativos da extensão heterogênea também serão informativos se mostrarem que o custo adicional de representar endereços não se converte em ganho para a classificação de transações.

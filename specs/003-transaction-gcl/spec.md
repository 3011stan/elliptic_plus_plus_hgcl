# Feature Specification: S003-TxGCL Transaction Classification

**Feature Branch**: `003-transaction-gcl`

**Created**: 2026-09-27

**Status**: Draft

**Input**: Iniciar o SDD do Estudo 003 a partir da proposta científica consolidada para classificar transações ilícitas no Elliptic++ sob escassez de rótulos, sem herdar silenciosamente o protocolo do Estudo 002.

## Clarifications

### Session 2026-09-27

- Q: Como o orçamento rotulado de cada fração deve ser dividido entre ajuste e validação interna antes do teste final? → A: Split estratificado 80/20 dentro de cada fração; após a seleção, reajustar o classificador com 100% da mesma fração antes do teste.
- Q: Qual métrica deve controlar a seleção de hiperparâmetros, o early stopping e o threshold no conjunto interno de validação? → A: Maximizar F1 da classe ilícita; usar MCC como primeiro desempate.
- Q: Quais baselines devem ser obrigatórios na matriz principal de 1%, 5%, 10% e 100%? → A: MLP X-only, Random Forest, XGBoost, GCN, GraphSAGE, GIN supervisionado, Inspection-L/DGI e GCPAL.
- Q: Quais controles de mascaramento são obrigatórios para sustentar a contribuição dos blocos funcionais? → A: No regime de 1%, comparar mascaramento aleatório individual, grupos aleatórios de mesma cardinalidade e blocos funcionais.
- Q: Qual protocolo de sementes e inferência estatística deve sustentar as comparações primárias? → A: Usar as sementes fixas `[11, 23, 37, 53, 71]`, média e desvio padrão, diferenças pareadas, intervalo de confiança de 95%, teste t pareado bilateral e tamanho de efeito para as comparações com GCPAL e Inspection-L em 1% e 5%, com correção por comparações múltiplas predefinida no plano; significância isolada não basta para concluir superioridade.

### Session 2026-09-27 — revisão pós-plano com literatura

- Q: Como alinhar direção, encoder, features, seleção SSL, baselines, métricas e blindagem do teste após a auditoria no NotebookLM autorizado? → A: Usar propagação dirigida no núcleo e uma ablação condicionada com arestas reversas; GIN de 2 camadas e 128 dimensões; 182 features financeiras, mantendo `Time step` somente como metadado; épocas SSL fixas e predeclaradas após o dry-run, sem probe rotulado; F1 ilícito pooled como resultado principal acompanhado de resultados temporais; downstream nativo de Inspection-L e GCPAL; e shadow test em 1–34 com single-unblinding auditado de 35–49.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Executar o experimento principal sem vazamento temporal (Priority: P1)

Como pesquisador, quero executar o S003-TxGCL sobre snapshots de transações para obter resultados reproduzíveis de classificação de transações futuras sem usar informações dos passos de teste durante treinamento, validação ou seleção.

**Why this priority**: Este é o resultado científico mínimo do Estudo 003. Sem isolamento temporal e rastreabilidade, as métricas não podem sustentar a dissertação nem uma publicação.

**Independent Test**: Pode ser testado executando um dry-run completo com dados originais, seguido de auditoria dos manifests de partição, transformações, seleção e avaliação, demonstrando que apenas os passos 1–34 influenciaram o modelo e que 35–49 foram usados somente na avaliação final por snapshot.

**Acceptance Scenarios**:

1. **Given** os CSVs originais verificados do Elliptic++, **When** o pesquisador prepara o experimento principal, **Then** são produzidos snapshots homogêneos `Tx→Tx`, com nós-transação, arestas de fluxo, 183 colunas não identificadoras preservadas e 182 features financeiras como entrada, sem `Time step`, identificadores ou atributos de endereço em `X_tx`.
2. **Given** um experimento ainda não avaliado, **When** ocorre pré-treino, validação ou seleção, **Then** nenhuma aresta, atributo, rótulo, métrica ou estatística ajustada dos passos 35–49 influencia essas etapas.
3. **Given** um modelo final selecionado, **When** os passos 35–49 são avaliados, **Then** cada snapshot é processado sem acesso a conexões ou atributos de snapshots posteriores e os resultados são reportados individualmente e de forma agregada.

---

### User Story 2 - Medir eficiência de rótulos e contribuição do pré-treino (Priority: P1)

Como pesquisador, quero comparar o método proposto com baselines e ablações sob diferentes orçamentos de rótulos para determinar se as representações contrastivas acrescentam informação útil além dos atributos brutos.

**Why this priority**: A contribuição científica depende de separar ganhos do pré-treino, da concatenação de atributos e da capacidade supervisionada, especialmente sob escassez extrema.

**Independent Test**: Pode ser testado produzindo a curva de 1%, 5%, 10% e 100% nas sementes `[11, 23, 37, 53, 71]` e, no regime de 1%, comparando `X-only`, `H-only`, `H‖X_tx` e as ablações das visões contrastivas — inclusive os três controles de mascaramento — com exatamente os mesmos subconjuntos rotulados.

**Acceptance Scenarios**:

1. **Given** os nós rotulados dos passos 1–34, **When** os orçamentos são gerados para uma semente, **Then** eles são estratificados por classe, aninhados entre 1%, 5%, 10% e 100%, divididos internamente em 80% para ajuste e 20% para validação, e incluem todos os rótulos consumidos nessas etapas.
2. **Given** embeddings produzidos sem rótulos, **When** ocorre a avaliação downstream principal, **Then** o encoder permanece congelado e o mesmo classificador compara `X-only`, `H-only` e `H‖X_tx` no subconjunto de 1%.
3. **Given** o método completo e seus baselines finais, **When** a matriz principal termina, **Then** cada combinação declarada de método, fração e semente possui resultado verificável ou falha explícita, sem substituição silenciosa do protocolo.
4. **Given** os resultados pareados das cinco sementes, **When** as comparações primárias com GCPAL e Inspection-L são analisadas em 1% e 5%, **Then** o relatório apresenta diferenças pareadas, intervalo de confiança de 95%, teste t pareado bilateral, tamanho de efeito e a correção por comparações múltiplas predefinida, sem tratar significância estatística isolada como evidência suficiente de superioridade.

---

### User Story 3 - Auditar a origem de cada resultado (Priority: P1)

Como orientador, revisor ou pesquisador, quero reconstruir a proveniência de qualquer número reportado para confirmar os dados, a configuração, os subconjuntos rotulados e a revisão do estudo que o produziram.

**Why this priority**: Resultados sem proveniência não são reproduzíveis nem defensáveis academicamente.

**Independent Test**: Pode ser testado escolhendo uma linha do relatório final e recuperando, a partir de seu identificador, hashes dos dados, configuração, revisão, semente, IDs amostrados, partições, artefatos de seleção e métricas por snapshot.

**Acceptance Scenarios**:

1. **Given** um resultado final, **When** seu identificador é consultado, **Then** todos os insumos e decisões necessários à reprodução são localizáveis sem depender de arquivos do Estudo 002.
2. **Given** uma execução interrompida ou inválida, **When** o relatório é consolidado, **Then** seu estado e motivo aparecem explicitamente e não são tratados como resultado concluído.
3. **Given** duas execuções com a mesma configuração e semente, **When** os manifests são comparados, **Then** partições, subconjuntos rotulados e identidades dos dados são iguais.

---

### User Story 4 - Decidir sobre a extensão heterogênea sem comprometer o núcleo (Priority: P3)

Como pesquisador, quero avaliar previamente a causalidade e o custo do grafo `Addr↔Tx` para decidir se a extensão heterogênea entra na matriz, sem bloquear o experimento homogêneo principal.

**Why this priority**: A extensão pode acrescentar novidade, mas apresenta risco de vazamento temporal e custo elevado. Ela não deve colocar em risco a conclusão do núcleo publicável.

**Independent Test**: Pode ser testado executando o gate documentado com um dry-run e obtendo uma decisão reproduzível de incluir ou adiar a extensão.

**Acceptance Scenarios**:

1. **Given** atributos globais de endereço que incorporam eventos futuros, **When** o gate de causalidade é aplicado, **Then** esses atributos são rejeitados para o snapshot corrente.
2. **Given** uma representação causal de endereços e recursos suficientes, **When** o dry-run heterogêneo é aprovado, **Then** a extensão usa o mesmo alvo, splits, subconjuntos, sementes, classificador e métricas do núcleo `Tx→Tx`.
3. **Given** falha em qualquer condição do gate, **When** o escopo final é consolidado, **Then** a extensão é registrada como trabalho futuro e o experimento principal continua inalterado.

### Edge Cases

- A fração de 1% deve preservar exemplos de ambas as classes; qualquer impossibilidade deve invalidar a semente antes do treinamento.
- Nós desconhecidos podem participar do pré-treino e da propagação estrutural, mas nunca fornecem rótulo à perda downstream.
- Transações sem vizinhos elegíveis em um snapshot devem permanecer representáveis sem criação de arestas para passos futuros.
- Vizinhos `Tx→Tx` ou KNN usados como positivos não podem simultaneamente aparecer como negativos para a mesma âncora.
- Snapshots vazios para uma classe devem ser reportados sem produzir métricas enganosas ou divisão por zero.
- Falhas, interrupções e retomadas não podem duplicar resultados nem sobrescrever artefatos válidos de outra configuração.
- Normalização, imputação, seleção de atributos, índice KNN e threshold não podem ser ajustados com os passos 35–49.
- O time step e identificadores não podem ser mascarados como se fossem atributos financeiros.
- Artefatos e configurações do S02 não podem ser selecionados por defaults ou nomes ambíguos.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O estudo MUST identificar toda execução, configuração e artefato com `study_id=s003` e um identificador prefixado por `s003-`.
- **FR-002**: O experimento principal MUST classificar nós-transação, usando snapshots homogêneos, direcionados e temporais `Tx→Tx`.
- **FR-003**: O experimento principal MUST preservar as 183 colunas não identificadoras do Elliptic++ (`Time step` mais 182 features financeiras), MUST usar somente as 182 features financeiras em `X_tx` e MUST NOT incluir `Time step`, identificadores ou atributos de endereço na entrada do encoder ou na concatenação downstream.
- **FR-004**: O sistema MUST preservar os CSVs originais sem alteração e registrar suas identidades verificáveis em cada preparação de dados.
- **FR-005**: O sistema MUST reservar os time steps 1–34 para pré-treino, validação interna e treino, e MUST reservar 35–49 exclusivamente para avaliação final.
- **FR-006**: Toda transformação ajustável, seleção de modelo, escolha de hiperparâmetro e threshold MUST usar somente informações permitidas dos passos 1–34. Eventual threshold ajustável MUST ser escolhido na validação interna e congelado antes do teste.
- **FR-007**: A inferência final MUST processar cada snapshot de 35–49 sem acessar snapshots posteriores.
- **FR-008**: O pré-treino principal MUST compartilhar o encoder entre snapshots de 1–34 e MUST NOT usar rótulos para construir visões, positivos, negativos ou a perda contrastiva.
- **FR-009**: O pré-treino MUST oferecer três visões: perturbação estocástica controlada, mascaramento por blocos funcionais e similaridade KNN.
- **FR-010**: Os blocos de mascaramento MUST ser derivados do dicionário de dados, versionados e independentes das classes; MUST distinguir atributos locais, agregados de vizinhança e os 17 atributos aumentados.
- **FR-011**: O time step e identificadores MUST NOT ser alvos do mascaramento funcional.
- **FR-012**: O índice KNN MUST ser construído separadamente por snapshot a partir de atributos normalizados sem criar conexões entre passos temporais.
- **FR-013**: Para cada âncora, o próprio nó, seus vizinhos `Tx→Tx` e seus vizinhos KNN MUST formar o conjunto positivo; nenhum membro desse conjunto pode integrar seus negativos.
- **FR-014**: A perda multi-positivo MUST manter uma variante diretamente comparável à formulação do GCPAL, permitindo atribuir diferenças ao mascaramento funcional.
- **FR-015**: O encoder principal MUST ser um GIN de duas camadas com dimensão escondida e embedding 128, alinhado estruturalmente aos baselines diretos Inspection-L e GCPAL; diferenças inevitáveis de orçamento MUST ser registradas nas análises que isolam o pré-treino.
- **FR-016**: O resultado downstream principal MUST congelar o encoder e treinar um classificador leve sobre `H‖X_tx`.
- **FR-017**: O estudo MUST avaliar 1%, 5%, 10% e 100% dos rótulos conhecidos dos passos 1–34 nas cinco sementes fixas `[11, 23, 37, 53, 71]`.
- **FR-018**: Para cada semente, os subconjuntos rotulados MUST ser estratificados e aninhados; cada fração MUST ser dividida internamente em 80% para ajuste e 20% para validação, e o orçamento declarado MUST incluir ambos os subconjuntos. Após a seleção, o classificador final MUST ser reajustado com 100% da mesma fração antes do teste.
- **FR-019**: O regime de 1% MUST comparar `X-only`, `H-only` e `H‖X_tx` com o mesmo classificador, IDs rotulados e sementes.
- **FR-020**: O regime de 1% MUST comparar o método completo com, no mínimo, variantes sem KNN, sem edge dropout e com três políticas de mascaramento sob a mesma intensidade e orçamento: atributos individuais aleatórios, grupos aleatórios de mesma cardinalidade dos blocos funcionais e blocos funcionais.
- **FR-021**: A matriz principal de 1%, 5%, 10% e 100% MUST incluir MLP `X-only`, Random Forest, XGBoost, GCN, GraphSAGE, GIN supervisionado, Inspection-L/DGI e GCPAL. A adaptação Inspection-L MUST preservar GIN 2×128, DGI e Random Forest de 100 árvores; a adaptação GCPAL MUST preservar GIN 2×128, suas duas visões estocásticas, KNN com `K=10`, perda multi-positivo e MLP de duas camadas sobre `H‖X_tx`. Ambos MUST usar os mesmos 182 atributos, splits, budgets e seeds do S003, e toda diferença causada pela migração de dataset MUST ser declarada.
- **FR-022**: Todos os métodos comparáveis MUST usar os mesmos splits, sementes, subconjuntos rotulados, alvo e política de avaliação.
- **FR-023**: F1 da classe ilícita MUST controlar seleção de hiperparâmetros, early stopping e threshold dos componentes supervisionados na validação interna; MCC MUST ser o primeiro critério de desempate. O encoder SSL MUST usar um número fixo de épocas, congelado após o dry-run e antes de qualquer avaliação em 35–49, sem probe rotulado nem seleção de checkpoint por métricas downstream. F1 ilícito e MCC MUST ser métricas primárias no relatório final. Precision, Recall e PR-AUC da classe ilícita MUST ser reportadas; métricas secundárias devem ser identificadas como tais.
- **FR-024**: O relatório MUST calcular as métricas agregadas concatenando as predições conhecidas dos snapshots 35–49 antes do cálculo; em particular, o F1 principal é o F1 da classe ilícita pooled e MUST NOT ser denominado micro-F1. O relatório também MUST apresentar média e desvio padrão entre as cinco sementes, além de métricas e suportes por snapshot em 35–49.
- **FR-025**: Alignment, uniformity e effective rank MUST ser diagnósticos comparativos de representação e MUST NOT substituir as métricas downstream nem usar um threshold universal não validado.
- **FR-026**: Cada execução MUST registrar dados, configuração, revisão, ambiente, semente, IDs amostrados, partições, duração, estado e artefatos produzidos.
- **FR-027**: O pipeline MUST distinguir execução concluída, interrompida, inválida e falha, sem converter estados incompletos em sucesso.
- **FR-028**: Um dry-run de uma semente MUST validar o pipeline completo, a política temporal e os limites declarados de recursos antes da liberação da matriz científica.
- **FR-029**: A execução da matriz completa MUST exigir aceite explícito das evidências do dry-run.
- **FR-030**: A extensão `Addr↔Tx` MUST permanecer fora do caminho crítico e somente poderá entrar na matriz após aprovação documentada dos gates de causalidade, supervisão, comparabilidade e recursos.
- **FR-031**: Se executada, a extensão heterogênea MUST classificar as mesmas transações do núcleo e MUST NOT usar atributos ou rótulos de endereço derivados de eventos posteriores ao snapshot.
- **FR-032**: Falha no gate heterogêneo MUST produzir uma decisão de adiamento rastreável, sem alterar critérios de conclusão do núcleo homogêneo.
- **FR-033**: O Estudo 003 MUST manter configurações, código específico, testes, documentos e artefatos em seus namespaces dedicados e MUST NOT alterar artefatos congelados do S02.
- **FR-034**: Reutilização de um componente histórico MUST ser explícita, testada quanto ao contrato S003 e registrada como decisão antes de ser incorporada.
- **FR-035**: O estudo MUST considerar cientificamente válido um resultado sem ganho sobre os baselines, desde que o protocolo e as evidências estejam completos.
- **FR-036**: Para as comparações predefinidas do método completo contra GCPAL e Inspection-L em 1% e 5%, o relatório MUST apresentar diferenças pareadas por semente, intervalo de confiança de 95%, teste t pareado bilateral e tamanho de efeito. O plano MUST definir antes da avaliação final a correção aplicada à família dessas quatro comparações, e a conclusão MUST considerar magnitude, incerteza e consistência do efeito, sem usar `p < 0,05` isoladamente como critério de superioridade.
- **FR-037**: Todas as GNNs da matriz principal MUST usar somente as arestas dirigidas `Tx→Tx` originais. Uma ablação P2 em 1% MAY adicionar explicitamente as arestas reversas ao S003-TxGCL, somente se aprovada no gate de recursos; ela MUST ser identificada como bidirecional e MUST NOT substituir o resultado principal dirigido.
- **FR-038**: Smoke e dry-run MUST usar um shadow test contido em 1–34 e MUST NOT calcular métricas com rótulos de 35–49. O acesso aos rótulos 35–49 MUST ocorrer somente após seleção e threshold congelados, ser auditado por run e permitir apenas retomada ou rerun técnico com os mesmos hashes, pesos e threshold.
- **FR-039**: A preparação MUST produzir uma auditoria versionada de disponibilidade causal para cada grupo de features. Evidência de qualquer atributo calculado com eventos posteriores ao snapshot MUST interromper o trabalho antes do treinamento e exigir decisão do pesquisador.
- **FR-040**: O método completo MUST produzir duas representações aumentadas por âncora — estocástica e por blocos — e otimizar uma perda multi-positivo simétrica entre elas, com pesos iguais. A visão KNN MUST somente ampliar o conjunto positivo; negativos MUST ser os demais nós elegíveis do mesmo snapshot/batch após deduplicação e exclusão integral dos positivos.
- **FR-041**: Métodos supervisionados MUST receber os mesmos IDs e uma política de desbalanceamento calculada somente no fit: pesos inversos por classe para perdas neurais, `class_weight` equivalente para RF e `scale_pos_weight` equivalente para XGBoost. Desvios exigidos por uma reprodução MUST ser declarados.
- **FR-042**: Reprodutibilidade MUST exigir igualdade exata de hashes, IDs, splits e configs. Na mesma plataforma, scores neurais recarregados MUST respeitar `rtol=1e-5`, `atol=2e-6` e produzir as mesmas classes no threshold congelado; diferenças entre plataformas MUST ser quantificadas, não ocultadas.
- **FR-043**: IDs duplicados ou arestas com referência ausente/time step divergente MUST invalidar a preparação. Arestas duplicadas e self-loops de entrada MUST ser removidos e contabilizados. Snapshots sem arestas e nós isolados MUST permanecer válidos; KNN MUST usar `min(K,N-1)`. Âncoras sem negativo elegível MUST ser excluídas da perda e uma época sem âncora válida MUST invalidar a run.
- **FR-044**: Uma comparação inferencial primária MUST exigir os cinco pares de seeds completos e métricas definidas. Na falta de qualquer par, o relatório MUST mostrar os valores disponíveis de forma descritiva, marcar a comparação incompleta e MUST NOT calcular p-value, IC, tamanho de efeito ou alegar superioridade.
- **FR-045**: “Superior em uma métrica e budget” MUST exigir diferença média positiva, IC 95% da diferença acima de zero, `p` ajustado por Holm abaixo de 0,05 e direção positiva em ao menos quatro das cinco seeds, sem degradação estatisticamente significativa na outra métrica primária. “Eficiente em rótulos” somente poderá ser usado para uma comparação que satisfaça esse critério em 1% ou 5%. Robustez temporal MUST ser descrita por resultados por snapshot, sem rótulo categórico predefinido. “Publicável” MUST NOT ser tratado como resultado mensurável do experimento.
- **FR-046**: Falha do `doctor` em qualquer versão ou capacidade obrigatória MUST bloquear dry-run e matriz naquele ambiente; o sistema MUST NOT trocar dispositivo, versão ou método silenciosamente.

### Key Entities

- **Transaction**: Unidade de predição; possui identificador externo, time step como metadado, classe conhecida ou desconhecida, 182 features financeiras de modelo e relações de fluxo com outras transações.
- **Temporal Snapshot**: Recorte de um único time step contendo somente transações, atributos e arestas permitidos naquele instante.
- **Label Budget**: Subconjunto estratificado e aninhado de transações conhecidas em 1%, 5%, 10% ou 100%, incluindo ajuste e validação interna.
- **Contrastive View**: Transformação sem rótulos de um snapshot usada para obter representações comparáveis da mesma âncora.
- **Positive Set**: Para uma âncora, união controlada do próprio nó, vizinhos de fluxo e vizinhos KNN; é disjunto do conjunto negativo correspondente.
- **Experiment Configuration**: Declara identidade do estudo, topologia, método, fração, semente, política temporal, seleção, métricas e limites de recursos.
- **Experiment Run**: Tentativa imutavelmente identificada de executar uma configuração, com estado, proveniência, duração e artefatos.
- **Evaluation Result**: Métricas de uma execução para a classe ilícita, agregadas e por snapshot, vinculadas à configuração e aos manifests de origem.
- **Heterogeneous Extension Decision**: Registro das evidências dos gates e da decisão de incluir ou adiar `Addr↔Tx`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% das combinações P1 declaradas na matriz possuem resultado concluído ou falha explícita e rastreável; nenhuma combinação ausente é apresentada como concluída.
- **SC-002**: Auditorias automatizadas encontram zero uso de dados, rótulos, estatísticas ajustadas ou decisões provenientes dos passos 35–49 antes da avaliação final.
- **SC-003**: Para uma mesma configuração e semente, execuções repetidas produzem os mesmos hashes de dados, splits, subconjuntos rotulados e conjuntos de comparação.
- **SC-004**: O relatório final apresenta, para cada método e orçamento principal, cobertura das sementes `[11, 23, 37, 53, 71]`, métricas pooled sobre as predições concatenadas de 35–49, média, desvio padrão e resultados nos 15 snapshots de teste; nas quatro comparações primárias predefinidas, também apresenta diferenças pareadas, intervalo de confiança de 95%, teste t pareado bilateral, tamanho de efeito e correção por comparações múltiplas.
- **SC-005**: No regime de 1%, o relatório separa quantitativamente o efeito de `X`, `H`, `H‖X_tx`, KNN, edge dropout e das políticas de mascaramento individual aleatório, por grupos aleatórios de mesma cardinalidade e por blocos funcionais.
- **SC-006**: Uma linha de resultado escolhida arbitrariamente pode ser rastreada até todos os seus insumos e decisões por um identificador único, sem consultar estado não versionado do S02.
- **SC-007**: O dry-run percorre preparação, pré-treino, downstream, seleção, inferência e relatório dentro dos limites de recursos declarados antes de qualquer matriz completa ser autorizada.
- **SC-008**: A decisão sobre `Addr↔Tx` registra o resultado de 100% dos gates definidos e não impede a conclusão do núcleo quando algum gate falha.
- **SC-009**: Nenhum arquivo congelado do S02 é modificado para implementar ou executar o S003.
- **SC-010**: A conclusão científica distingue claramente protocolo concluído de hipótese confirmada e permanece válida mesmo se o método proposto não superar os baselines.

## Assumptions

- Os nove CSVs originais do Elliptic++ já disponíveis localmente continuam sendo a fonte de dados autorizada e permanecerão somente leitura.
- As 183 colunas não identificadoras estão disponíveis no instante do respectivo time step conforme a auditoria consolidada; apenas as 182 features financeiras integram `X_tx`, enquanto `Time step` permanece metadado. Qualquer evidência contrária interrompe o trabalho e exige decisão do pesquisador.
- O NotebookLM autorizado contém as fontes de literatura necessárias; consultas futuras usarão somente o notebook registrado no contexto S003.
- Os valores concretos de hiperparâmetros e o orçamento de busca serão definidos no plano, registrados antes do teste e escolhidos sem acessar 35–49.
- A intensidade de mascaramento será mantida comparável entre os três controles; a regra determinística para formar grupos aleatórios por semente será definida no plano sem usar rótulos.
- A correção por comparações múltiplas e a medida de tamanho de efeito serão fixadas no plano antes de qualquer avaliação nos passos 35–49.
- A divisão interna de ajuste e validação preservará o orçamento total de rótulos declarado por meio do split estratificado 80/20 definido em FR-018.
- Fine-tuning end-to-end é uma análise secundária condicionada ao orçamento e não substitui o protocolo principal com encoder congelado.
- O diretório `artifacts/s003/` é local e não versionado; manifests e relatórios pequenos necessários à auditoria serão persistidos em locais versionados definidos no plano.
- O nome `S003-TxGCL` é um identificador interno e não antecipa o nome final de publicação.

## Scope Boundaries

### In Scope

- Classificação binária de transações nos snapshots `Tx→Tx` do Elliptic++.
- Pré-treino contrastivo sem rótulos, avaliação sob escassez e baselines necessários à atribuição de ganhos.
- Auditoria temporal, proveniência, retomada, dry-run e relatório científico.
- Avaliação condicionada da extensão heterogênea `Addr↔Tx`.

### Out of Scope

- Classificação de carteiras ou entidades como alvo principal.
- Alteração, reexecução ou reinterpretação dos resultados congelados do Estudo 002.
- Uso dos passos 35–49 para tuning, early stopping, normalização, calibração ou seleção.
- Alegação de que o método é o primeiro GCL para fraude ou o primeiro modelo heterogêneo de fraude.
- Identificação de usuários reais, atribuição legal de responsabilidade ou implantação em produção AML.
- Execução da matriz completa antes do aceite explícito do dry-run.

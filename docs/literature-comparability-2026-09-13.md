# Revisão focada de comparabilidade — H-GCL Elliptic++

Data: 2026-09-13. Revisão documental autorizada pelo pesquisador após discussão
do retorno do NotebookLM. Não constitui revisão sistemática, certificação de
ineditismo ou autorização para alterar a matriz. Nenhum experimento foi executado.

## Resultado e contribuição candidata

O desenho atual permite estudar o benefício incremental do SSL e da fusão na
classificação de endereços com poucos rótulos. Ainda não permite alegar superioridade
sobre o estado da arte. A novidade não pode se apoiar apenas em SSL, grafo bipartido,
classificação de endereços ou escassez de rótulos, pois há antecedentes próximos.

Formulação candidata, sem promessa de resultado:

> Investigar se o pré-treinamento contrastivo em grafos endereço–transação e a fusão
> com Random Forest melhoram a classificação de endereços sob escassez de rótulos,
> usando entradas restritas ao passo, avaliação temporal e resultados separados para
> endereços vistos e inéditos.

O complemento nativo compara políticas de entrada; não isola causalmente vazamento.
Melhoria, equivalência prática ou ausência de ganho exigem evidência. Cinco seeds
medem variabilidade entre execuções, não equivalem a cinco datasets independentes.
Se a alegação pretendida for equivalência com menos rótulos, definir margem e análise
antes do teste; a matriz atual não autoriza concluir equivalência pela proximidade de médias.

## Busca e critérios

Consultas web nesta sessão: `Elliptic++ illicit address classification graph learning
wallet 2025 2026`, `Elliptic++ wallet address classification self supervised contrastive
learning`, `Inspection-L self supervised graph neural networks money laundering
detection random forest`, nomes dos métodos combinados com `github`/`code`.
Incluídos trabalhos próximos por alvo, representação ou objetivo SSL e fontes originais.
Foram lidos trechos de método/experimentos dos artigos abaixo. Resultados publicados
não foram reproduzidos; repositórios não foram executados nem tiveram licenças auditadas.
Ausência de código localizado nesta busca não significa inexistência de código.

## Comparabilidade científica

| Trabalho e fonte primária | Alvo, dados e entradas | Tempo e rótulos | Uso no nosso estudo |
| --- | --- | --- | --- |
| [Elliptic++, Elmougy e Liu (2023)](https://arxiv.org/html/2306.06108v1), §§3.2, 4.1–4.2 | Transações e endereços; atributos nativos; RF, LR, MLP, LSTM e XGB | Treino 1–34, teste 35–49; não explicita janela de validação separada em §4.1 | Referência do dataset e dos baselines tabulares. Nosso RF usa outra política de entradas e configuração; não é reprodução do número publicado. |
| [Inspection-L, Lo et al. (arXiv v4, 2022)](https://arxiv.org/html/2203.10465v4), §§4.1–4.2, 5.3 | Transações Elliptic; GIN/DGI, embeddings e atributos locais ou completos para RF | SSL em 34 grafos; classificação nos 15 restantes; desconhecidos excluídos da supervisão | Precedente de SSL + RF. Não é nossa fusão de scores; adaptar para endereços exige explicitar mudanças. |
| [GCPAL, Lu et al. (2024)](https://link.springer.com/article/10.1007/s44196-024-00720-4), §§4–5.1 | Transações Elliptic, 166 atributos e variantes LF/AF; também AMLworld; pré-treinamento contrastivo e classificador MLP | Frações supervisionadas 1/2/5/10/20%, cinco seeds. Corte temporal exato não estabelecido no texto consultado | Antecedente direto da pergunta sobre eficiência de rótulos, mas não do mesmo alvo. O link de disponibilidade aponta dados, não comprova código GCPAL. |
| [HeteroGCL, Chen et al. (2026)](https://www.mdpi.com/2076-3417/16/6/2860), §§4.1, 5.1 | Transações Elliptic; nós de transação e agrupamento temporal; 166 atributos | Treino 1–34, validação 35–43, teste 44–49; avalia rótulos limitados | Relacionado em SSL heterogêneo. Inclusão de endereços é indicada como trabalho futuro. Não comparar seu F1 diretamente com nosso alvo e janela. |
| [LaundroGraph (2022)](https://arxiv.org/html/2210.14360), §§3, 4.1 | Grafo cliente–transação bancário; SSL por predição de ligação; 66 atributos de cliente e 12 de transação | Dados bancários de identidade não divulgada; grafo de seis meses, teste no seguinte; rótulos da tarefa são ligações/não ligações | Precedente de SSL bipartido em AML, não avaliação de ilicitude no Elliptic++. Inclui DGI por tipo como comparador. |
| [Ichull e Sopuru (2026), preprint v1](https://www.preprints.org/manuscript/202606.1770), método e Dataset Configuration | Declara classificação de transações e carteiras no Elliptic++; múltiplos grafos e atenção; seleção de 183 para 103 atributos de transação | Treino 1–35, validação 36–42, teste 43–49; relata avaliação com 10% dos rótulos | Sobreposição de alvo e escassez; não revisado por pares nesta versão. A unidade/contagens e semântica das classes exigem esclarecimento antes de reprodução. |
| [ETDNet, Full-History Graphs (2026)](https://journals.sagepub.com/doi/10.3233/FAIA251186), §4.2 | Denomina dados Elliptic++, mas define alvo transação e 94 atributos; grafo com histórico | Treino 1–30, validação 31–40, teste 41–49; três seeds | Contextualiza modelagem temporal. Acesso ao histórico e alvo diferem; não é comparador direto do protocolo por passos independentes. |

## Ressalvas verificáveis nas fontes

No preprint de Ichull e Sopuru, a tabela 7 chama a classe 3 de lícita, enquanto o
dataset original a define como desconhecida. A configuração informa 1.268.260
carteiras, número correspondente às ocorrências temporais no artigo original, e
2.661 carteiras ilícitas no total. Essas diferenças precisam de explicação; não se
conclui daqui que o código fez a mesma classificação incorreta, pois não foi verificado.
O trabalho é pertinente ao estado da arte, mas seu número não é alvo confiável de
comparação direta sem resolver população e protocolo.

No GCPAL, a descrição experimental é de transações Elliptic mesmo com um link ao
repositório Elliptic++. Não inferir avaliação de endereços a partir desse link.
No ETDNet, conservar na nota a descrição dos autores sem adotar como fatos do nosso
dataset a periodicidade mensal ou a interpretação de atributos anonimados.

## Código e viabilidade de reprodução

| Referência | Evidência disponível nesta revisão | Esforço/condição para uso |
| --- | --- | --- |
| Elliptic++ | [Repositório dos autores com tutoriais](https://github.com/git-disl/EllipticPlusPlus) | Reproduzir tutorial exige conferir versão, entradas e seleção; aplicar RF/XGB ao nosso contrato é comparação adaptada. |
| Inspection-L | Método e parâmetros no artigo; implementação oficial não confirmada na busca | DGI + RF adaptado exige objetivo por tipo e tratamento dos atributos; reutilizar nosso InfoNCE e chamar de Inspection-L seria incorreto. |
| GCPAL | Texto e links de datasets; código oficial não confirmado | Reimplementação de múltiplas vistas/KNN e definição dos cortes; custo e fidelidade ainda não estabelecidos. |
| HeteroGCL | Método e protocolo publicados; código oficial não confirmado | HGAT, relações adicionais e objetivo próprios; conversão para endereços seria outra adaptação substancial. |
| LaundroGraph | Algoritmo e avaliação descritos; dados bancários não identificados | Reprodução exata dos resultados não disponível com nossos dados; adaptação da tarefa de ligação demanda novo desenho. |
| Preprint multigrafo | Sem link GitHub localizado no texto/busca focada | Antes de implementar, esclarecer classes, população, máscaras e obter implementação verificável. |
| ETDNet | Artigo; repositório específico não confirmado | Construção histórica contradiz a restrição atual de passos independentes; só avaliar em extensão expressamente definida. |

Custos acima são estimativas qualitativas de implementação, não medições de GPU.
Nenhum candidato foi validado na RTX 2060.

## Correções ao retorno do NotebookLM

- Diferenças em relação ao HeteroGCL são sustentadas; ineditismo e publicação não
  estão garantidos. Trabalho futuro em um artigo não certifica lacuna em toda a literatura.
- Nosso H-GCL usa atributos e topologia. A fusão não o torna imune à evasão nem
  remove vazamento de atributos; controles de entrada e partições são mecanismos separados.
- `src/hgcl/models/augmentations.py` aplica mascaramento aleatório de valores e
  dropout pareado com reversas computacionais. Não implementa semantic block masking.
- As vistas aumentadas não representam necessariamente livros contábeis válidos.
  Arestas não recuperam alocações completas de UTXOs.
- O alvo é endereço–passo com classe global; não identidade civil, entidade inteira
  ou prova de crime. Nenhum teste adversário faz parte do escopo atual.

## Recomendação para decisão do pesquisador

Manter os quatro métodos atuais como núcleo de análise controlada. Para fortalecer
um artigo empírico, proponho duas extensões principais, ainda NÃO autorizadas:

1. **XGBoost tabular**, mesmas 123 entradas, máscaras, seleção e população. Testa se
   o ganho depende de comparar apenas contra RF. Exige dependência/lock, busca
   delimitada e testes de integração. É baseline adaptado, não reprodução de um artigo.
2. **DGI por tipo + mesmo fine-tuning**, mantendo encoder e cabeça do controle.
   Testa o objetivo contrastivo escolhido contra outro SSL. É inspirado em
   Inspection-L e no comparador DGI de LaundroGraph, não reprodução de Inspection-L.
   Exige objetivo, corrupção, resumo por tipo e regras de lotes especificados antes
   da implementação. Não é objetivo idêntico ao RF sobre embeddings congelados.

Escopo sugerido das extensões: somente cenário principal, quatro frações e cinco
seeds. Cada método acrescentaria 20 avaliações; ambos levariam de 100 para **140**,
preservando as 20 avaliações nativas existentes. DGI exigiria cinco novos caches
SSL se compartilhado entre frações. Contagens de configurações candidatas, memória
e tempo só poderão ser fixadas após desenho e probe; não extrapolar o smoke.

Alternativa de menor escopo: executar as 100 avaliações atuais e formular a conclusão
como estudo dos componentes, sem alegação de estado da arte. Para uma alegação de
superioridade sobre métodos recentes específicos, essas duas extensões ainda não
bastam: selecionar e reproduzir/adaptar explicitamente tais métodos sob protocolo
comum. Não adicionar arquiteturas apenas porque são recentes ou têm F1 maior.

## Atualização D048 — 2026-09-14

O pesquisador decidiu manter as 100 avaliações da primeira rodada e adiar ambas
as extensões, conforme [planejamento](../specs/001-hgcl-experiment/deferred-extensions.md).
As recomendações acima são preservadas como histórico e não bloqueiam a T031.
Próximo passo: [protocolo operacional](../specs/001-hgcl-experiment/t031-execution-protocol.md).

## Ponto de decisão em 2026-09-13 (resolvido por D048)

Revisão focada entregue. Próxima decisão: manter a matriz atual ou autorizar o desenho
das duas extensões propostas. Após a decisão, atualizar spec/plan/tasks e contratos
somente no que for aprovado; verificar implementação; redigir o protocolo operacional
T031 com auditoria dos 49 passos e pausa após dry-run. Preparação integral e treinamento
não iniciados. A comparação documental não valida o ambiente nem altera seu aceite T030.

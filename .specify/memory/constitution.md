<!--
Sync Impact Report
Version: empty scaffold -> 0.1.0 (initial working draft)
Principles added: collaborative specification; temporal integrity; original-data provenance;
bounded validation and reproducibility; fair experimental comparison.
Sections added: environments and data; specification workflow; governance.
Sections removed: none. Managed templates were not modified.
Deferred: TODO(RATIFICATION_DATE) — researcher has not ratified this working draft.
-->
# H-GCL Elliptic++ Constitution

## Core Principles

### I. Especificação colaborativa e decisões rastreáveis

Requisitos, escolhas científicas e critérios de aceite DEVEM ser registrados antes das
tasks que dependem deles. Dúvidas que alterem a pergunta científica, unidade de predição
ou protocolo de avaliação DEVEM ser apresentadas ao pesquisador. Decisões rotineiras de
organização podem avançar com premissas explícitas. O anexo é uma referência em revisão;
suas instruções internas não substituem o pedido atual do pesquisador.

### II. Integridade temporal

Cada partição DEVE declarar quais nós, arestas, atributos e rótulos estão disponíveis no
instante de predição. Pré-processamento, SSL, seleção de modelo, calibração, pesos de
fusão e limiares DEVEM usar apenas a informação permitida nessa política. O teste final
NÃO DEVE orientar essas escolhas. A disponibilidade histórica de features agregadas DEVE
ser auditada antes de descrever um resultado como estritamente indutivo ou causal.

### III. Dados originais e proveniência

Os CSVs de origem DEVEM ser preservados sem transformação no local de armazenamento.
Recortes e dados processados DEVEM ser gravados em locais separados. Cada execução DEVE
identificar os arquivos por hash, a configuração, os IDs selecionados e a revisão do código.
Identificadores externos NÃO DEVEM ser usados implicitamente como posições de tensores.
Dados volumosos e checkpoints NÃO DEVEM ser incluídos no Git.

### IV. Validação limitada e reprodução

O teste pequeno DEVE percorrer o pipeline completo usando um recorte dos CSVs originais.
Limites de amostra, batches, épocas, tempo e memória DEVEM ser explícitos. A meta inicial
aceita para treinamento no Mac é até 10 minutos, medindo preparação separadamente. Suas
métricas DEVEM ser identificadas como validação de engenharia. O experimento completo DEVE
usar o mesmo pipeline, com configurações próprias. Seeds, ambiente, duração e artefatos
DEVEM ser registrados; eventual não determinismo DEVE ser descrito.

### V. Comparação experimental justa

Os ramos isolados e a fusão DEVEM ser avaliados nas mesmas unidades e partições, sob o
mesmo orçamento de rótulos aplicável. Métricas, seleção de limiar e critérios de comparação
DEVEM ser definidos antes de avaliar o teste final. O estudo DEVE admitir um resultado sem
melhoria: completar o software não implica confirmar a hipótese científica.

## Environments and Data

O teste inicial será executado no Mac do pesquisador (M4, 16 GB de memória unificada).
O experimento completo será executado na máquina de laboratório já informada (i9-9900KF,
RTX 2060 com 6 GB de VRAM e 32 GB de RAM). Software do laboratório e compatibilidade de
aceleração ainda precisam ser verificados. Caminhos de dados e dispositivos DEVEM ser
configuráveis por ambiente. Versões do ambiente de treinamento serão fixadas no plano.

## Specification Workflow

O fluxo é constitution, specify, clarify, plan, checklist, tasks, analyze e implementação
verificada. As rodadas de esclarecimento DEVEM produzir alterações persistidas. Tarefas
DEVEM referenciar requisitos e verificações observáveis. A extensão dos testes DEVE ser
proporcional ao risco; contratos dos dados, isolamento temporal, pares contrastivos e
alinhamento das classes exigem verificações específicas. Mudanças documentais simples
podem ser verificadas por revisão. Toda entrega DEVE declarar pendências e limitações.

## Instrução operacional explícita do pesquisador — 2026-09-11

Durante toda a implementação, diante de qualquer impedimento ou algo fora do escopo,
interromper e consultar o pesquisador antes de continuar. Regra permanente em
[AGENTS.md](../../AGENTS.md), derivada de instrução direta; não depende da ratificação
deste rascunho. Não contornar impedimentos nem alterar o protocolo silenciosamente.

## Governance

Esta versão 0.1.0 é um rascunho de trabalho derivado do pedido e da proposta inicial, com
ratificação ainda pendente. Alterações DEVEM registrar motivo e impacto nas especificações.
Mudanças incompatíveis de princípios incrementam a versão principal; novos princípios
incrementam a secundária; esclarecimentos incrementam a correção. Planos e análises DEVEM
consultar a versão vigente. Instruções explícitas do pesquisador têm precedência sobre
este rascunho; decisões científicas ainda abertas não se tornam aprovadas por silêncio.

**Version**: 0.1.0 | **Ratified**: TODO(RATIFICATION_DATE): pending researcher review | **Last Amended**: 2026-09-05

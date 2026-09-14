# Handoff de contexto — experimento H-GCL com Elliptic++

Data: 2026-09-11
Projeto: `/Users/stan/Projects/masters-degree/hgcl-elliptic`
Repositório remoto: `3011stan/elliptic_plus_plus_hgcl`

## Como usar este documento

Este arquivo registra o estado necessário para continuar o trabalho em um novo chat. Antes de agir, o novo agente deve conferir o estado atual do Git e ler os documentos normativos apontados abaixo. Não deve presumir que uma etapa foi executada apenas porque estava planejada.

## Instrução permanente do pesquisador

Se surgir qualquer impedimento, resultado inesperado ou necessidade de sair do escopo definido, interromper a execução e consultar o pesquisador. Não corrigir, ampliar o escopo ou alterar decisões metodológicas silenciosamente. Essa regra está registrada em `AGENTS.md` e vale durante toda a implementação.

O pesquisador prefere interações objetivas e economia de tokens. Para saídas extensas, solicitar ou ler artefatos JSON e extrair somente os campos necessários. Não colar logs integrais sem necessidade.

## Objetivo científico

Construir e avaliar um experimento sobre o Elliptic++ comparando:

1. Random Forest tabular.
2. Encoder de grafos supervisionado treinado do zero.
3. O mesmo encoder inicializado por pré-treinamento auto-supervisionado H-GCL e posteriormente ajustado com rótulos.
4. Fusão tardia dos scores da Random Forest e do H-GCL.

A contribuição principal investiga se o aprendizado auto-supervisionado e a fusão ajudam sob escassez de rótulos, sem tentar reproduzir integralmente outro artigo ou reinventar baselines já estabelecidos.

## Unidade de predição e interpretação

- Unidade: par `(address, timestep)`.
- O rótulo disponível em `wallets_classes.csv` é global por endereço.
- O rótulo não informa quando a atividade ilícita começou nem quando foi descoberta.
- Portanto, o experimento mede se o modelo reconhece a classe global usando a observação do endereço naquele passo.
- O modelo pode classificar endereços inéditos. O teste mantém endereços recorrentes e inéditos, e os resultados devem ser separados nesses estratos.
- Não alegar que o experimento demonstra detecção antes do início do crime, pois o dataset não contém esse marco temporal.

## Protocolo temporal

- Treino: passos 1–28.
- Validação: passos 29–34.
- Teste: passos 35–49.
- Cada passo é um grafo bipartido endereço–transação independente.
- Não há adjacência, estado oculto ou grafo cumulativo entre passos.
- Um único encoder, com pesos compartilhados, é aplicado aos grafos; não são 49 modelos.
- Teste não participa de pré-processamento, SSL, treino, seleção, calibração, escolha de limiar ou fusão.
- Validação é usada para inferência e seleção, sem gradientes do encoder.

## Entradas do cenário principal

Nós de transação:

- 15 propriedades explícitas, incluindo BTC, taxas e tamanho.
- 15 indicadores de ausência.
- 2 contagens de graus recalculadas no grafo do passo.
- Total: 32 entradas.

Nós de endereço e Random Forest:

- 3 contagens de atividade.
- Para cada uma das 15 propriedades e para os papéis remetente/destinatário: `sum`, `mean`, `max` e fração observada.
- Total: 123 entradas.
- Esses valores resumem o contexto das transações incidentes no próprio passo.
- O valor integral de uma transação com múltiplos participantes não é interpretado como montante efetivamente enviado ou recebido por cada endereço, pois as arestas fornecidas não contêm a alocação por endereço/UTXO.

IDs, timestep e classes não são atributos preditivos. Imputação, `log1p` e estatísticas de normalização são ajustadas somente no treino.

## Complemento com atributos nativos

- Quatro métodos, apenas com 100% dos rótulos, cinco seeds: 20 avaliações.
- Substitui os 123 atributos de endereço por 55 atributos nativos mais 55 indicadores de ausência: 110 entradas.
- Os 32 atributos das transações e a topologia permanecem iguais.
- O horizonte histórico de todos os atributos nativos não está certificado; resultados devem declarar essa limitação.
- Esse complemento mede uma mudança de política de entradas, não um efeito causal puro de vazamento.

## Matriz experimental

Cenário principal:

- Frações: 1%, 5%, 10% e 100%.
- Métodos: 4.
- Seeds: 11, 23, 37, 53 e 71.
- Total: 80 avaliações.

Complemento nativo:

- Fração: 100%.
- Métodos: 4.
- Seeds: 5.
- Total: 20 avaliações.

Total geral: 100 avaliações, organizadas em 25 grupos, com 10 caches SSL e 150 configurações candidatas previstas.

Amostragem de rótulos:

- Por endereços únicos conhecidos, não por ocorrências endereço–tempo.
- Estratificada por classe, determinística, sem reposição e aninhada entre frações.
- Mesmas máscaras para métodos comparados.
- Validação possui orçamento separado com a mesma porcentagem sobre sua população.
- Nenhuma ampliação silenciosa por falta de exemplos.

## Treinamento e avaliação

- H-GCL: SSL sem rótulos somente nos passos de treino; depois descarte da cabeça contrastiva e ajuste do encoder e da cabeça supervisionada.
- Controle supervisionado: mesma arquitetura, cabeça, máscaras, busca e orçamento; inicialização do encoder do zero.
- Cache SSL separado por regime e seed e reutilizado entre frações, sem mutação.
- Random Forest usa os descritores de endereço correspondentes ao regime.
- Fusão: `alpha * score_RF + (1-alpha) * score_HGCL`, com `alpha` em passos de 0,05.
- Checkpoint selecionado por AP na validação.
- Limiar e alpha escolhidos por F1 ilícito na validação e congelados antes do teste.
- Métrica principal: F1 da classe ilícita por par endereço–tempo.
- Complementares: precisão, recall, AP e MCC.
- Reportar suporte, visto/inédito, exposição prévia ao rótulo, média, desvio-padrão amostral e diferenças pareadas por seed.
- Ausência de melhoria é um resultado válido.

## Estado da implementação

- T001–T030 estão marcadas como concluídas.
- T026–T029 implementaram e validaram a orquestração da matriz, retomada e relatórios.
- Regressão registrada após a correção de recarga: 63 testes aprovados e 2 opcionais ignorados.
- Smoke CPU com dados originais passou.
- T030, ambiente do laboratório, foi concluída.
- T031–T033 permanecem pendentes.

Há alterações documentais locais posteriores que podem ainda não estar commitadas. Conferir `git status --short --branch` antes de qualquer edição ou execução.

## Ambiente do laboratório validado

Máquina:

- NixOS 26.05, x86_64.
- Python 3.11.16.
- NVIDIA GeForce RTX 2060, 6 GB.
- Driver 595.71.05.
- Torch 2.6.0+cu124.
- PyTorch Geometric 2.6.1.
- `pyg-lib` 0.4.0+pt26cu124.
- Aproximadamente 62 GiB de RAM e 123 GB livres na medição inicial.

`doctor` remoto: PASS, incluindo CUDA, amostragem bipartida, forward/backward e hashes dos nove CSVs.

Locks gerados no laboratório e enviados ao GitHub:

- `flake.lock`.
- `requirements/lab-cuda.lock`.
- Commit remoto mencionado: `96455c1` (`fix: Register env nixos and cuda dependencies`).

Smoke GPU final: `gpu-smoke-002`, PASS.

Verificação de recarga:

- Quatro métodos dentro de `rtol=1e-5`, `atol=2e-6`.
- Diferença absoluta máxima abaixo de `6e-8`.
- Zero divergências nas decisões binárias.

O primeiro smoke GPU falhou somente porque a igualdade numérica exigida era estrita demais. A correção autorizada mantém tolerância float32 e exige zero divergências de decisão.

## Arquivos normativos principais

- `AGENTS.md`: regra permanente de parada e consulta.
- `specs/001-hgcl-experiment/spec.md`: requisitos científicos e funcionais.
- `specs/001-hgcl-experiment/input-contract.md`: identidade, grafos, atributos e pré-processamento.
- `specs/001-hgcl-experiment/training-design.md`: treinamento, SSL, controle e fusão.
- `specs/001-hgcl-experiment/contracts/run-config.md`: valores fixos dos perfis.
- `specs/001-hgcl-experiment/contracts/cli.md`: ciclo dos comandos.
- `specs/001-hgcl-experiment/tasks.md`: estado das tasks.
- `docs/validation/lab-preflight.md`: evidências do laboratório.
- `docs/validation/matrix-readiness.md`: validação da implementação da matriz.

## Ponto exato onde a conversa parou

O pesquisador questionou se a linha atual da T031 é detalhada o bastante:

> Run matrix dry-run and then the laboratory experiment via `configs/lab.yaml`, retaining `artifacts/matrices/s02-001/` and a portable summary in `docs/validation/experiment.md`; do not mark complete for dry-run alone or without all required results.

Conclusão: essa linha funciona como índice e herda requisitos dos documentos normativos, mas não é suficiente como protocolo autônomo para uma execução científica longa. Ainda não foi autorizada alteração documental nesse ponto nem a preparação integral dos 49 passos.

## Próximo passo recomendado

Antes de executar a preparação integral, detalhar e submeter ao pesquisador um protocolo da T031, preferencialmente em `specs/001-hgcl-experiment/t031-execution-protocol.md`, e fazer a task referenciá-lo. O protocolo deve conter:

1. Congelamento do commit, árvore Git limpa, dados, configuração e locks.
2. Preparação e validação dos 49 grafos.
3. Auditoria das dimensões, partições, alvos, máscaras, recorrência, ausência e recursos.
4. Materialização e validação dos orçamentos de rótulos.
5. Dry-run explícito sobre os dados preparados.
6. Pausa obrigatória para o pesquisador revisar o dry-run antes do treinamento.
7. Execução serial e retomável dos 25 grupos/100 avaliações.
8. Política de falha sem alterações silenciosas.
9. Critérios objetivos de completude e criação do relatório portátil.

Duas lacunas concretas precisam ser resolvidas na especificação/fluxo:

- `scripts/lab/run.py ... matrix` executa um dry-run interno e começa a matriz imediatamente; não oferece a pausa humana desejada entre as duas etapas. O dry-run deve ser executado explicitamente e revisado antes do comando da matriz.
- O comando de relatório gera artefatos dentro da matriz, mas a T031 também exige `docs/validation/experiment.md`; a criação desse resumo portátil precisa estar expressamente definida.

Os pontos de controle planejados devem ser:

1. Preparação completa dos 49 passos → apresentar recibo e auditoria.
2. Dry-run com dados preparados → apresentar plano e aguardar aprovação.
3. Matriz completa → gerar relatório e verificar completude.

Não iniciar automaticamente a preparação nem a matriz ao retomar o novo chat. Primeiro revisar/persistir a especificação detalhada da T031 com o pesquisador.

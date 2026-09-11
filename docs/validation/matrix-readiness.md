# T026–T029 — verificação concluída

D039 autoriza implementação dos controles de acesso, matriz, retomada e relatórios.
Checklist: 16 verificados, 0 pendentes. Sem hooks de extensão configurados.

## Teste de isolamento da T026

Comando: .venv/bin/python -m pytest tests/contract/test_evaluation_access.py -q
Resultado: 3 aprovados, 1 falha, 2,06 segundos. Fixture artificial com quatro passos,
sem leitura ou treinamento dos originais nesta investigação.

Falha: test_fit_does_not_materialize_global_or_test_targets.
Caminho reproduzido: training/runner.py fit -> pipeline.py validate -> leitura de
prepared/targets/global.parquet. Esse arquivo contém rótulos globais de todas as partições.
A leitura foi interceptada antes de materializar os valores e antes de iniciar modelos.

A rotina usa esses rótulos para conferir budgets/máscaras, não para otimizar modelos.
O achado não demonstra uso de rótulos de teste em gradientes ou seleção no smoke anterior,
mas a fronteira de acesso de fit não atende ao contrato mais estrito da T026.

Correção proposta: separar validação estrutural/hash e validação semântica de rótulos;
fit e matrix dry-run só executam a primeira. Validar alvos permitidos por partição nas
etapas apropriadas; manter validação completa no comando explícito de preparação/validate.
Acrescentar testes que bloqueiem leitura de rótulos globais/teste durante fit e dry-run.

Ponto de parada anterior resolvido sob D040: correção aplicada e testes de acesso aprovados.


## Histórico: retomada sob D040 e nova pausa

Implementados, ainda sem aceite final: matriz de 100 avaliações em 25 grupos e dez
caches SSL; CLI matrix/resume/report; retomada com histórico e rejeição de configuração
alterada/refit após teste; relatórios por estrato, média/desvio amostral e diferenças pareadas.
Doze testes específicos passaram (4,33 s), incluindo treinamento/retomada em fixture.
CLI matrix --config configs/lab.yaml --dry-run terminou com exit 0: somente enumeração,
sem validação dos dados integrais, sem treinamento ou acesso aos rótulos de teste.

Regressão `.venv/bin/python -m pytest -q`: 54 aprovados, 2 opcionais ignorados,
1 falha (4,17 s). Falhou test_serial_restart_reuses_groups_and_ten_caches.
A fixture construiu Config diretamente com artifacts_root relativo (`artifacts`),
em vez de um caminho absoluto dentro de tmp_path. Ela gravou resultados simulados
em artifacts/matrices/fixture e artifacts/runs/fixture-*, reaproveitados pela segunda
execução com outro project_root temporário. O controle de identidade então rejeitou
corretamente a retomada incompatível. O uso normal por load_config resolve os caminhos.

Correção proposta: caminhos absolutos temporários na fixture; verificar repetibilidade
do teste; remover somente os artefatos simulados identificados, após autorização.
Nenhum CSV original ou resultado smoke foi sobrescrito. O smoke com originais desta
etapa ainda não foi repetido. T026–T029 permanecem sem aceite final. Execução interrompida
conforme D034; nova correção não aplicada, aguardando consulta.


## Conclusão após autorização D041

Fixture corrigida para gravar exclusivamente em caminhos absolutos de tmp_path.
Antes da remoção, os 25 runs simulados foram identificados pelo manifesto da matriz,
proveniência fictícia (preparation=p/payload=d) e métricas vazias. Removidos somente esses
runs e artifacts/matrices/fixture. smoke-001 preservado. Nenhum original modificado.

Verificação:
- Teste de orquestração isolado: 2 aprovados em 2,62 s; repetido na suíte completa.
- Regressão: 55 aprovados, 2 opcionais ignorados, nenhuma falha, 4,39 s.
- Smoke original separado: 1 aprovado em 9,24 s; run smoke-002, reload conferido.
- Treinamento + seleção: 2,226 s; avaliação final: 0,116 s, contabilizada separadamente.
- Dry-run CLI lab: exit 0, 100 chaves (80+20), 25 grupos, 10 caches SSL previstos,
  150 configurações candidatas de ajuste supervisionado/RF. Esses números são de
  planejamento; não representam execuções científicas realizadas ou épocas reexecutadas.

T026: acesso a rótulos isolado, recusa de backfill entre partições, chaves e estratos.
T027: DAG, execução serial, caches compartilhados e preservação de grupos incompletos.
T028: CLI e retomada compatível pré-teste, histórico e recusa de mudanças/refit pós-teste.
T029: suporte por estrato/exposição a rótulos, médias/desvio amostral, comparações
pareadas RQ1/RQ2, ausência de melhoria e resultados incompletos explicitados.

Evidências legíveis por máquina: matrix-results.json e matrix-plan.json.
Testes de orquestração usam treinamento simulado; teste de retomada e smoke exercitam
os modelos reais em CPU. A matriz científica não foi executada. Enumeração sem
--prepared não valida dados integrais. Preparação dos 49 passos, CUDA/sampler e lock
laboratorial permanecem para as etapas seguintes. A retomada não refaz ajuste após
predições de teste; falhas nesse estágio exigem inspeção, não retreinamento automático.

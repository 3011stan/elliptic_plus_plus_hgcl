# Estado do SDD — S02

## Aceite do dry-run registrado

Em 2026-09-14, o pesquisador aceitou iniciar a matriz `s02-001` com base no
dry-run remoto: `status=planned`, `data_integrity_checked=true`,
`training_performed=false`, 100 avaliações, 10 caches SSL e 150 configurações
candidatas. Identidades aceitas: source
`3b29d072fd51f2519df364b2ad0b4280b4edb88bb21841130f0ae807b0b96860`, preparation
`aa547c131b4c2a2ff376b9e2ca4bd19d8badccccea186f8196d276a9f027cc5b` e payload
`9c771e2d0cb27099508df4540a7ae886d99cfa4bca76bf1a42342259da149225`.
Próximo passo autorizado: executar a matriz serial no laboratório. T031 permanece
aberta até a completude das 100 avaliações e a geração do relatório.

## Preparação integral e auditoria remotas conferidas

Recibos/resumo enviados pelo pesquisador: prepare PASS em 47,800 s, pico RSS
6,075 GiB; audit/validation PASS, 1.279 arquivos e 20 orçamentos verificados,
49 passos por regime, 112.879.108.096 bytes livres em disco.
Source: `3b29d072fd51f2519df364b2ad0b4280b4edb88bb21841130f0ae807b0b96860`.
Preparation: `aa547c131b4c2a2ff376b9e2ca4bd19d8badccccea186f8196d276a9f027cc5b`.
Payload: `9c771e2d0cb27099508df4540a7ae886d99cfa4bca76bf1a42342259da149225`.
Identidades concordantes. O dry-run foi posteriormente executado e aceito; a
matriz ainda não foi iniciada. T031 permanece aberta.

Smoke remoto D050 conferido: `gpu-smoke-003` complete, identidade esperada, quatro
métodos, treino/seleção 5,782 s e recarga aprovada sem divergência de decisões.
Evidência: [lab-smoke.json](validation/lab-smoke.json). Próximo passo autorizado
pelo fluxo vigente: preparação integral e audit no laboratório; trazer os dois
recibos para revisão antes do dry-run. Sem repetição de doctor/smoke necessária.
T031 permanece aberta até a matriz científica completa.

## Diagnóstico D049 — correção local concluída sob D050

D050: exclusão de diretórios `*.egg-info` implementada em `source_identity`.
Regressão: 66 testes aprovados, 2 opcionais ignorados; metadados de instalação
não alteram o hash e mudanças de código continuam detectadas. Doctor remoto D050
recebido e conferido: PASS, source_sha256
`3b29d072fd51f2519df364b2ad0b4280b4edb88bb21841130f0ae807b0b96860`
igual ao esperado; ambos os locks coincidem, nove originais e probes CUDA aprovados.
Divergência resolvida. Próximo passo: smoke GPU com ID ainda inexistente e revisão
do novo lab-smoke.json, antes da preparação integral. T031 permanece
aberta. O histórico abaixo registra o impedimento anterior à autorização D050;
nenhum recibo foi reescrito.

Saída enviada pelo pesquisador: doctor PASS, nove originais, CUDA/amostragem e
forward/backward aprovados; hashes dos locks coincidem com os locais. Entretanto,
source_sha256 remoto `8b3ba58b5b553ecb4261cf91342753c864628c7ed3533259c462fa5476f4b519`
difere do local `78b6b4c475b31b1167270baff80af062411cec955174b4fc167386295b538bc7`
(commit local `5bcf15ed9dc7333e84ca66734cc7a03070fe3a68`, árvore limpa na conferência).
Não avançar ao novo smoke até esclarecer a identidade remota. Solicitar commit e
estado Git do laboratório; causa ainda não determinada. Este ponto não revoga
o PASS ambiental nem o aceite histórico T030; impede atribuir o probe ao código local atual.

Diagnóstico posterior: pesquisador confirmou o mesmo commit e árvore remota limpa.
O cálculo local inclui seis arquivos ignorados em `src/hgcl_elliptic.egg-info/`.
Recalculando apenas em memória sem esses metadados, os 67 arquivos restantes geram
exatamente o hash remoto `8b3ba58b5b553ecb4261cf91342753c864628c7ed3533259c462fa5476f4b519`.
A divergência é explicada pelos metadados de instalação locais, sem evidência de
diferença do código versionado. Proposta pendente: excluir diretórios `*.egg-info`
da identidade de fonte e adicionar regressão específica. Nenhuma correção aplicada;
consulta antes de alterar a proveniência e repetir o doctor na nova versão.

Atualizado: 2026-09-14. **T001–T030 concluídas. Matriz de 100 avaliações mantida; protocolo operacional T031 preparado.**

D048: XGBoost e DGI foram [adiados para após a primeira rodada](../specs/001-hgcl-experiment/deferred-extensions.md),
sem dependência ou critério de aceite adicional para T031–T033.
[Protocolo T031](../specs/001-hgcl-experiment/t031-execution-protocol.md) formalizado:
congelamento, preparação/validação no laboratório, auditoria, dry-run explícito,
aceite antes do treinamento, retomada e relatório portátil.
Próxima ação operacional: pesquisador versionar/sincronizar a entrega e conferir
pré-requisitos no laboratório antes da preparação integral. Nenhuma execução integral
foi realizada nesta entrega; T031 permanece aberta. [Revisão científica](literature-comparability-2026-09-13.md).

D049: implementadas localmente etapas `audit` e `dry-run` no wrapper de laboratório,
com resumos e identidades vinculados ao recibo da preparação. Regressão local: 65
testes aprovados e 2 opcionais ignorados. Código novo exige doctor/smoke remotos
atualizados; não invalida o aceite histórico T030. [Comandos e evidências locais](validation/t031-local-readiness.md).

## Objetivo científico e protocolo

Comparar fusão RF + H-GCL aos métodos isolados e medir o benefício do pré-treinamento
com poucos rótulos. Classificar endereços ativos por passo, inclusive inéditos; rótulos
globais são alvos de supervisão/avaliação, não entradas exigidas para inferência.
Treino 1–28, validação 29–34, teste 35–49; quatro métodos; 1/5/10/100%; cinco seeds;
80 avaliações principais + 20 complementares. Resultados vistos/inéditos separados.
Nenhuma dessas escolhas mudou durante a implementação do marco 1.

## Marcos

| Marco | Tasks | Estado |
| --- | --- | --- |
| 1. Ambiente e dados | T001–T012 | Concluído; 20 testes passaram, duas preparações originais idênticas |
| 2. Smoke no Mac | T013–T025 | Concluído; smoke original passou, treino + seleção em 2,18 s |
| 3. Matriz no laboratório | T026–T031 | T026–T030 concluídas; T031 preparação integral e matriz pendentes |
| 4. Fechamento | T032–T033 | Pendente; verificação completa e comandos finais |

## Evidências e limites

CLI `doctor`, `prepare`, `validate` funcionando. Recorte original preparado:
1.024 transações, 4.399 pares endereço–passo, 4.930 relações; atributos 123/32.
Evidências: [relatório](validation/milestone-1.md), [resultados](validation/milestone-1-results.json).
Testes do complemento nativo usam fixtures; preparação integral dos 49 passos ainda
não executada. Os quatro métodos foram executados no recorte de engenharia; nenhuma avaliação
da matriz científica foi executada. Evidências: [smoke.md](validation/smoke.md).
INV-001 e auditorias anteriores preservadas. Constituição continua como rascunho.

T026–T029 concluídas sob D039–D041. Regressão: 55 testes aprovados e dois opcionais
ignorados. Smoke original adicional executado separadamente: aprovado, treino + seleção
em 2,23 s (smoke-002). Matriz de 100 chaves enumerada; nenhuma execução científica.
Comandos matrix/resume/report disponíveis. Próximo passo: T030, preparar e verificar
ambiente, CUDA, amostrador e lock na máquina real do laboratório.
Evidência: [matrix-readiness.md](validation/matrix-readiness.md).
Regra permanente: [AGENTS.md](../AGENTS.md). Progresso: [tasks.md](../specs/001-hgcl-experiment/tasks.md).

T030: 61 testes aprovados; smoke CPU original repetido em 2,16 s de treino/seleção.
[Entrega local e pendências remotas](validation/lab-preflight.md);
[comandos para o laboratório](laboratorio-nixos.md).

Atualização remota: Flake/lock instalados, nove originais verificados e doctor CUDA
aprovado. `gpu-smoke-001` treinou os ramos, mas a repetição da inferência CUDA falhou
na tolerância antiga sem registrar magnitude. D045 autorizou auditoria numérica com
zero divergências de decisão. Regressão atual: 63 aprovados, 2 opcionais ignorados;
smoke CPU `smoke-004` aprovado. Próximo teste: `gpu-smoke-002` após atualizar o clone
e repetir o doctor para a nova identidade do código.

`gpu-smoke-002` aprovado: quatro métodos, 314 observações alinhadas por método,
treino + seleção em 5,820 s e avaliação em 0,188 s. As 1.300 pontuações por método
foram repetidas; diferença máxima 5,96e-8 e zero mudanças de decisão. Locks Nix/Python
versionados e conferidos pelos hashes do doctor. T030 concluída; T031 ainda não executada.

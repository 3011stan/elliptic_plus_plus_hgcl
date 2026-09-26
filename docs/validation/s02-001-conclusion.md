# Conclusão científica — s02-001

Estado: ciclo por carteira congelado em 2026-09-26.

## Conclusão sobre o aprendizado auto-supervisionado

No protocolo do `s02-001`, o pré-treinamento auto-supervisionado não apresentou
benefício consistente sobre o mesmo encoder treinado somente de forma supervisionada.
No regime principal, a diferença média pareada de F1 ilícito (`hgcl` menos
`graph_supervised`) foi de -10,45 pontos percentuais com 1% dos rótulos, +0,60 com
5%, +1,60 com 10% e -7,31 com 100%.

Assim, a expectativa de maior benefício do SSL sob escassez extrema de rótulos,
especialmente com 1%, não foi confirmada nesta rodada. Os pequenos ganhos com 5% e
10% vieram acompanhados de dispersão entre seeds maior que os ganhos médios e não
sustentam uma conclusão de superioridade geral.

## Diagnóstico resumido

O recall do H-GCL foi menor que o do grafo supervisionado em todas as frações. Com
5% e 10%, o aumento de precisão compensou parcialmente essa queda. Com 1%, o efeito
foi positivo no estrato de endereços vistos (+3,37 pontos percentuais de F1), mas
negativo nos não vistos (-10,59), que dominam a população de teste.

Esse padrão é compatível com uma transferência insuficiente para novos endereços,
mas não identifica sozinho a causa. O relatório portátil preserva médias, desvios e
seeds pareadas, não os valores individuais por seed; portanto, não permite afirmar
se cada diferença ocorreu nas cinco seeds ou se houve concentração em algumas delas.

## Limites da interpretação

- As cinco seeds são repetições sobre um único dataset, não datasets independentes.
- Os valores absolutos de F1 da literatura não são comparações diretas, pois mudam
  alvo, representação e protocolo experimental.
- O regime `native_wallet` muda a política de entrada e não constitui uma estimativa
  causal pura de vazamento.
- Os rótulos globais não determinam quando a atividade ilícita começou ou foi
  descoberta.

## Evidência preservada

- [Relatório agregado](s02-001-report.json): `status=complete`, 100/100 avaliações e
  nenhuma ausente.
- [Grupos executados](s02-001-groups.json): 25/25 grupos com estado `complete`.
- [Manifesto portátil](s02-001-manifest.json): preservado sem reescrita; corresponde
  ao planejamento/dry-run e ainda registra `status=planned` e
  `training_performed=false`, apesar dos resultados completos posteriores.

A especificação que originou o experimento permanece histórica e não foi alterada
para acomodar decisões tomadas após a observação dos resultados.

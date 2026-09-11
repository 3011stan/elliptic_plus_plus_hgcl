# Inspeção inicial dos CSVs — 2026-09-05

**Atualização 2026-09-07**: a [INV-001](inv-001/README.md) aprofundou a disponibilidade
temporal de três atributos, com contagens globais comprovadas, exemplos e limites de
interpretação. Esta página preserva o escopo e as limitações da inspeção inicial;
os resultados dirigidos posteriores estão no relatório vinculado.

Inspeção local de todos os nove arquivos fornecidos pelo pesquisador. O relatório completo,
com nomes exatos das colunas e contagens por tempo, está em [initial-audit.json](initial-audit.json).
Os hashes foram calculados antes da tentativa de movimentação e comparados após a cópia.
[Manifesto de origem](source-manifest.json); [comprovante da cópia](relocation.json).

O pesquisador posteriormente colocou os dados em `elliptic-plus-plus/raw` dentro do
projeto. Os nove hashes foram novamente verificados nessa localização, registrada em
[current-location.json](current-location.json). O caminho de origem no manifesto é
histórico e não determina o caminho de execução.

## Inventário observado

| Arquivo | Linhas de dados | Colunas |
| --- | ---: | ---: |
| wallets_features.csv | 1.268.260 | 57 |
| wallets_classes.csv | 822.942 | 2 |
| wallets_features_classes_combined.csv | 1.268.260 | 58 |
| txs_features.csv | 203.769 | 184 |
| txs_classes.csv | 203.769 | 2 |
| AddrTx_edgelist.csv | 477.117 | 2 |
| TxAddr_edgelist.csv | 837.124 | 2 |
| AddrAddr_edgelist.csv | 2.868.964 | 2 |
| txs_edgelist.csv | 234.355 | 2 |

Todos os arquivos têm cabeçalho e todas as linhas inspecionadas têm a largura esperada.
O nome real da coluna de tempo é `Time step`.

## Diferenças em relação ao anexo

- Carteiras: 57 colunas = endereço + tempo + **55 atributos**. Transações: 184 colunas =
  identificador + tempo + **182 atributos**. Esses são os números ao excluir ID e tempo;
  incluir tempo como entrada do modelo é uma decisão separada.
- Há 822.942 endereços únicos e 920.691 pares distintos `(address, Time step)`. Portanto,
  347.569 linhas são repetições adicionais de uma chave endereço-tempo.
- 61.487 endereços aparecem em mais de um tempo. As 55 colunas não temporais são idênticas
  em todas as ocorrências de cada endereço, inclusive entre tempos diferentes.
- `wallets_classes.csv` tem uma classe por endereço, sem duplicatas de ID. As classes do
  arquivo combinado coincidem com esse mapeamento em todas as linhas; seus atributos
  não temporais coincidem com os do arquivo de features por endereço.
- `txs_features.csv` tem 203.769 IDs únicos, com uma linha por transação.
- Os blocos semânticos de índices 0–15, 16–31, 32–47 e 48–55 do anexo não podem ser adotados
  como contrato das features. É necessário mapear nomes e significado das colunas reais.

## Classes de carteiras

| Unidade contada | Ilícita (1) | Lícita (2) | Desconhecida (3) |
| --- | ---: | ---: | ---: |
| Endereço único | 14.266 | 251.088 | 557.588 |
| Linha do arquivo combinado | 28.601 | 338.871 | 900.788 |

Contar linhas repetidas como exemplos independentes altera a ponderação das classes.
A unidade de predição e a política de deduplicação precisam ser explicitadas no protocolo.

## Integridade referencial observada

- Todas as referências de `AddrTx` e `TxAddr` existem nas tabelas de features.
- Para todas essas arestas, a carteira tem ocorrência no tempo da transação associada.
- Todos os extremos de `AddrAddr` e `txs_edgelist` existem nas tabelas correspondentes.
- As arestas de `txs_edgelist` conectam transações do mesmo tempo.
- Todas as carteiras com features possuem registro de classe.

## Limites da inspeção

Não foi feita uma varredura completa de validade numérica, valores ausentes ou duplicatas
de arestas. A verificação de chaves do arquivo combinado testa existência; não é uma
comparação da multiplicidade de cada chave contra a tabela de features.

**A disponibilidade histórica das features ainda não foi demonstrada.** A igualdade dos
atributos de uma carteira entre tempos é uma observação; a hipótese de que resumam toda
a sua atividade precisa ser investigada na documentação de extração. Até isso ser
resolvido, um corte de tempos não basta para justificar ausência de vazamento em features.

Os hashes identificam exatamente a cópia fornecida; não autenticam uma versão publicada
pelos autores. Os arquivos originais não foram deduplicados nem transformados.

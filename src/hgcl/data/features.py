"""Aggregate observed raw values on unique physical snapshot relations."""
import polars as pl
from hgcl.data.schema import TX_COLUMNS, ADDRESS_COLUMNS, canonicalize


def address_features(tx, wallets, edges):
    wallets=wallets.select('address','step').unique().sort('address','step')
    edges=edges.unique().sort('step','role','address','tx_id')
    joined=edges.join(tx.select('tx_id','step',*TX_COLUMNS),on=['tx_id','step'],how='left',validate='m:1')
    out=wallets
    for role in ('sender','receiver'):
        group=joined.filter(pl.col('role')==role)
        expressions=[pl.len().cast(pl.Float64).alias(f'n_{role}_txs')]
        for column in TX_COLUMNS:
            prefix=f'{role}_context__{column}__'
            expressions += [pl.col(column).sum().fill_null(0).alias(prefix+'sum'),
                            pl.col(column).mean().fill_null(0).alias(prefix+'mean'),
                            pl.col(column).max().fill_null(0).alias(prefix+'max'),
                            (pl.col(column).count()/pl.len()).alias(prefix+'observed_fraction')]
        summary=group.group_by('address','step').agg(expressions)
        out=out.join(summary,on=['address','step'],how='left')
    union=edges.select('address','step','tx_id').unique().group_by('address','step').len().rename({'len':'n_incident_txs'})
    return out.join(union,on=['address','step'],how='left').with_columns(
        [pl.col(c).fill_null(0).cast(pl.Float64) for c in ADDRESS_COLUMNS]
    ).select('address','step',*ADDRESS_COLUMNS).sort('address','step')


def transaction_features(tx, edges):
    out=canonicalize(tx,['tx_id'],'transaction features')
    for role,direction in [('sender','input'),('receiver','output')]:
        name=f'graph_num_{direction}_addresses'
        group=edges.filter(pl.col('role')==role).select('address','tx_id').unique().group_by('tx_id').len().rename({'len':name})
        out=out.join(group,on='tx_id',how='left').with_columns(pl.col(name).fill_null(0).cast(pl.Float64))
    return out.sort('tx_id')

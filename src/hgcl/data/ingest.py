"""Projected original-data reads, full source identity checks and separate targets."""
from dataclasses import dataclass
from pathlib import Path
import json
import polars as pl
from hgcl.provenance import file_hash
from hgcl.data.schema import TX_COLUMNS, NATIVE_COLUMNS, canonicalize, numeric


@dataclass
class Inputs:
    transactions: pl.DataFrame
    wallets: pl.DataFrame
    edges: pl.DataFrame
    diagnostics: dict


def verify_sources(root: Path, manifest_path: Path) -> dict:
    expected=json.loads(manifest_path.read_text())['files']
    if len(expected)!=9:
        raise ValueError('Expected the nine-file original manifest')
    for name, record in expected.items():
        path=root/name
        if not path.is_file() or path.stat().st_size!=record['bytes'] or file_hash(path)!=record['sha256']:
            raise ValueError(f'Original source identity mismatch: {name}')
    return expected


def read_columns(path: Path, columns: list[str]) -> pl.DataFrame:
    try:
        return pl.scan_csv(path,infer_schema=False).select(columns).collect(engine='streaming')
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f'{path.name}: CSV/schema error: {exc}') from exc


def steps(frame: pl.DataFrame, locator: str) -> pl.DataFrame:
    text=pl.col('Time step')
    parsed=text.cast(pl.Int64,strict=False)
    bad=frame.filter(parsed.is_null() | ~parsed.is_between(1,49))
    if bad.height:
        raise ValueError(f'{locator}: invalid step: {bad.head(3).to_dicts()}')
    return frame.with_columns(parsed.alias('Time step')).rename({'Time step':'step'})


def require_references(left, right, keys, locator):
    missing=left.select(keys).unique().join(right.select(keys).unique(),on=keys,how='anti')
    if missing.height:
        raise ValueError(f'{locator}: unsupported endpoints/occurrences: {missing.head(3).to_dicts()} (count={missing.height})')


def load_inputs(root: Path, native: bool = False) -> Inputs:
    tx=steps(read_columns(root/'txs_features.csv',['txId','Time step',*TX_COLUMNS]),'txs_features.csv').rename({'txId':'tx_id'})
    raw_tx=tx.height
    tx=canonicalize(numeric(tx,TX_COLUMNS,'txs_features.csv'),['tx_id'],'txs_features.csv')
    wallet_columns=['address','Time step',*(NATIVE_COLUMNS if native else [])]
    wallets=steps(read_columns(root/'wallets_features.csv',wallet_columns),'wallets_features.csv')
    raw_wallets=wallets.height
    if native: wallets=numeric(wallets,NATIVE_COLUMNS,'wallets_features.csv')
    wallets=canonicalize(wallets,['address','step'],'wallets_features.csv')
    relations=[]; duplicates={}
    for name,role,columns in [('AddrTx_edgelist.csv','sender',['input_address','txId']),('TxAddr_edgelist.csv','receiver',['output_address','txId'])]:
        edge=read_columns(root/name,columns).rename({columns[0]:'address','txId':'tx_id'})
        n=edge.height;edge=canonicalize(edge,['address','tx_id'],name);duplicates[role]=n-edge.height
        require_references(edge,tx,['tx_id'],name)
        edge=edge.join(tx.select('tx_id','step'),on='tx_id',how='left')
        require_references(edge,wallets,['address','step'],name)
        relations.append(edge.with_columns(pl.lit(role).alias('role')))
    edges=pl.concat(relations).sort('step','role','address','tx_id')
    require_references(wallets,edges,['address','step'],'wallet coverage')
    return Inputs(tx,wallets,edges,{'duplicate_transaction_rows':raw_tx-tx.height,'duplicate_wallet_rows':raw_wallets-wallets.height,'duplicate_edges':duplicates})


def load_targets(root: Path, wallets: pl.DataFrame) -> pl.DataFrame:
    targets=canonicalize(read_columns(root/'wallets_classes.csv',['address','class']),['address'],'wallets_classes.csv')
    if targets.filter(~pl.col('class').is_in(['1','2','3']) | pl.col('class').is_null()).height:
        raise ValueError('Invalid wallet class; expected 1/2/3')
    require_references(wallets,targets,['address'],'wallet target coverage')
    require_references(targets,wallets,['address'],'target wallet coverage')
    return targets.with_columns(pl.col('class').cast(pl.Int8))

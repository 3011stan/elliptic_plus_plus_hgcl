"""Exact feature allowlists and key/numeric validation."""
import polars as pl

TX_COLUMNS = ['total_BTC','fees','size','num_input_addresses','num_output_addresses',
              *['in_BTC_'+s for s in ('min','max','mean','median','total')],
              *['out_BTC_'+s for s in ('min','max','mean','median','total')]]
NATIVE_COLUMNS = ['num_txs_as_sender','num_txs_as receiver','first_block_appeared_in',
 'last_block_appeared_in','lifetime_in_blocks','total_txs','first_sent_block','first_received_block',
 'num_timesteps_appeared_in',
 *[p+s for p in ('btc_transacted_','btc_sent_','btc_received_','fees_','fees_as_share_',
                 'blocks_btwn_txs_','blocks_btwn_input_txs_','blocks_btwn_output_txs_')
   for s in ('total','min','max','mean','median')],
 'num_addr_transacted_multiple',
 *['transacted_w_address_'+s for s in ('total','min','max','mean','median')]]
ADDRESS_COLUMNS = ['n_sender_txs','n_receiver_txs','n_incident_txs'] + [
    f'{role}_context__{column}__{stat}' for role in ('sender','receiver')
    for column in TX_COLUMNS for stat in ('sum','mean','max','observed_fraction')]


def canonicalize(frame: pl.DataFrame, keys: list[str], locator: str) -> pl.DataFrame:
    if frame.select(pl.any_horizontal([pl.col(k).is_null() | (pl.col(k).cast(pl.String)=='') for k in keys]).any()).item():
        raise ValueError(f'{locator}: null/empty key')
    unique=frame.unique(maintain_order=True)
    conflicts=unique.group_by(keys).len().filter(pl.col('len')>1)
    if conflicts.height:
        raise ValueError(f'{locator}: conflicting duplicate keys: {conflicts.head(3).to_dicts()}')
    return unique.sort(keys)


def numeric(frame: pl.DataFrame, columns: list[str], locator: str) -> pl.DataFrame:
    for column in columns:
        text=pl.col(column).cast(pl.String).str.strip_chars().replace('',None)
        parsed=text.cast(pl.Float64,strict=False)
        bad=frame.filter(text.is_not_null() & (parsed.is_null() | ~parsed.is_finite() | (parsed<0)))
        if bad.height:
            raise ValueError(f'{locator}: invalid {column}: {bad.select(list(dict.fromkeys(frame.columns[:2]+[column]))).head(3).to_dicts()}')
        frame=frame.with_columns(parsed.alias(column))
    return frame

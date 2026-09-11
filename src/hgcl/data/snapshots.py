"""Deterministic label-free recut and independent per-step graph arrays."""
import hashlib
import numpy as np
import polars as pl


def recut(tx, selected_steps, cap, seed):
    kept=tx.filter(pl.col('step').is_in(selected_steps))
    if cap is not None:
        ids=[]
        for step in sorted(selected_steps):
            candidates=kept.filter(pl.col('step')==step)['tx_id'].to_list()
            ids.extend(sorted(candidates,key=lambda t:(hashlib.sha256(f'{seed}|{step}|{t}'.encode()).digest(),t))[:cap])
        kept=kept.filter(pl.col('tx_id').is_in(ids))
    return kept.sort('tx_id')


def make_snapshot(address,tx,edges,address_preprocessor,tx_preprocessor):
    address=address.sort('address');tx=tx.sort('tx_id');edges=edges.unique().sort('role','address','tx_id')
    steps=set(address['step'].to_list()+tx['step'].to_list()+edges['step'].to_list())
    if len(steps)!=1:raise ValueError('Snapshot must contain exactly one step')
    a_ids=address['address'].to_list();t_ids=tx['tx_id'].to_list()
    ai={key:i for i,key in enumerate(a_ids)};ti={key:i for i,key in enumerate(t_ids)}
    tabular,ax=address_preprocessor.transform(address);_,txx=tx_preprocessor.transform(tx)
    result={'step':np.array(next(iter(steps)),dtype=np.int64),'address_ids':np.array(a_ids,dtype=str),
            'transaction_ids':np.array(t_ids,dtype=str),'address_x':ax,'transaction_x':txx,'tabular_x':tabular}
    for role in ('sender','receiver'):
        rows=edges.filter(pl.col('role')==role).select('address','tx_id').rows()
        pairs=[(ai[a],ti[t]) if role=='sender' else (ti[t],ai[a]) for a,t in rows]
        forward=np.asarray(pairs,dtype=np.int64).reshape(-1,2).T.copy()
        result[role]=forward;result[role+'_reverse']=forward[::-1].copy()
        result[role+'_edge_ids']=np.arange(len(pairs),dtype=np.int64)
    return result

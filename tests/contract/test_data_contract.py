"""Independent expected values for the scientific data contract."""
import numpy as np
import polars as pl
import pytest
from hgcl.data.schema import TX_COLUMNS, canonicalize, numeric
from hgcl.data.features import address_features, transaction_features
from hgcl.data.preprocessing import Preprocessor
from hgcl.data.snapshots import recut, make_snapshot


def fixture():
    tx=pl.DataFrame({'tx_id':['x','y','z'], 'step':[1,1,35], **{c:[2.,None,999.] for c in TX_COLUMNS}})
    wallets=pl.DataFrame({'address':['a','b','c','d'], 'step':[1,1,1,35]})
    edges=pl.DataFrame({'address':['a','a','b','c','d'], 'tx_id':['x','x','x','y','z'], 'role':['sender','receiver','receiver','sender','receiver'], 'step':[1,1,1,1,35]})
    return tx,wallets,edges


def test_duplicates_and_roles():
    tx,w,e=fixture()
    a=address_features(tx,w,e)
    repeated=address_features(tx,w.vstack(w.head(1)),e.vstack(e.head(1)))
    assert a.equals(repeated)
    row=a.filter(pl.col('address')=='a').row(0,named=True)
    assert [row[c] for c in ('n_sender_txs','n_receiver_txs','n_incident_txs')]==[1,1,1]
    assert row['sender_context__fees__sum']==2
    assert row['receiver_context__fees__observed_fraction']==1
    assert a.filter(pl.col('address')=='c')['sender_context__fees__observed_fraction'][0]==0
    assert a.filter(pl.col('address')=='b')['n_sender_txs'][0]==0
    assert a.width==125
    t=transaction_features(tx,e)
    assert t.filter(pl.col('tx_id')=='x')['graph_num_output_addresses'][0]==2


def test_missing_and_zero_are_distinct():
    tx,w,e=fixture();tx=tx.with_columns(pl.when(pl.col('tx_id')=='x').then(0.).otherwise(pl.col('fees')).alias('fees'))
    t=transaction_features(tx,e)
    p=Preprocessor.fit(t.filter(pl.col('step')==1),kind='transaction')
    raw,scaled=p.transform(t)
    assert raw[0,p.names.index('fees')]==0
    assert raw[0,p.names.index('fees__missing')]==0
    assert raw[1,p.names.index('fees__missing')]==1
    assert np.isfinite(scaled).all()


def test_future_perturbation_cannot_change_training():
    tx,w,e=fixture();changed=tx.with_columns(pl.when(pl.col('step')==35).then(1e12).otherwise(pl.col('fees')).alias('fees'))
    for builder,kind in [(lambda t:address_features(t,w,e),'principal'),(lambda t:transaction_features(t,e),'transaction')]:
        first=builder(tx).filter(pl.col('step')==1);second=builder(changed).filter(pl.col('step')==1)
        assert first.equals(second)
        assert Preprocessor.fit(first,kind=kind).to_dict()==Preprocessor.fit(second,kind=kind).to_dict()


def test_bad_data_fails_with_locator():
    with pytest.raises(ValueError,match='conflict'):
        canonicalize(pl.DataFrame({'tx_id':['x','x'],'step':[1,2]}),['tx_id'],'fixture')
    for bad in ['oops','NaN','inf','-1']:
        with pytest.raises(ValueError,match='fees'):
            numeric(pl.DataFrame({'tx_id':['x'],'fees':[bad]}),['fees'],'fixture')
    tx,_,e=fixture();tx=tx.with_columns(pl.lit(None,dtype=pl.Float64).alias('fees'))
    with pytest.raises(ValueError,match='fees'):
        Preprocessor.fit(transaction_features(tx,e),kind='transaction')


def test_snapshot_ids_and_recut_are_label_free():
    tx,w,e=fixture();t1=recut(tx,[1,35],1,11);t2=recut(tx.reverse(),[1,35],1,11)
    assert t1.equals(t2)
    at=address_features(tx,w,e).filter(pl.col('step')==1)
    tt=transaction_features(tx,e).filter(pl.col('step')==1)
    ap=Preprocessor.fit(at,kind='principal');tp=Preprocessor.fit(tt,kind='transaction')
    snapshot=make_snapshot(at,tt,e.filter(pl.col('step')==1),ap,tp)
    assert 'class' not in snapshot and 'y' not in snapshot
    assert snapshot['address_x'].shape==(3,123)
    assert snapshot['transaction_x'].shape==(2,32)
    np.testing.assert_array_equal(snapshot['sender_reverse'],snapshot['sender'][::-1])
    np.testing.assert_array_equal(snapshot['receiver_reverse'],snapshot['receiver'][::-1])


def test_native_contract_and_conflicting_duplicate():
    from hgcl.data.schema import NATIVE_COLUMNS
    assert len(NATIVE_COLUMNS)==55 and len(set(NATIVE_COLUMNS))==55
    wallet=pl.DataFrame({'address':['a','b'],'step':[1,1],**{c:[0.,None] for c in NATIVE_COLUMNS}})
    pre=Preprocessor.fit(wallet,kind='native_wallet');raw,scaled=pre.transform(wallet)
    assert raw.shape==(2,110) and scaled.shape==(2,110)
    assert raw[0,55:].sum()==0 and raw[1,55:].sum()==55
    with pytest.raises(ValueError,match='conflict'):
        canonicalize(wallet.vstack(wallet.head(1).with_columns(pl.lit(7.).alias('fees_total'))),['address','step'],'native')


def test_native_allowlist_matches_spec(project_root):
    from hgcl.data.schema import NATIVE_COLUMNS
    text=(project_root/'specs/001-hgcl-experiment/input-contract.md').read_text()
    documented=text.split('```text\n',1)[1].split('```',1)[0].strip().splitlines()
    assert NATIVE_COLUMNS==documented

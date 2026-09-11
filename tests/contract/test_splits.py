import polars as pl
import pytest
from hgcl.data.splits import budgets, occurrence_masks


def test_nested_address_budgets_and_no_backfill():
    labels=pl.DataFrame({'address':[f'a{i}' for i in range(20)],'class':[1]*10+[2]*10})
    w=pl.DataFrame({'address':[f'a{i}' for t in [1,2,29,35] for i in range(20)],'step':[t for t in [1,2,29,35] for i in range(20)]})
    split={'train':[1,2],'validation':[29],'test':[35]}
    records,recurrence=budgets(w,labels,split,[.1,.5,1.],[11])
    assert len(records[0]['selected']['train'])==2
    assert set(records[0]['selected']['train'])<=set(records[1]['selected']['train'])
    masks=occurrence_masks(w,labels,split,records[0])
    assert masks['train_mask'].sum()==4
    assert not masks.filter(pl.col('step')==29)['train_mask'].any()
    assert not masks.filter(pl.col('step')==35)['validation_mask'].any()
    assert recurrence['seen_in_train'].all()
    with pytest.raises(ValueError,match='zero class'):budgets(w,labels,split,[.01],[11])


def test_unknown_not_negative_and_unseen_reporting():
    labels=pl.DataFrame({'address':['a','b','u','new'],'class':[1,2,3,1]})
    w=pl.DataFrame({'address':['a','b','a','b','a','b','u','new'],'step':[1,1,29,29,35,35,35,35]})
    split={'train':[1],'validation':[29],'test':[35]}
    records,recurrence=budgets(w,labels,split,[1.],[11])
    m=occurrence_masks(w,labels,split,records[0]);u=m.filter(pl.col('address')=='u')
    assert u['target'][0] is None and not u['test_mask'][0]
    assert not recurrence.filter(pl.col('address')=='new')['seen_in_development'][0]

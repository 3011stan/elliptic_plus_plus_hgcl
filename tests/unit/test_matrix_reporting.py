import copy
import statistics
import polars as pl
import pytest
from hgcl.config import defaults
from hgcl.matrix import design, dry_run
from hgcl.evaluation.reporting import aggregate, strata, summary


def test_matrix_has_80_plus_20_and_shared_ssl():
    plan=design(defaults('lab'))
    assert len(plan['groups'])==25
    assert sum(e['regime']=='principal' for e in plan['evaluations'])==80
    assert sum(e['regime']=='native_wallet' for e in plan['evaluations'])==20
    assert len({g['ssl'] for g in plan['groups']})==10
    assert plan['expected_candidate_configurations']==150
    visited=set()
    for node in plan['nodes']:
        assert set(node['depends_on'])<=visited
        assert node['id'] not in visited
        visited.add(node['id'])


def test_strata_separate_presence_and_label_exposure():
    frame=pl.DataFrame({'address':['a','b','c'], 'target':[1,0,1], 'score':[.9,.1,.2],
        'seen_in_development':[True,True,False], 'label_exposed_train':[True,False,False],
        'label_exposed_validation':[False,False,False]})
    result=strata(frame,.5)
    assert result['seen']['support']==2
    assert result['unseen']['support']==1
    assert result['train_label_exposed']['support']==1
    assert result['label_unexposed']['support']==2
    assert result['validation_label_exposed']['support']==0
    assert result['validation_label_exposed']['f1_illicit'] is None
    assert result['unseen']['ap'] is None


def test_five_seed_summary_and_negative_paired_difference():
    expected=design(defaults('lab'))['evaluations'];rows=[]
    for item in expected:
        value=.2 if item['method']=='fusion' else .8
        measurement={m:value for m in ('f1_illicit','precision','recall','ap','mcc')}
        measurement.update(support=10,positive=5,negative=5,unique_addresses=9)
        rows.append({**item,'strata':{'overall':measurement}})
    report=aggregate(rows,expected)
    assert report['status']=='complete'
    assert all(a['metrics']['f1_illicit']['n_defined']==5 for a in report['aggregates'])
    assert all(c['f1_difference']['mean']==pytest.approx(-.6) for c in report['paired_comparisons'] if c['question']=='RQ1')
    partial=aggregate(rows[:-1],expected)
    assert partial['status']=='incomplete' and partial['missing_evaluations']==[expected[-1]['id']]
    assert summary([1,2,3,4,5])['sample_sd']==statistics.stdev([1,2,3,4,5])
    assert not summary([1,None,3,4,5])['complete']
    assert summary([None])['mean'] is None
    with pytest.raises(ValueError,match='Duplicate'):aggregate(rows+rows[:1],expected)
    wrong=copy.deepcopy(rows);wrong[0]['seed']=999
    with pytest.raises(ValueError,match='metadata'):aggregate(wrong,expected)

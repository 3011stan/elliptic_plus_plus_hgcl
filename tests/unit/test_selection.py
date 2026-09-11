import numpy as np
import pytest
from hgcl.evaluation.selection import select_threshold,select_fusion
from hgcl.evaluation.metrics import metrics


def test_threshold_agrees_with_bruteforce_and_endpoints():
    y=np.array([0,1,1,0]);s=np.array([.2,.4,.4,.9])
    options=np.unique(np.r_[0.,s,np.nextafter(1.,2.)])
    best=min(options,key=lambda t:(-metrics(y,s,t)['f1_illicit'],abs(t-.5),t))
    assert select_threshold(y,s)['threshold']==best
    assert select_threshold([0,1],[0.,0.])['threshold']==0.
    # Equal F1 and equal distance to .5: accepted final tie-break is lower threshold.
    assert select_threshold([0,1],[1.,1.])['threshold']==0.


def test_fusion_ties_and_metric_nulls():
    best,rows=select_fusion([0,1],[.1,.9],[.1,.9],[0.,.5,1.])
    assert best['alpha']==.5 and len(rows)==3
    assert metrics([0,0],[.1,.2],.5)['f1_illicit'] is None
    assert metrics([0,1],[0.,0.],.5)['precision']==0.
    assert metrics([],[],.5)['mcc'] is None
    with pytest.raises(ValueError):select_threshold([0,1],[float('nan'),.5])


def test_threshold_tie_uses_closest_then_lower():
    assert select_threshold([0,1],[.7,.7])['threshold']==.7
    assert select_threshold([0,1],[1.,1.])['threshold']==0.

"""RF candidates using the same selected address observations as graph fits."""
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from hgcl.data.sampling import load_graph
from hgcl.training.supervised import permitted_targets
from hgcl.provenance import atomic_json


def table(prepared,regime,steps,targets=None):
    xs=[];rows=[]
    for step in steps:
        graph=load_graph(Path(prepared),regime,step)
        rows.extend(zip(graph.ids['address'].tolist(),[step]*len(graph.tabular)));xs.append(graph.tabular)
    keys=pl.DataFrame(rows,schema=['address','step'],orient='row').with_row_index('row')
    x=np.concatenate(xs)
    if targets is None:return keys,x
    joined=targets.join(keys,on=['address','step'],how='left',validate='1:1')
    if joined['row'].null_count():raise ValueError('Missing tabular observation')
    return x[joined['row'].to_numpy()],joined['target'].to_numpy()


def fit_rf(prepared,regime,values,record,directory,check):
    train=permitted_targets(prepared,record,values,'train');validation=permitted_targets(prepared,record,values,'validation')
    x,y=table(prepared,regime,values['split']['train'],train);vx,vy=table(prepared,regime,values['split']['validation'],validation)
    best=None;best_ap=-1.;reports=[]
    for leaf in values['rf']['min_leaf']:
        check();model=RandomForestClassifier(n_estimators=values['rf']['trees'],min_samples_leaf=leaf,max_features=values['rf']['max_features'],class_weight=values['rf']['class_weight'],n_jobs=values['rf']['jobs'],random_state=record['seed'])
        model.fit(x,y);check()
        score=model.predict_proba(vx)[:,list(model.classes_).index(1)]
        ap=float(average_precision_score(vy,score));reports.append({'min_leaf':leaf,'validation_ap':ap})
        if ap>best_ap:best=model;best_ap=ap
    Path(directory).mkdir(parents=True,exist_ok=True);atomic_json(Path(directory)/'rf-search.json',reports)
    return best,reports


def rf_scores(model,prepared,regime,steps,check):
    keys,x=table(prepared,regime,steps);check()
    scores=model.predict_proba(x)[:,list(model.classes_).index(1)];check()
    return keys.drop('row').with_columns(pl.Series('score',scores)).sort('address','step')

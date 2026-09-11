import math
import numpy as np
from sklearn.metrics import average_precision_score


def check_scores(y,scores):
    y=np.asarray(y);scores=np.asarray(scores,dtype=float)
    if y.ndim!=1 or scores.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Invalid binary targets or finite [0,1] scores')
    return y.astype(int),scores


def metrics(y,scores,threshold):
    y,scores=check_scores(y,scores)
    if not math.isfinite(threshold):raise ValueError('Threshold must be finite')
    pred=scores>=threshold
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    positive=tp+fn;negative=tn+fp
    denominator=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    values={'f1_illicit':2*tp/(2*tp+fp+fn) if positive else None,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/positive if positive else None,
            'ap':float(average_precision_score(y,scores)) if positive and negative else None,
            'mcc':(tp*tn-fp*fn)/math.sqrt(denominator) if denominator else None}
    reasons={k:('no positive targets' if k in ('f1_illicit','recall') else 'single class or empty population' if k=='ap' else 'zero MCC denominator') for k,v in values.items() if v is None}
    return {**values,'support':len(y),'positive':positive,'negative':negative,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'undefined':reasons}

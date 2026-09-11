import numpy as np
from hgcl.evaluation.metrics import check_scores,metrics


def select_threshold(y,scores):
    y,scores=check_scores(y,scores)
    if len(np.unique(y))!=2:raise ValueError('Validation requires both classes')
    candidates=np.unique(np.r_[0.,scores,np.nextafter(1.,2.)])
    # Stable descending score groups; evaluate every distinct cutoff in O(n log n).
    order=np.argsort(-scores,kind='stable');sorted_s=scores[order];sorted_y=y[order]
    positives=int(y.sum());tp=0;fp=0;lookup={float(np.nextafter(1.,2.)):0.}
    for i in range(len(order)):
        tp+=int(sorted_y[i]);fp+=1-int(sorted_y[i])
        if i==len(order)-1 or sorted_s[i+1]!=sorted_s[i]:lookup[float(sorted_s[i])]=2*tp/(tp+fp+positives)
    lookup[0.]=2*positives/(len(y)+positives)
    threshold=min(candidates,key=lambda t:(-lookup[float(t)],abs(t-.5),t))
    return {'threshold':float(threshold),'validation_f1':lookup[float(threshold)]}


def select_fusion(y,rf,hgcl,alphas):
    y,rf=check_scores(y,rf);_,hgcl=check_scores(y,hgcl)
    results=[]
    for alpha in alphas:
        if not 0<=alpha<=1:raise ValueError('Invalid fusion weight')
        results.append({'alpha':float(alpha),**select_threshold(y,alpha*rf+(1-alpha)*hgcl)})
    best=min(results,key=lambda r:(-r['validation_f1'],abs(r['alpha']-.5),r['alpha']))
    return best,results

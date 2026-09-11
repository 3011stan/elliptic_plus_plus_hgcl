"""Pure input transforms: no targets, no changing descriptors after edge dropout."""
import torch


def augment(x,edges,numeric_indices,feature_mask,edge_dropout,generator):
    if not 0<=feature_mask<=1 or not 0<=edge_dropout<=1:raise ValueError('Invalid augmentation rate')
    view={t:a.clone() for t,a in x.items()}
    for t,columns in numeric_indices.items():
        keep=torch.rand((len(x[t]),len(columns)),generator=generator)>=feature_mask
        view[t][:,columns]*=keep.to(view[t].device)
    changed={}
    for role in ('sender','receiver'):
        edge=edges[role]
        keep=torch.rand(edge.shape[1],generator=generator)>=edge_dropout
        changed[role]=edge[:,keep.to(edge.device)]
        changed[role+'_reverse']=changed[role].flip(0)
    return view,changed

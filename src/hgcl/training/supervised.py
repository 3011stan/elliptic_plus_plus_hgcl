"""Matched fine-tuning and from-scratch fits, with validation-only selection."""
from pathlib import Path
import copy
import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score
from hgcl.models.heads import Classifier
from hgcl.data.sampling import load_graph,anchor_batches,batch
from hgcl.training.pretrain import fresh_encoder,finite_gradients
from hgcl.training.inference import infer
from hgcl.training.checkpoint import save,load
from hgcl.provenance import atomic_json


def permitted_targets(prepared,record,values,partition):
    if partition not in ('train','validation'):raise ValueError('Fitting cannot request test targets')
    # Predicate/projection before materialization: no test target rows enter fitting.
    return (pl.scan_parquet(Path(prepared)/'targets'/(record['hash']+'.parquet'))
            .filter(pl.col('step').is_in(values['split'][partition]) & pl.col(partition+'_mask'))
            .select('address','step','target').sort('address','step').collect())


def selected_for_graph(graph,targets):
    rows=targets.filter(pl.col('step')==graph.step);mapping={str(a):i for i,a in enumerate(graph.ids['address'])}
    indices=torch.tensor([mapping[a] for a in rows['address']],dtype=torch.long)
    return indices,torch.tensor(rows['target'].to_list(),dtype=torch.float32)


@torch.no_grad()
def graph_scores(model,prepared,regime,steps,resources,check):
    previous=model.training;model.eval();rows=[];device=next(model.parameters()).device
    try:
        for step in steps:
            graph=load_graph(Path(prepared),regime,step)
            embedding=infer(model.encoder,graph.x,graph.edges,resources['inference_nodes'],resources['inference_edges'],check)['address']
            scores=[]
            for start in range(0,len(embedding),resources['inference_nodes']):
                check();scores.extend(torch.sigmoid(model.head(embedding[start:start+resources['inference_nodes']].to(device))).flatten().cpu().tolist())
            rows.extend(zip(graph.ids['address'].tolist(),[step]*len(scores),scores))
        return pl.DataFrame(rows,schema=['address','step','score'],orient='row').sort('address','step')
    finally:model.train(previous)


def fit_graph(prepared,regime,values,record,initial_state,directory,dependencies,check):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    train=permitted_targets(prepared,record,values,'train');validation=permitted_targets(prepared,record,values,'validation')
    positive=int(train['target'].sum());negative=train.height-positive
    if not positive or not negative:raise ValueError('Supervised training needs both classes')
    seed=record['seed'];first=load_graph(Path(prepared),regime,values['split']['train'][0]);device=values['resources']['device']
    results=[];chosen=None;chosen_ap=-1.
    for candidate,lr in enumerate(values['supervised']['lrs']):
        encoder=fresh_encoder(first.x['address'].shape[1],first.x['transaction'].shape[1],values,seed)
        if initial_state is not None:encoder.load_state_dict(copy.deepcopy(initial_state))
        model=Classifier(encoder,values['encoder']['hidden'],seed).to(device)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=values['supervised']['weight_decay'])
        generator=torch.Generator().manual_seed(seed+50000)
        latest=directory/f'candidate-{candidate}-latest.pt';epoch_start=0;best_ap=-1.;best_state=None;best_epoch=0;history=[]
        dep={**dependencies,'candidate':candidate,'lr':lr}
        if latest.exists():
            saved=load(latest,model,optimizer,dep,generators={'anchors':generator});epoch_start=saved['epoch']
            extra=saved['extra'];best_ap=extra['best_ap'];best_state=extra['best_state'];best_epoch=extra['best_epoch'];history=extra['history'];initial=extra['initial']
        for epoch in range(epoch_start,values['supervised']['epochs']):
            if best_epoch and epoch-best_epoch>=values['supervised']['patience']:break
            model.train();losses=[]
            for step in values['split']['train']:
                graph=load_graph(Path(prepared),regime,step);indices,labels=selected_for_graph(graph,train)
                lookup={int(i):float(y) for i,y in zip(indices,labels)}
                for anchors in anchor_batches(indices,values['resources']['anchor_batch'],generator):
                    if not len(anchors):continue
                    check();optimizer.zero_grad();x,edges,local,canonical=batch(graph,'address',anchors,values['resources'])
                    x={t:a.to(device) for t,a in x.items()};edges={r:e.to(device) for r,e in edges.items()}
                    target=torch.tensor([lookup[int(i)] for i in canonical],device=device)
                    logits=model(x,edges)[local.to(device)]
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(negative/positive,device=device))
                    finite_gradients(model,loss);optimizer.step();losses.append(loss.item());check()
            scored=graph_scores(model,prepared,regime,values['split']['validation'],values['resources'],check)
            joined=validation.join(scored,on=['address','step'],how='left',validate='1:1')
            if joined['score'].null_count():raise ValueError('Missing validation predictions')
            ap=float(average_precision_score(joined['target'].to_numpy(),joined['score'].to_numpy()))
            if ap>best_ap:best_ap=ap;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append({'epoch':epoch+1,'validation_ap':ap,'mean_loss':sum(losses)/len(losses)})
            save(latest,model,optimizer,epoch+1,dep,generators={'anchors':generator},extra={'best_ap':best_ap,'best_state':best_state,'best_epoch':best_epoch,'history':history,'initial':initial})
        if best_state is None:raise RuntimeError('No selected graph checkpoint')
        encoder_changed=any(not torch.equal(best_state[k],initial[k]) for k in initial if k.startswith('encoder.'))
        head_changed=any(not torch.equal(best_state[k],initial[k]) for k in initial if k.startswith('head.'))
        if not encoder_changed or not head_changed:raise RuntimeError('Selected graph encoder/head did not update')
        report={'candidate':candidate,'lr':lr,'validation_ap':best_ap,'best_epoch':best_epoch,'history':history,
                'encoder_changed':encoder_changed,'head_changed':head_changed,'positive_weight':negative/positive}
        results.append(report)
        if best_ap>chosen_ap:
            model.load_state_dict(best_state);chosen=model;chosen_ap=best_ap
    atomic_json(directory/'graph-search.json',results)
    return chosen,results

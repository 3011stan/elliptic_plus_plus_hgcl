"""Label-free fixed-epoch SSL with immutable final state and resumable epochs."""
from pathlib import Path
import copy
import torch
from torch import nn
from hgcl.models.encoder import Encoder,TYPES
from hgcl.models.contrastive import Projection,contrastive_loss
from hgcl.models.augmentations import augment
from hgcl.data.sampling import load_graph,anchor_batches,batch
from hgcl.training.checkpoint import save,load
from hgcl.provenance import file_hash,atomic_json


class SSLModel(nn.Module):
    def __init__(self,encoder,dim):
        super().__init__();self.encoder=encoder;self.projection=Projection(encoder.hidden,dim)


def fresh_encoder(address_dim,transaction_dim,values,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Encoder(address_dim,transaction_dim,values['encoder']['hidden'],values['encoder']['layers'])


def finite_gradients(model,loss):
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Nonfinite training gradient')


def pretrain(prepared,regime,values,seed,directory,dependencies,check,*,stop_after_epoch=None):
    prepared=Path(prepared);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    final=directory/'ssl-final.pt'
    if final.exists():
        state=torch.load(final,map_location='cpu',weights_only=False)
        if state['dependencies']!=dependencies:raise ValueError('Incompatible immutable SSL cache')
        return {k.removeprefix('encoder.'):v.clone() for k,v in state['model'].items() if k.startswith('encoder.')},state['extra']
    first=load_graph(prepared,regime,values['split']['train'][0])
    encoder=fresh_encoder(first.x['address'].shape[1],first.x['transaction'].shape[1],values,seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+20000);model=SSLModel(encoder,values['ssl']['projection_dim'])
    device=values['resources']['device'];model.to(device)
    initial={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=values['ssl']['lr'],weight_decay=values['ssl']['weight_decay'])
    generators={'anchors':torch.Generator().manual_seed(seed+30000),'views':torch.Generator().manual_seed(seed+40000)}
    epoch_start=0;history=[];updates=0
    latest=directory/'latest.pt'
    if latest.exists():
        state=load(latest,model,optimizer,dependencies,generators=generators)
        epoch_start=state['epoch'];history=state['extra']['history'];updates=state['extra']['updates'];initial=state['extra']['initial_encoder']
    for epoch in range(epoch_start,values['ssl']['epochs']):
        model.train();losses=[]
        for step in values['split']['train']:
            graph=load_graph(prepared,regime,step)
            anchors={t:anchor_batches(torch.arange(len(graph.x[t])),values['resources']['anchor_batch'],generators['anchors'],min_size=2) for t in TYPES}
            if any(not a for a in anchors.values()):raise ValueError('Snapshot lacks sufficient same-type SSL anchors')
            # Equal type weighting per update: cycle the shorter type's anchor list.
            for i in range(max(map(len,anchors.values()))):
                check();optimizer.zero_grad();terms=[]
                for kind in TYPES:
                    selected=anchors[kind][i%len(anchors[kind])]
                    x,edges,local,canonical=batch(graph,kind,selected,values['resources'])
                    for k in x:x[k]=x[k].to(device)
                    edges={k:e.to(device) for k,e in edges.items()}
                    x1,e1=augment(x,edges,graph.numeric,values['ssl']['feature_mask'],values['ssl']['edge_dropout'],generators['views'])
                    x2,e2=augment(x,edges,graph.numeric,values['ssl']['feature_mask'],values['ssl']['edge_dropout'],generators['views'])
                    local=local.to(device);canonical=canonical.to(device)
                    z1=model.projection(kind,model.encoder(x1,e1)[kind][local])
                    z2=model.projection(kind,model.encoder(x2,e2)[kind][local])
                    terms.append(contrastive_loss(z1,z2,canonical,canonical,values['ssl']['temperature']))
                loss=torch.stack(terms).mean();finite_gradients(model,loss);optimizer.step();updates+=1
                losses.append(loss.item());check()
        history.append({'epoch':epoch+1,'mean_loss':sum(losses)/len(losses),'updates':updates})
        extra={'history':history,'updates':updates,'initial_encoder':initial}
        save(latest,model,optimizer,epoch+1,dependencies,generators=generators,extra=extra)
        if stop_after_epoch==epoch+1:raise InterruptedError('Requested epoch-boundary test interruption')
    changed=any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in encoder.state_dict().items())
    if not changed:raise RuntimeError('SSL encoder weights did not change')
    report={'history':history,'updates':updates,'encoder_changed':changed,'type_batch_policy':'cycle shorter type list; equal type losses per update'}
    save(final,model,optimizer,values['ssl']['epochs'],dependencies,generators=generators,extra=report,immutable=True)
    atomic_json(directory/'ssl-report.json',{**report,'checkpoint_sha256':file_hash(final)})
    return {k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()},report

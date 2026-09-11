"""Exact all-neighbor inference, bounded by destination and edge chunks."""
import torch
from hgcl.models.encoder import TYPES,RELATIONS


@torch.no_grad()
def infer(encoder,x,edges,node_chunk=4096,edge_chunk=100000,check=lambda:None):
    if node_chunk<1 or edge_chunk<1:raise ValueError('Inference chunk sizes must be positive')
    previous=encoder.training;encoder.eval();device=next(encoder.parameters()).device
    try:
        h={}
        for t in TYPES:
            blocks=[]
            for start in range(0,len(x[t]),node_chunk):
                check();blocks.append(encoder.projections[t](x[t][start:start+node_chunk].to(device)).cpu())
            h[t]=torch.cat(blocks) if blocks else torch.empty((0,encoder.hidden))
        for layer in encoder.layers:
            updated={t:torch.empty_like(h[t]) for t in TYPES}
            for dst in TYPES:
                for start in range(0,len(h[dst]),node_chunk):
                    end=min(start+node_chunk,len(h[dst]));check()
                    summed=(1+layer.epsilon[dst])*h[dst][start:end].to(device)
                    for name,(src,destination) in RELATIONS.items():
                        if destination!=dst:continue
                        edge=edges[name].cpu()
                        for e_start in range(0,edge.shape[1],edge_chunk):
                            block=edge[:,e_start:e_start+edge_chunk]
                            block=block[:,(block[1]>=start)&(block[1]<end)]
                            if not block.shape[1]:continue
                            messages=layer.messages[name](h[src][block[0]].to(device))
                            summed.index_add_(0,(block[1]-start).to(device),messages)
                            check()
                    updated[dst][start:end]=layer.update[dst](summed).cpu()
            h=updated
        return h
    finally:encoder.train(previous)

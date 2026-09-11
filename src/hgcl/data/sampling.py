"""Graph-only loading with explicit anchor batches and bounded neighbor adapter."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from hgcl.models.encoder import RELATIONS,TYPES


@dataclass
class Graph:
    step: int
    x: dict
    edges: dict
    ids: dict
    numeric: dict
    tabular: np.ndarray


def load_graph(prepared: Path,regime: str,step: int):
    import json
    folder=prepared/regime/f'step-{step:02}'
    def load(name):return np.load(folder/(name+'.npy'),allow_pickle=False)
    ap=json.loads((prepared/regime/'address-preprocessor.json').read_text())
    tp=json.loads((prepared/'transaction-preprocessor.json').read_text())
    return Graph(step,{t:torch.from_numpy(load(t+'_x')) for t in TYPES},
                 {r:torch.from_numpy(load(r)) for r in RELATIONS},
                 {t:load(t+'_ids') for t in TYPES},
                 {'address':ap['numeric_indices'],'transaction':tp['numeric_indices']},load('tabular_x'))


def anchor_batches(indices,batch_size,generator=None,shuffle=True,min_size=1):
    indices=torch.as_tensor(indices,dtype=torch.long)
    if batch_size<min_size:raise ValueError('Batch size below minimum anchor count')
    if shuffle:indices=indices[torch.randperm(len(indices),generator=generator)]
    chunks=list(indices.split(batch_size))
    if chunks and len(chunks[-1])<min_size:
        if len(chunks)>1:
            tail=chunks.pop()
            chunks[-1]=torch.cat([chunks[-1],tail])
        else:chunks=[]
    return chunks


def guard(x,edges,max_nodes,max_edges):
    if sum(len(a) for a in x.values())>max_nodes or sum(a.shape[1] for a in edges.values())>max_edges:
        raise RuntimeError('Graph loading exceeds the configured node/edge guard')


def full_batch(graph,kind,anchors,resources):
    guard(graph.x,graph.edges,resources['max_nodes'],resources['max_edges'])
    return graph.x,graph.edges,torch.as_tensor(anchors),torch.as_tensor(anchors)


def neighbor_batch(graph,kind,anchors,resources):
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.typing import WITH_PYG_LIB,WITH_TORCH_SPARSE
    if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
        raise RuntimeError('Neighbor backend unavailable; laboratory operation probe required by T030')
    data=HeteroData()
    for t in TYPES:data[t].x=graph.x[t]
    for name,(src,dst) in RELATIONS.items():data[src,name,dst].edge_index=graph.edges[name]
    seeds=torch.as_tensor(anchors,dtype=torch.long)
    loader=NeighborLoader(data,num_neighbors={k:resources['fanouts'] for k in data.edge_types},
                          input_nodes=(kind,seeds),batch_size=len(seeds),shuffle=False,num_workers=resources['workers'])
    sampled=next(iter(loader))
    return paired_sample(graph,sampled,kind,resources)


def paired_sample(graph,sampled,kind,resources):
    x={t:sampled[t].x for t in TYPES};maps={t:{int(g):i for i,g in enumerate(sampled[t].n_id)} for t in TYPES}
    edges={}
    for role in ('sender','receiver'):
        selected=[]
        for name in (role,role+'_reverse'):
            src,dst=RELATIONS[name];store=sampled[src,name,dst]
            if 'e_id' in store:selected.extend(store.e_id.tolist())
        physical=graph.edges[role][:,sorted(set(selected))]
        src,dst=RELATIONS[role]
        pairs=[(maps[src][int(a)],maps[dst][int(b)]) for a,b in physical.T]
        edges[role]=torch.tensor(pairs,dtype=torch.long).reshape(-1,2).T.contiguous()
        edges[role+'_reverse']=edges[role].flip(0)
    guard(x,edges,resources['max_nodes'],resources['max_edges'])
    count=sampled[kind].batch_size
    return x,edges,torch.arange(count),sampled[kind].n_id[:count]


def batch(graph,kind,anchors,resources):
    loader=full_batch if resources['loading']=='full' else neighbor_batch
    return loader(graph,kind,anchors,resources)

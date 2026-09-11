"""Small reproducible CPU operation probe; no data access or installation."""
from __future__ import annotations
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path


def doctor(project_root: Path) -> dict:
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GINConv
    expected = {"torch": "2.6.0", "torch-geometric": "2.6.1", "scikit-learn": "1.6.1"}
    versions = {n: metadata.version(n) for n in (*expected, "numpy", "polars", "pyarrow", "PyYAML", "pytest")}
    if sys.version_info[:2] != (3, 11) or any(versions[n] != v for n, v in expected.items()):
        raise ValueError("Environment differs from the accepted Python/core version pins")
    lock = project_root / "requirements/mac-cpu.lock"
    if not lock.is_file():
        raise ValueError(f"Missing lock: {lock}")
    # fork_rng keeps this diagnostic from changing a caller's random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        g = HeteroData()
        g["address"].x = torch.randn(3, 4, requires_grad=True)
        g["transaction"].x = torch.randn(2, 4, requires_grad=True)
        relation = ("address", "sent_to", "transaction")
        g[relation].edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
        g.validate(raise_on_error=True)
        model = GINConv(torch.nn.Linear(4, 4))
        out = model((g["address"].x, g["transaction"].x), g[relation].edge_index)
        loss = out.square().mean()
        loss.backward()
        tensors = [g["address"].x, g["transaction"].x, *model.parameters()]
        if not torch.isfinite(out).all() or any(
            t.grad is None or not torch.isfinite(t.grad).all() or t.grad.abs().sum() == 0
            for t in tensors
        ):
            raise RuntimeError("CPU graph gradient/finite-value probe failed")
    return {"status": "PASS", "device": "cpu", "python": sys.version,
            "platform": platform.platform(), "versions": versions,
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "loss": loss.item(), "scope": "CPU bipartite GIN forward/backward; not full pipeline or CUDA validation"}


if __name__ == "__main__":
    print(json.dumps(doctor(Path(__file__).resolve().parents[2]), indent=2))


def operation_probe(resources, hidden=32):
    """Exercise the same adapter/relations as training, with anchors of both types."""
    import numpy as np
    import torch
    from hgcl.data.sampling import Graph, batch
    from hgcl.models.encoder import Encoder, TYPES
    device=resources['device']
    cuda_devices=[torch.cuda.current_device()] if device=='cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(11)
        x={'address':torch.randn(12,5),'transaction':torch.randn(6,4)}
        edges={'sender':torch.stack([torch.arange(12),torch.arange(12)%6]),
               'receiver':torch.stack([torch.arange(12)%6,(torch.arange(12)+3)%12])}
        for role in ('sender','receiver'):edges[role+'_reverse']=edges[role].flip(0)
        graph=Graph(1,x,edges,{}, {},np.zeros((12,5)))
        model=Encoder(5,4,hidden,2).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        reports={}
        for kind in TYPES:
            features,sampled,local,canonical=batch(graph,kind,torch.tensor([0,1]),resources)
            if canonical.tolist()!=[0,1]:raise RuntimeError('Sampler lost anchor identity/order')
            for role in ('sender','receiver'):
                if not torch.equal(sampled[role].flip(0),sampled[role+'_reverse']):raise RuntimeError('Sampler reverse pairing mismatch')
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            optimizer.zero_grad()
            out=model({k:v.to(device) for k,v in features.items()}, {k:v.to(device) for k,v in sampled.items()})[kind][local.to(device)]
            target=torch.randn_like(out);loss=(out-target).square().mean();loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in grads):raise RuntimeError('Nonfinite/missing probe gradients')
            optimizer.step()
            if not any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise RuntimeError('Probe model did not update')
            reports[kind]={'loss':loss.item(),'anchors':canonical.tolist(),
                           'nodes':sum(len(a) for a in features.values()),'directed_edges':sum(e.shape[1] for e in sampled.values())}
        if device=='cuda':torch.cuda.synchronize()
    return reports


def lab_doctor(config):
    import os
    import shutil
    import torch
    from hgcl.data.ingest import verify_sources
    from hgcl.provenance import file_hash,source_identity
    from torch_geometric.typing import WITH_PYG_LIB
    if config.values['resources']['device']!='cuda':raise ValueError('Laboratory probe requires CUDA; no CPU fallback')
    expected={'torch':'2.6.0+cu124','torch-geometric':'2.6.1','scikit-learn':'1.6.1','pyg-lib':'0.4.0+pt26cu124'}
    versions={name:metadata.version(name) for name in expected}
    if platform.system()!='Linux' or platform.machine()!='x86_64' or sys.version_info[:2]!=(3,11):
        raise ValueError('Expected Linux x86_64 / Python 3.11')
    if versions!=expected or torch.version.cuda!='12.4':raise ValueError(f'Unexpected laboratory binary versions: {versions}')
    if not torch.cuda.is_available() or not WITH_PYG_LIB:raise RuntimeError('CUDA or pyg-lib unavailable; no fallback')
    lock=config.root/'requirements/lab-cuda.lock';flake=config.root/'flake.lock'
    lock_hash=file_hash(lock);flake_hash=file_hash(flake)
    originals=verify_sources(config.data_root,config.root/config.values['provenance']['source_manifest'])
    config.artifacts_root.mkdir(parents=True,exist_ok=True)
    free_disk=shutil.disk_usage(config.artifacts_root).free
    if free_disk<10*1024**3:raise RuntimeError('At least 10 GiB free disk required for initial preparation; monitor during experiment')
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    properties=torch.cuda.get_device_properties(0);free_gpu,total_gpu=torch.cuda.mem_get_info()
    resources=config.values['resources']
    if memory['MemAvailable']<resources['rss_gib']*1024**3:raise RuntimeError('Available RAM below configured RSS budget')
    if free_gpu<resources['gpu_gib']*1024**3:raise RuntimeError('Available GPU memory below configured GPU budget')
    before=source_identity(config.root)['sha256']
    torch.cuda.reset_peak_memory_stats()
    operations=operation_probe(resources,config.values['encoder']['hidden'])
    peak=torch.cuda.max_memory_allocated()
    if peak>resources['gpu_gib']*1024**3:raise RuntimeError('Probe exceeded GPU budget')
    if source_identity(config.root)['sha256']!=before:raise RuntimeError('Source changed during probe')
    return {'status':'PASS','scope':'CUDA sampled bipartite encoder forward/backward and original hashes; not full matrix acceptance',
            'device':'cuda','python':sys.version,'platform':platform.platform(),'versions':versions,
            'cuda_runtime':torch.version.cuda,'gpu':properties.name,'gpu_total_bytes':total_gpu,
            'gpu_free_bytes':free_gpu,'gpu_peak_allocated_bytes':peak,'memory':memory,'disk_free_bytes':free_disk,
            'source_sha256':before,'lock_sha256':lock_hash,'flake_lock_sha256':flake_hash,
            'original_files':len(originals),'operations':operations}

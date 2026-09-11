"""Atomic trusted-local checkpoints including all replay random streams."""
from pathlib import Path
import os
import random
import tempfile
import numpy as np
import torch


def save(path,model,optimizer,epoch,dependencies,*,generators=None,extra=None,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():raise ValueError('Refusing to overwrite immutable SSL checkpoint')
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'dependencies':dependencies,
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
           'generators':{k:g.get_state() for k,g in (generators or {}).items()},'extra':extra or {}}
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def load(path,model,optimizer,dependencies,*,generators=None):
    # Only checkpoints created locally by this pipeline are accepted here.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['dependencies']!=dependencies:raise ValueError('Incompatible checkpoint dependencies')
    if set(state['generators'])!=set(generators or {}):raise ValueError('Checkpoint random-stream mismatch')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    for k,g in (generators or {}).items():g.set_state(state['generators'][k])
    return state

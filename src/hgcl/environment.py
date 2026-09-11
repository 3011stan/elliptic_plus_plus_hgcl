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

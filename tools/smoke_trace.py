"""Smoke test: produce a small CUDA torch.profiler trace for TraceLens.

Run on 10.51.198.60 inside /model/wyy/optframe/.venv.
Keeps GPU footprint small (~a few hundred MB) so it can coexist with a live model.
"""
import os
import sys
import torch
from torch.profiler import ProfilerActivity, profile

OUT = sys.argv[1] if len(sys.argv) > 1 else "/model/wyy/optframe/runs/smoke/trace.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

torch.manual_seed(0)
dev = "cuda"
d = 2048
x = torch.randn(d, d, device=dev, dtype=torch.bfloat16)
w = torch.randn(d, d, device=dev, dtype=torch.bfloat16)
g = torch.randn(d, device=dev, dtype=torch.bfloat16)


def step():
    h = torch.mm(x, w)
    h = torch.nn.functional.silu(h)
    h = h * g
    return torch.softmax(h, dim=-1)


# warmup (excluded from trace)
for _ in range(10):
    step()
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
    profile_memory=False,
) as prof:
    for _ in range(30):
        step()
    torch.cuda.synchronize()

prof.export_chrome_trace(OUT)
print(f"trace written: {OUT} ({os.path.getsize(OUT)} bytes)")

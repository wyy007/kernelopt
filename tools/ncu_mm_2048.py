import torch
torch.manual_seed(0)
d = 2048
a = torch.randn(d, d, device="cuda", dtype=torch.bfloat16)
b = torch.randn(d, d, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    c = torch.mm(a, b)
torch.cuda.synchronize()
print("done", float(c[0, 0]))

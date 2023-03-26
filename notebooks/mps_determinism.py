import torch

torch.use_deterministic_algorithms(True)
device = torch.device('mps')

idx = torch.tensor([
    0, 0, 0, 0, 2, 2, 1, 0, 2, 1, 0, 1, 2, 1, 0, 2, 2, 2, 2, 2, 0, 0, 2, 1, 2, 1, 0, 0, 2, 0, 2, 1, 1, 2, 2, 0, 2, 1, 0, 2
], device=device)
# t_new = torch.arange(len(idx), device=device)

for _ in range(5):
    t = torch.zeros(3, dtype=torch.long, device=device)
    t[idx] = torch.arange(len(idx), device=device)
    print(t)
    
for _ in range(5):
    t = torch.zeros(3, dtype=torch.long)
    t[idx.cpu()] = torch.arange(len(idx))
    print(t)
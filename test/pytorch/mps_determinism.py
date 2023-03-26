import torch

torch.use_determinstic_algorithms(True)
device = torch.device('mps')

idx = torch.tensor([0, 2, 1, 2, 0, 1], device=device)
for _ in range(10):
    t = torch.zeros(3, dtype=torch.long, device=device)
    t[idx] = torch.arange(len(idx), device=device)
    print(t)
import torch

device = torch.device('mps')

t = torch.zeros(5, device=device)
u = torch.ones(2, device=device).long()
t[2:4] = u
print(t)

t = torch.zeros(5, device=device)
u = torch.tensor(2)
t[2:4] = u
print(t)

t = torch.tensor([0., torch.nan, 2., torch.nan, 2., 1., 0., 1., 2., 0.])
print(t.unique(sorted=True, dim=None, return_counts=True))
print(t.unique(sorted=True, dim=0))

t = torch.tensor([0., torch.nan, 2., torch.nan, 2., 1., 0., 1., 2., 0.], device=device)
print(t.unique(sorted=True, dim=None, return_counts=True))
# print(t.unique(sorted=True, dim=0))

t = torch.ones((2,6,), device=device)[1].reshape(2,3)
print("No error yet")
t = t + 1
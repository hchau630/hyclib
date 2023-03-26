import timeit

import torch

t = torch.randint(10, size=(100000,), device='mps')
t = t[t.argsort()]
t_cpu = t.cpu()

number = 10

print(timeit.timeit('t.unique()', globals=globals(), number=number))
print(timeit.timeit('t.unique(return_inverse=True)', globals=globals(), number=number))
print(timeit.timeit('t.unique(return_counts=True)', globals=globals(), number=number))
print(timeit.timeit('t_cpu.unique()', globals=globals(), number=number))
print(timeit.timeit('t_cpu.unique(return_inverse=True)', globals=globals(), number=number))
print(timeit.timeit('t_cpu.unique(return_counts=True)', globals=globals(), number=number))
print()
print(timeit.timeit('t.unique_consecutive()', globals=globals(), number=number))
print(timeit.timeit('t.unique_consecutive(return_inverse=True)', globals=globals(), number=number))
print(timeit.timeit('t.unique_consecutive(return_counts=True)', globals=globals(), number=number))
print(timeit.timeit('t_cpu.unique_consecutive()', globals=globals(), number=number))
print(timeit.timeit('t_cpu.unique_consecutive(return_inverse=True)', globals=globals(), number=number))
print(timeit.timeit('t_cpu.unique_consecutive(return_counts=True)', globals=globals(), number=number))
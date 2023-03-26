import argparse

import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', default=10, type=int, help='Length of tensor being indexed. (default: 10)')
    parser.add_argument('-M', default=3001, type=int, help='Length of index tensor. Unexpected behavior occurs when M > 3000. (default: 3001)')
    parser.add_argument('-d', '--deterministic', action='store_true', help='If enabled, uses torch deterministic algorithm.')
    args = parser.parse_args()
    
    torch.use_deterministic_algorithms(args.deterministic)
    N, M = args.N, args.M
    
    idx = torch.randint(N, size=(M,))
    value = torch.normal(mean=0.0, std=1.0, size=(M,))

    expected = torch.zeros(N)
    for i in range(N):
        v = value[idx == i]
        if v.numel() > 0:
            expected[i] = v[-1]

    ts, arrs = [], []
    N_trials = 1000
    for _ in range(N_trials):
        t = torch.zeros(N)
        t[idx] = value
        arr = np.zeros(N)
        arr[idx.numpy()] = value.numpy()

        ts.append(t)
        arrs.append(arr)

    ts = torch.stack(ts, dim=0)
    arrs = np.stack(arrs, axis=0)

    print(f"Number of incorrect numpy arrays: {(~(arrs == expected.numpy()).all(axis=1)).sum()}/{N_trials}")
    print(f"Number of incorrect torch tensors: {(~(ts == expected).all(dim=1)).sum()}/{N_trials}")
    
if __name__ == '__main__':
    main()
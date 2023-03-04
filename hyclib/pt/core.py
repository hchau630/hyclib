import torch
import numpy as np

from ..itertools import flatten_seq

__all__ = ['meshgrid_dd', 'bincount']

def meshgrid_dd(tensors):
    """
    Pytorch version of a generalized np.meshgrid
    Mesh together list of tensors of shapes (n_1_1,...,n_1_{M_1},N_1), (n_2_1,...,n_2_{M_2},N_2), ..., (n_P_1, ..., n_P_{M_P},N_P)
    Returns tensors of shapes 
    (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,n_P_1, ..., n_P_{M_P},N_1),
    (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,n_P_1, ..., n_P_{M_P},N_2),
    ...
    (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,n_P_1, ..., n_P_{M_P},N_P)
    """
    sizes = [list(tensor.shape[:-1]) for tensor in tensors] # [[n_1,...,n_{M_1}],[n_1,...,.n_{M_2}],...]
    Ms = np.array([tensor.ndim - 1 for tensor in tensors]) # [M_1, M_2, ...]
    M_befores = np.cumsum(np.insert(Ms[:-1],0,0))
    M_afters = np.sum(Ms) - np.cumsum(Ms)
    Ns = [tensor.shape[-1] for tensor in tensors]
    shapes = [[1]*M_befores[i]+sizes[i]+[1]*M_afters[i]+[Ns[i]] for i, tensor in enumerate(tensors)]
    expanded_tensors = [tensor.reshape(shapes[i]).expand(flatten_seq(sizes)+[Ns[i]]) for i, tensor in enumerate(tensors)]
    return expanded_tensors

def bincount(indices, weights=None):
    """
    Similar to torch.bincount, but supports auto-differentiation and allows batched weights.
    Always performs bincount on the last dimension, with the leading dimensions interpreted as batch dimensions.
    
    Benchmark:
    
    N, M = 100000, 100
    a = torch.randint(M, size=(N,))
    w = torch.normal(mean=0.0, std=1.0, size=(N,))
    a_np, w_np = a.numpy(), w.numpy()

    %timeit utils.pt.bincount(a, weights=w)
    116 µs ± 377 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    
    %timeit a.bincount(weights=w)
    93.5 µs ± 187 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    
    %timeit np.bincount(a_np, weights=w_np)
    161 µs ± 727 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    """
    if indices.is_floating_point():
        raise TypeError(f"indices must be a tensor of integer dtype, but {indices.dtype=}.")
    if indices.min() < 0:
        raise ValueError(f"indices must not be negative, but {indices.min()=}.")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, but {indices.ndim=}.")
        
    if weights is None:
        weights = torch.ones(indices.shape, dtype=torch.long, device=indices.device)
    else:
        if not indices.device == weights.device:
            raise ValueError(f"indices and weights must be on the same device, but {indices.device=} and {weights.device=}.")
        indices = indices.broadcast_to(weights.shape)
        
    shape = (*weights.shape[:-1], indices.max().item()+1)
        
    t = torch.zeros(shape, dtype=weights.dtype, device=weights.device)
    t.scatter_add_(-1, indices, weights)
    
    return t
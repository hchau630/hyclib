import collections

import torch
import numpy as np

from ..itertools import flatten_seq

__all__ = ['meshgrid_dd', 'bincount', 'ravel_multi_index', 'unravel_index']

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

    %timeit lib.pt.bincount(a, weights=w)
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

def ravel_multi_index(multi_index, dims):
    """
    Similar to np.ravel_multi_index
    """
    multi_index = torch.stack(multi_index) if not isinstance(multi_index, torch.Tensor) else multi_index
    dims = torch.as_tensor(dims, device=multi_index.device)
    
    if len(multi_index) != len(dims):
        raise ValueError(f"multi_index and dims must have same length, but {len(multi_index)=} and {len(dims)=}.")
    
    if torch.is_floating_point(multi_index):
        raise TypeError(f"multi_index must be integer dtype, but {multi_index.dtype=}.")
        
    if (multi_index.min(dim=1).values < 0).any():
        raise ValueError(f"multi_index must be non-negative, but {multi_index.min(dim=1).values=}.")
        
    if (multi_index.max(dim=1).values >= dims).any():
        raise ValueError(f"multi_index must be less than dims along each dimension, but {multi_index.max(dim=1).values=} and {dims=}.")
        
    multipliers = np.cumprod((dims[1:].tolist() + [1])[::-1])[::-1]
    return torch.stack([index * multiplier for index, multiplier in zip(multi_index, multipliers)], dim=0).sum(dim=0)

def unravel_index(indices, shape, *, as_tuple=True):
    r"""
    Modified from https://github.com/pytorch/pytorch/pull/66687/files
    
    Converts a `Tensor` of flat indices into a `Tensor` of coordinates for the given target shape.
    Args:
        indices: An integral `Tensor` containing flattened indices of a `Tensor` of dimension `shape`.
        shape: The shape (can be an `int`, a `Sequence` or a `Tensor`) of the `Tensor` for which
               the flattened `indices` are unraveled.
    Keyword Args:
        as_tuple: A boolean value, which if `True` will return the result as tuple of Tensors,
                  else a `Tensor` will be returned. Default: `True`
    Returns:
        unraveled coordinates from the given `indices` and `shape`. See description of `as_tuple` for
        returning a `tuple`.
    .. note:: The default behaviour of this function is analogous to
              `numpy.unravel_index <https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html>`_.
    Example::
        >>> indices = torch.tensor([22, 41, 37])
        >>> shape = (7, 6)
        >>> torch.unravel_index(indices, shape)
        (tensor([3, 6, 6]), tensor([4, 5, 1]))
        >>> torch.unravel_index(indices, shape, as_tuple=False)
        tensor([[3, 4],
                [6, 5],
                [6, 1]])
        >>> indices = torch.tensor([3, 10, 12])
        >>> shape_ = (4, 2, 3)
        >>> torch.unravel_index(indices, shape_)
        (tensor([0, 1, 2]), tensor([1, 1, 0]), tensor([0, 1, 0]))
        >>> torch.unravel_index(indices, shape_, as_tuple=False)
        tensor([[0, 1, 0],
                [1, 1, 1],
                [2, 0, 0]])
    """
    def _helper_type_check(inp, name):
        # `indices` is expected to be a tensor, while `shape` can be a sequence/int/tensor
        if name == "shape" and isinstance(inp, collections.abc.Sequence):
            for dim in inp:
                if not isinstance(dim, int):
                    raise TypeError("Expected shape to have only integral elements.")
                if dim < 0:
                    raise ValueError("Negative values in shape are not allowed.")
        elif name == "shape" and isinstance(inp, int):
            if inp < 0:
                raise ValueError("Negative values in shape are not allowed.")
        elif isinstance(inp, torch.Tensor):
            if torch.is_floating_point(inp):
                raise TypeError(f"Expected {name} to be an integral tensor, but found dtype: {inp.dtype}")
            if torch.any(inp < 0):
                raise ValueError(f"Negative values in {name} are not allowed.")
        else:
            allowed_types = "Sequence/Scalar (int)/Tensor" if name == "shape" else "Tensor"
            msg = f"{name} should either be a {allowed_types}, but found {type(inp)}"
            raise TypeError(msg)

    _helper_type_check(indices, "indices")
    _helper_type_check(shape, "shape")

    # Convert to a tensor, with the same properties as that of indices
    if isinstance(shape, collections.abc.Sequence):
        shape_tensor = indices.new_tensor(shape)
    elif isinstance(shape, int) or (isinstance(shape, Tensor) and shape.ndim == 0):
        shape_tensor = indices.new_tensor((shape,))
    else:
        shape_tensor = shape

    # By this time, shape tensor will have dim = 1 if it was passed as scalar (see if-elif above)
    assert shape_tensor.ndim == 1, "Expected dimension of shape tensor to be <= 1, "
    f"but got the tensor with dim: {shape_tensor.ndim}."

    # In case no indices passed, return an empty tensor with number of elements = shape.numel()
    if indices.numel() == 0:
        # If both indices.numel() == 0 and shape.numel() == 0, short-circuit to return itself
        shape_numel = shape_tensor.numel()
        if shape_numel == 0:
            raise ValueError("Got indices and shape as empty tensors, expected non-empty tensors.")
        else:
            output = [indices.new_tensor([]) for _ in range(shape_numel)]
            return tuple(output) if as_tuple else torch.stack(output, dim=1)

    if torch.max(indices) >= torch.prod(shape_tensor):
        raise ValueError("Target shape should cover all source indices.")

    coefs = shape_tensor[1:].flipud().cumprod(dim=0).flipud()
    coefs = torch.cat((coefs, coefs.new_tensor((1,))), dim=0)
    coords = torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape_tensor

    if as_tuple:
        return tuple(coords[..., i] for i in range(coords.size(-1)))
    return coords
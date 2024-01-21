import itertools
import collections

import torch
import numpy as np
import pytest

import hyclib as lib

@pytest.mark.parametrize('x, dim, expected', [
    (torch.tensor(2), None, True),
    (torch.tensor(2), (), torch.tensor(True)),
    (torch.tensor([]), None, True),
    (torch.tensor([]), 0, True),
    (torch.tensor([1.0]), None, True),
    (torch.tensor([1.0]), 0, True),
    (torch.tensor([1.0]), (), torch.tensor([True])),
    (torch.tensor([[1.0]]), None, True),
    (torch.tensor([[1.0]]), 0, True),
    (torch.tensor([[1.0]]), 1, True),
    (torch.tensor([[1.0]]), (), torch.tensor([[True]])),
    (torch.tensor([[1.0,1.0,1.0]]), None, True),
    (torch.tensor([[1.0,0.5,1.0]]), None, False),
    (torch.tensor([[1.0,0.5,1.0]]), (), torch.tensor([[True, True, True]])),
    (torch.tensor([[1.0,1.0,1.0],
                   [2.0,2.0,2.0]]), None, False),
    (torch.tensor([[1.0,1.0,1.0],
                   [2.0,2.0,2.0]]), 0, False),
    (torch.tensor([[1.0,1.0,1.0],
                   [2.0,2.0,2.0]]), 1, True),
    (torch.tensor([[1.0,1.0,1.0],
                   [2.0,2.0,2.0]]), (0,1), False),
    (
        torch.tensor([[1.0,1.0,1.0],
                      [2.0,2.0,2.0]]),
        (),
        torch.tensor([[True,True,True],
                      [True,True,True]])
    ),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), None, False),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), 0, True),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), 1, False),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), -1, True),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (0,1), False),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (1,0), False),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (0,2), True),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (2,0), True),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (0,-1), True),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (-1,0), True),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (1,2), False),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (2,1), False),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (-1,-2), False),
    (torch.tensor([[[1.0,1.0,1.0],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,2.0,2.0]]]), (-2,-1), False),
    (torch.tensor([[[1.0,1.0,0.5],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,0.5,2.0]]]),
     0,
     torch.tensor([[True, True, False],
                   [True, False, True]])),
    (torch.tensor([[[1.0,1.0,0.5],
                    [1.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,1.0,2.0]]]),
     1,
     torch.tensor([[True, False, False],
                   [False, True, False]])),
    (torch.tensor([[[1.0,1.0,0.5],
                    [2.0,2.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,1.0,2.0]]]),
     2,
     torch.tensor([[False, True],
                   [True, False]])),
    (torch.tensor([[[1.0,1.0,0.5],
                    [2.0,1.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,1.0,2.0]]]),
     (0,1),
     torch.tensor([False, True, False])),
    (torch.tensor([[[1.0,1.0,1.0],
                    [0.5,1.0,2.0]],
                   [[1.0,1.0,1.0],
                    [2.0,1.0,2.0]]]),
     (0,2),
     torch.tensor([True, False])),
    (torch.tensor([[[1.0,1.0,1.0],
                    [0.5,1.0,2.0]],
                   [[1.0,1.0,1.0],
                    [1.0,1.0,1.0]]]),
     (1,2),
     torch.tensor([False, True])),
])
def test_isconst(x, dim, expected):
    assert torch.all(lib.pt.isconst(x, dim=dim) == expected)

def get_devices():
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices

def as_tuple(out):
    return out if isinstance(out, tuple) else (out,)

@pytest.mark.parametrize('M, D, shape, O, dim', [
    (3, 2, (100,), 0, -1),
    (3, 2, (100,), 0, 0),
    (3, 2, (100,), 20, -1),
    (3, 2, (100,), 20, 0),
    (4, 6, (99, 100), 60_000, -1),
    (4, 6, (99, 100), 6_000, -1),
    (4, 6, (99, 100), 6_000, 0),
    (4, 6, (99, 100), 6_000, 1),
])
@pytest.mark.parametrize('device', get_devices())
def test_lexsort(M, D, shape, O, dim, device):
    t = torch.randint(M, size=(D,*shape))
    if O > 0:
        t = t.float()
        indices = [torch.randint(D, size=(O,))]
        indices += [torch.randint(N, size=(O,)) for N in shape]
        t[tuple(indices)] = torch.nan
    t = t.to(device)
    a = t.cpu().numpy()
    
    pt_idx = lib.pt.lexsort(t, dim=dim)
    np_idx = np.lexsort(a, axis=dim)
    
    torch.testing.assert_close(pt_idx, torch.from_numpy(np_idx).to(device), equal_nan=True)
    
@pytest.mark.parametrize('M, shape, O, dim', [
    (3, (100, 2), 0, None),
    (3, (100, 2), 0, 0),
    (3, (100, 2), 20, None),
    (3, (100, 2), 20, 0),
    (3, (2, 100), 0, 1),
    (3, (2, 100), 20, 1),
    (3, (2, 100), 20, -1),
    (4, (10_000, 6), 60_000, None),
    (4, (10_000, 6), 6_000, None),
    (4, (10_000, 6), 6_000, 0),
    (4, (6, 10_000), 6_000, 1),
])
@pytest.mark.parametrize('return_index', [True, False])
@pytest.mark.parametrize('return_inverse', [True, False])
@pytest.mark.parametrize('return_counts', [True, False])
@pytest.mark.parametrize('sorted', [True, False])
@pytest.mark.parametrize('first_index', [True, False])
@pytest.mark.parametrize('device', get_devices())
# @pytest.mark.parametrize('device', ['cpu'])
# @pytest.mark.benchmark(
#     max_time=0.25,
# )
def test_unique(M, shape, O, dim, sorted, return_index, return_inverse, return_counts, first_index, device, benchmark):
    """
    IMPORTANT: Must have torch>=2.1.0, which fixes a bunch of MPS bugs.
    """
    kwargs = {
        'return_index': return_index,
        'return_inverse': return_inverse,
        'return_counts': return_counts,
    }
    np_kwargs = kwargs.copy()
    np_kwargs['return_index'] = True # always use return_index=True in order to have stable sort
    
    t = torch.randint(M, size=shape)
    if O > 0:
        t = t.float()
        indices = [torch.randint(D, size=(O,)) for D in shape]
        t[tuple(indices)] = torch.nan
    t = t.to(device)
    a = t.cpu().numpy()

    # torch_results = benchmark(lib.pt.unique, t, dim=dim, sorted=sorted, first_index=first_index, **kwargs)
    torch_results = lib.pt.unique(t, dim=dim, sorted=sorted, first_index=first_index, **kwargs)
    torch_results = list(as_tuple(torch_results))
    np_results = as_tuple(np.unique(t.cpu().numpy(), axis=dim, equal_nan=False, **np_kwargs))
    if not kwargs['return_index']:
        np_results = list(np_results)
        del np_results[1]

    if not sorted:
        if dim is None:
            sort_idx = torch_results[0].argsort(stable=True)
        else:
            sort_idx = lib.pt.lexsort(torch_results[0].movedim(dim, -1).flip(0))
        sort_idx_inv = lib.pt.inv_perm(sort_idx)

    keys = ['x'] + [k for k, v in kwargs.items() if v]

    for key, torch_result, np_result in zip(keys, torch_results, np_results):
        if not first_index and key == 'return_index':
            if dim is None:
                torch.testing.assert_close(torch_results[0], t.reshape(-1)[torch_result], equal_nan=True)
            else:
                torch.testing.assert_close(torch_results[0], t.index_select(dim, torch_result), equal_nan=True)
            continue # no need to check against numpy since the indices are not guaranteed to be the same due to non-determinism

        if sorted:
            torch.testing.assert_close(torch_result, torch.from_numpy(np_result).to(device), equal_nan=True)
        else:
            if key == 'return_inverse':
                torch.testing.assert_close(sort_idx_inv[torch_result], torch.from_numpy(np_result).to(device), equal_nan=True)
            else:
                if dim is None or key != 'x':
                    torch.testing.assert_close(torch_result[sort_idx], torch.from_numpy(np_result).to(device), equal_nan=True)
                else:
                    torch.testing.assert_close(torch_result.movedim(dim, 0)[sort_idx].movedim(0, dim), torch.from_numpy(np_result).to(device), equal_nan=True)

@pytest.mark.parametrize('indexing, dims, shared_shape, pre_shapes, post_shapes, inserted_dims', [
    ('ij', None, (3, 5, 2, 4, 3, 5, 7, 2, 4), None, [(), (), ()], [(3, 4, 5, 6, 7, 8), (0, 1, 2, 5, 6, 7, 8), (0, 1, 2, 3, 4)]),
    ('ij', 0, (), None, [(3, 5, 2), (4, 3), (5, 7, 2, 4)], [(), (), ()]),
    ('ij', -1, (3, 5, 4, 5, 7, 2), None, [(2,), (3,), (4,)], [(2, 3, 4, 5), (0, 1, 3, 4, 5), (0, 1, 2)]),
    ('ij', 2, (3, 5, 4, 3, 5, 7), None, [(2,), (), (2, 4)], [(2, 3, 4, 5), (0, 1, 4, 5), (0, 1, 2, 3)]),
    ('ij', [(None, 2), (None, -1), (None, 0)], (3, 5, 4), None, [(2,), (3,), (5, 7, 2, 4)], [(2,), (0, 1), (0, 1, 2)]),
    ('xy', None, (4, 3, 3, 5, 2, 5, 7, 2, 4), None, [(), (), ()], [(0, 1, 5, 6, 7, 8), (2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4)]),
    ('xy', 0, (), None, [(3, 5, 2), (4, 3), (5, 7, 2, 4)], [(), (), ()]),
    ('xy', -1, (4, 3, 5, 5, 7, 2), None, [(2,), (3,), (4,)], [(0, 3, 4, 5), (1, 2, 3, 4, 5), (0, 1, 2)]),
    ('xy', 2, (4, 3, 3, 5, 5, 7), None, [(2,), (), (2, 4)], [(0, 1, 4, 5), (2, 3, 4, 5), (0, 1, 2, 3)]),
    ('xy', [(None, 2), (None, -1), (None, 0)], (4, 3, 5), None, [(2,), (3,), (5, 7, 2, 4)], [(0,), (1, 2), (0, 1, 2)]),
    ('ij', [(0, None)] * 3, (3, 5, 2, 4, 3, 5, 7, 2, 4), [(), (), ()], None, [(3, 4, 5, 6, 7, 8), (0, 1, 2, 5, 6, 7, 8), (0, 1, 2, 3, 4)]),
    ('ij', [(-1, None)] * 3, (2, 3, 4), [(3, 5), (4,), (5, 7, 2)], None, [(-1, -2), (-1, -3), (-2, -3)]),
    ('ij', [(2, None)] * 3, (2, 2, 4), [(3, 5), (4, 3), (5, 7)], None, [(-1, -2), (-1, -2, -3), (-3,)]),
    ('ij', [(2, None), (-1, None), (0, None)], (2, 3, 5, 7, 2, 4), [(3, 5,), (4,), ()], None, [(-1, -2, -3, -4, -5), (-1, -2, -3, -4, -6), (-5, -6)]),
    ('ij', [(-2, None), (-1, None), (-1, None)], (5, 2, 3, 4), [(3,), (4,), (5, 7, 2)], None, [(-1, -2), (-1, -3, -4), (-2, -3, -4)]),
    ('xy', [(0, None)] * 3, (4, 3, 3, 5, 2, 5, 7, 2, 4), [(), (), ()], None, [(0, 1, 5, 6, 7, 8), (2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4)]),
    ('xy', [(-1, None)] * 3, (3, 2, 4), [(3, 5), (4,), (5, 7, 2)], None, [(-1, -3), (-1, -2), (-2, -3)]),
    ('xy', [(2, None)] * 3, (2, 2, 4), [(3, 5), (4, 3), (5, 7)], None, [(-1, -2), (-1, -2, -3), (-3,)]),
    ('xy', [(2, None), (-1, None), (0, None)], (3, 2, 5, 7, 2, 4), [(3, 5,), (4,), ()], None, [(-1, -2, -3, -4, -6), (-1, -2, -3, -4, -5), (-5, -6)]),
    ('xy', [(-2, None), (-1, None), (-1, None)], (3, 5, 2, 4), [(3,), (4,), (5, 7, 2)], None, [(-1, -4), (-1, -2, -3), (-2, -3, -4)]),
    ('ij', [(1, -1), (-1, None), (1, -1)], (5, 3, 7, 2), [(3,), (4,), (5,)], [(2,), (), (4,)], [(2, 3, 4), (1, 3, 4), (1, 2)]),
    ('xy', [(1, -1), (-1, None), (1, -1)], (3, 5, 7, 2), [(3,), (4,), (5,)], [(2,), (), (4,)], [(1, 3, 4), (2, 3, 4), (1, 2)]),
])
@pytest.mark.parametrize('sparse', [False, True])
def test_meshgrid(indexing, dims, sparse, shared_shape, pre_shapes, post_shapes, inserted_dims):
    if pre_shapes is None:
        pre_shapes = [()] * 3
    if post_shapes is None:
        post_shapes = [()] * 3
    
    sizes = ((3, 5, 2), (4, 3), (5, 7, 2, 4))
    in_tensors = [torch.rand(size) for size in sizes]
    out_tensors = lib.pt.meshgrid(*in_tensors, indexing=indexing, dims=dims, sparse=sparse)

    masks, indices = [], []
    for out_tensor, inserted_dim in zip(out_tensors, inserted_dims):
        mask = torch.zeros(out_tensor.ndim).bool()
        mask[list(inserted_dim)] = True
        index = np.array([slice(None) for _ in range(out_tensor.ndim)])
        index[mask.numpy()] = 0
        masks.append(mask)
        indices.append(tuple(index))

    for out_tensor, pre_shape, post_shape, mask in zip(out_tensors, pre_shapes, post_shapes, masks):
        expected_shape = pre_shape + shared_shape + post_shape
        if sparse:
            expected_shape = torch.tensor(expected_shape)
            expected_shape[mask] = 1
            expected_shape = tuple(expected_shape.tolist())
        assert out_tensor.shape == expected_shape
            
    assert all(lib.pt.isconst(out_tensor, dim=inserted_dim).all() for out_tensor, inserted_dim in zip(out_tensors, inserted_dims))
    assert all((out_tensor[index] == in_tensor).all() for in_tensor, out_tensor, index in zip(in_tensors, out_tensors, indices))

@pytest.mark.parametrize('dims', [
    0.5,
    (0, 1, 2),
    [(0, 1), (0, 1)],
    [(0, 1, 2), (0, 1, 2), (0, 1, 2)],
])
def test_meshgrid_invalid(dims):
    sizes = ((3, 5, 2), (4, 3), (5, 7, 2, 4))
    in_tensors = [torch.rand(size) for size in sizes]
    with pytest.raises(TypeError):
        lib.pt.meshgrid(*in_tensors, dims=dims)
    
@pytest.mark.parametrize('indexing', ['ij', 'xy'])
def test_meshgrid_1d(indexing):
    a, b, c = torch.rand((3,)), torch.rand((4,)), torch.rand((5,))
    a1, b1, c1 = lib.pt.meshgrid(a, b, c, indexing=indexing)
    a2, b2, c2 = torch.meshgrid(a, b, c, indexing=indexing)
    torch.testing.assert_close(a1, a2)
    torch.testing.assert_close(b1, b2)
    torch.testing.assert_close(c1, c2)

@pytest.mark.parametrize('dtype', [torch.long, torch.float])
@pytest.mark.parametrize('device', get_devices())
def test_repeat_interleave(dtype, device):
    tensor = torch.tensor([0, 5, 2, 10, 11, 20, 21, 22, 23], dtype=dtype, device=device)

    chunks = torch.tensor([3, 2, 4], device=device)
    repeats = torch.tensor([1, 3, 2], device=device)
    
    out = lib.pt.repeat_interleave(tensor, repeats, chunks=chunks)
    expected = torch.tensor([0, 5, 2, 10, 11, 10, 11, 10, 11, 20, 21, 22, 23, 20, 21, 22, 23], dtype=dtype, device=device)

    torch.testing.assert_close(out, expected)

@pytest.mark.parametrize('dtype', [torch.long, torch.float])
@pytest.mark.parametrize('device', get_devices())
def test_repeat_interleave_empty(dtype, device):
    """Fix bug in edge case where everything is zero-length."""
    tensor = torch.tensor([], dtype=dtype, device=device)

    chunks = torch.tensor([], dtype=torch.long, device=device)
    repeats = torch.tensor([], dtype=torch.long, device=device)
    
    out = lib.pt.repeat_interleave(tensor, repeats, chunks=chunks)
    expected = torch.tensor([], dtype=dtype, device=device)

    torch.testing.assert_close(out, expected)

@pytest.mark.xfail(reason='0 repeats currently not supported')
@pytest.mark.parametrize('dtype', [torch.long, torch.float])
@pytest.mark.parametrize('device', get_devices())
@pytest.mark.parametrize('repeats, expected', [
    ([3, 0, 2], [0, 5, 2, 0, 5, 2, 0, 5, 2, 20, 21, 22, 23, 20, 21, 22, 23]),
    ([0, 3, 2], [10, 11, 10, 11, 10, 11, 20, 21, 22, 23, 20, 21, 22, 23]),
    ([3, 2, 0], [0, 5, 2, 0, 5, 2, 0, 5, 2, 10, 11, 10, 11]),
])
def test_repeat_interleave_zero_repeat(repeats, expected, dtype, device):
    """Fix bug in edge case where there are zeros in repeats."""
    tensor = torch.tensor([0, 5, 2, 10, 11, 20, 21, 22, 23], dtype=dtype, device=device)

    chunks = torch.tensor([3, 2, 4], device=device)
    repeats = torch.tensor(repeats, device=device)
    
    out = lib.pt.repeat_interleave(tensor, repeats, chunks=chunks)
    expected = torch.tensor(expected, dtype=dtype, device=device)

    torch.testing.assert_close(out, expected)

@pytest.mark.parametrize('multi_index, dims, expected', [
    ([[3, 6, 6], [4, 5, 1]], (7, 6), [22, 41, 37]),
])
@pytest.mark.parametrize('as_tuple', [True, False])
@pytest.mark.parametrize('device', get_devices())
def test_ravel_multi_index(multi_index, dims, expected, as_tuple, device):
    multi_index = torch.tensor(multi_index, device=device)
    if as_tuple:
        multi_index = tuple(multi_index)
    out = lib.pt.ravel_multi_index(multi_index, dims)
    expected = torch.tensor(expected, device=device)
    torch.testing.assert_close(out, expected)

@pytest.mark.parametrize('multi_index, dims, err', [
    ([[3, 6, -1], [4, 5, 1]], (7, 6), ValueError),
    ([[3, 6, 7], [4, 5, 1]], (7, 6), ValueError),
    ([[3, 6, 6.0], [4, 5, 1]], (7, 6), TypeError),
    ([[3, 6, 6], [4, 5, 1]], (7, 6.0), TypeError),
    ([[4, 5, 1]], (7, 6), ValueError),
    ([[3, 6, 6], [4, 5, 1]], {7, 6}, TypeError),
])
@pytest.mark.parametrize('as_tuple', [True, False])
def test_ravel_multi_index_error(multi_index, dims, as_tuple, err):
    multi_index = torch.tensor(multi_index)
    if as_tuple:
        multi_index = tuple(multi_index)
    with pytest.raises(err):
        lib.pt.ravel_multi_index(multi_index, dims)

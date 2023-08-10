import torch
import numpy as np
import pytest

import hyclib as lib

@pytest.mark.parametrize('x, dim, expected', [
    (torch.tensor(2), None, True),
    (torch.tensor([]), None, True),
    (torch.tensor([]), 0, True),
    (torch.tensor([1.0]), None, True),
    (torch.tensor([1.0]), 0, True),
    (torch.tensor([[1.0]]), None, True),
    (torch.tensor([[1.0]]), 0, True),
    (torch.tensor([[1.0]]), 1, True),
    (torch.tensor([[1.0,1.0,1.0]]), None, True),
    (torch.tensor([[1.0,0.5,1.0]]), None, False),
    (torch.tensor([[1.0,1.0,1.0],
                   [2.0,2.0,2.0]]), None, False),
    (torch.tensor([[1.0,1.0,1.0],
                   [2.0,2.0,2.0]]), 0, False),
    (torch.tensor([[1.0,1.0,1.0],
                   [2.0,2.0,2.0]]), 1, True),
    (torch.tensor([[1.0,1.0,1.0],
                   [2.0,2.0,2.0]]), (0,1), False),
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
def test_unique(M, shape, O, dim, sorted, return_index, return_inverse, return_counts, first_index, device):
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

    torch_results = [out_i for out_i in as_tuple(lib.pt.unique(t, dim=dim, sorted=sorted, first_index=first_index, **kwargs))]
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

                    
def test_meshgrid_dd():
    a, b, c = torch.normal(0.0, 1.0, size=(3,5,2)), torch.normal(0.0, 1.0, size=(4,3)), torch.normal(0.0, 1.0, size=(5,7,2,4))
    a_, b_, c_ = lib.pt.meshgrid_dd(a, b, c)
    assert a_.shape == (3,5,4,5,7,2,2) and b_.shape == (3,5,4,5,7,2,3) and c_.shape == (3,5,4,5,7,2,4)
    assert lib.pt.isconst(a_, dim=(2,3,4,5)).all() and (a_[:,:,0,0,0,0,:] == a).all()
    assert lib.pt.isconst(b_, dim=(0,1,3,4,5)).all() and (b_[0,0,:,0,0,0,:] == b).all()
    assert lib.pt.isconst(c_, dim=(0,1,2)).all() and (c_[0,0,0,:,:,:,:] == c).all()
    
    
def test_meshgrid():
    a, b, c = torch.normal(0.0, 1.0, size=(3,5)), torch.normal(0.0, 1.0, size=(4,)), torch.normal(0.0, 1.0, size=(5,7,2))
    a_, b_, c_ = lib.pt.meshgrid(a, b, c)
    assert a_.shape == (3,5,4,5,7,2) and b_.shape == (3,5,4,5,7,2) and c_.shape == (3,5,4,5,7,2)
    assert lib.pt.isconst(a_, dim=(2,3,4,5)).all() and (a_[:,:,0,0,0,0] == a).all()
    assert lib.pt.isconst(b_, dim=(0,1,3,4,5)).all() and (b_[0,0,:,0,0,0] == b).all()
    assert lib.pt.isconst(c_, dim=(0,1,2)).all() and (c_[0,0,0,:,:,:] == c).all()
    
    a, b, c = torch.normal(0.0, 1.0, size=(3,)), torch.normal(0.0, 1.0, size=(4,)), torch.normal(0.0, 1.0, size=(5,))
    a1, b1, c1 = lib.pt.meshgrid(a, b, c, indexing='ij')
    a2, b2, c2 = torch.meshgrid(a, b, c, indexing='ij')
    torch.testing.assert_close(a1, a2)
    torch.testing.assert_close(b1, b2)
    torch.testing.assert_close(c1, c2)
    
    a, b, c = torch.normal(0.0, 1.0, size=(3,)), torch.normal(0.0, 1.0, size=(4,)), torch.normal(0.0, 1.0, size=(5,))
    a1, b1, c1 = lib.pt.meshgrid(a, b, c, indexing='xy')
    a2, b2, c2 = torch.meshgrid(a, b, c, indexing='xy')
    torch.testing.assert_close(a1, a2)
    torch.testing.assert_close(b1, b2)
    torch.testing.assert_close(c1, c2)
from contextlib import nullcontext as does_not_raise

import pytest
import torch
import numpy as np

import hyclib as lib

@pytest.fixture
def x():
    x = np.array(
        [[[ 0.03476371,  0.17975505, -0.54908644, -0.64903577],
          [-0.03724603,  1.81708173, -0.36515959,      np.inf],
          [ 0.41836838,     -np.inf, -0.49636453,  0.18997459]],

         [[ 1.466666  ,  1.0255607 ,      np.nan, -1.31296034],
          [ 1.08370192, -0.09796155, -0.05370452, -0.34803141],
          [ 0.13195273, -1.43293285,  0.82484756,  0.83380857]]]
    )
    return x

@pytest.mark.parametrize('dim', [
    None,
    0,
    1,
    2,
])
@pytest.mark.parametrize('func_name', [
    'nanmax',
    'nanmin',
])
def test_nan_stats(x, dim, func_name):
    result = getattr(lib.pt.stats, func_name)(torch.from_numpy(x), dim=dim)
    expected = getattr(np, func_name)(x, axis=dim)
    
    if dim is None:
        result = result.item()
        expected = expected.item()
        np.testing.assert_allclose(result, expected)
    else:
        result = result.values
        expected = torch.from_numpy(expected)
        torch.testing.assert_close(result, expected)
        
@pytest.mark.parametrize('D, bins', [
    (1, 100),
    (2, 10),
    (2, (20,5)),
    (2, ([-2.5,-2,-1,0,1,2,3.0],5)),
    (2, ([-2.5,-2,-1,0,1,2,3.0],[-2.5,-2,-1,0,1,2,3.0])),
])
@pytest.mark.parametrize('nan_policy, expectation', [
    ('raise', pytest.raises(ValueError)),
    ('omit', does_not_raise()),
])
def test_bin_dd(D, bins, nan_policy, expectation):
    M, N = 100, 100000

    arr = np.random.normal(size=(N,D))
    indices_0 = np.random.randint(N, size=M)
    indices_1 = np.random.randint(D, size=M)
    arr[indices_0, indices_1] = np.nan

    with expectation:
        binnumbers, centers, edges = lib.sp.stats.bin_dd(arr, bins=bins, nan_policy=nan_policy)

    t = torch.tensor(arr)
    with expectation as exc_info:
        tbinnumbers, tcenters, tedges = lib.pt.stats.bin_dd(t, bins=bins, nan_policy=nan_policy)

    if exc_info is None:
        torch.testing.assert_close(torch.from_numpy(binnumbers), tbinnumbers)
        for i in range(D):
            torch.testing.assert_close(torch.from_numpy(centers[i]), tcenters[i], equal_nan=True)
            torch.testing.assert_close(torch.from_numpy(edges[i]), tedges[i])


@pytest.mark.parametrize('bins', [
    100,
    [-2.5,-2,-1,0,1,2,3.0],
])
@pytest.mark.parametrize('nan_policy, expectation', [
    ('raise', pytest.raises(ValueError)),
    ('omit', does_not_raise()),
])
def test_bin(bins, nan_policy, expectation):
    M, N = 100, 100000

    arr = np.random.normal(size=N)
    indices = np.random.randint(N, size=M)
    arr[indices] = np.nan

    with expectation:
        binnumbers, centers, edges = lib.sp.stats.bin(arr, bins=bins, nan_policy=nan_policy)

    t = torch.tensor(arr)
    with expectation as exc_info:
        tbinnumbers, tcenters, tedges = lib.pt.stats.bin(t, bins=bins, nan_policy=nan_policy)

    if exc_info is None:
        torch.testing.assert_close(torch.from_numpy(binnumbers), tbinnumbers)
        torch.testing.assert_close(torch.from_numpy(centers), tcenters, equal_nan=True)
        torch.testing.assert_close(torch.from_numpy(edges), tedges)

def test_bin_precision():
    """
    Previous there was a precision issue with torch.linspace(..., device='mps') that causes the rightmost point
    to be classified outside of the max bin instead of in the max bin.
    """
    for _ in range(500):
        t = torch.normal(mean=0, std=1, size=(10,), device='mps')
        out_1 = lib.pt.stats.bin(t.cpu(), bins=3)[0]
        out_2 = lib.pt.stats.bin(t, bins=3)[0].cpu()
        out_3 = lib.sp.stats.bin(t.cpu().numpy(), bins=3)[0]
        assert (out_1 == out_2.cpu()).all() and (out_1.numpy() == out_3).all()

@pytest.mark.parametrize('w', [
    torch.tensor([[ 0.6455,  torch.nan, -0.9379, -0.0359, -0.1765, -0.3408, -1.9616,  0.1720,
                    0.4691, -0.5099],
                  [ 1.2700,  0.1971, -0.8720, -0.6560,  1.4493,  0.3326, -0.9554,  0.7140,
                   -1.5305, -1.1244]], requires_grad=True),
    torch.tensor([[ 0.6455,  torch.nan, -0.9379, -0.0359, -0.1765, -0.3408, -1.9616,  0.1720,
                    0.4691, -0.5099],
                  [ 1.2700,  0.1971, -0.8720, -0.6560,  1.4493,  0.3326, -0.9554,  0.7140,
                   -1.5305, -1.1244]]),
    torch.tensor([ 0.6455,  torch.nan, -0.9379, -0.0359, -0.1765, -0.3408, -1.9616,  0.1720,
                   0.4691, -0.5099]),
])
@pytest.mark.parametrize('minlength', [0, 10, 15])
def test_bincount(w, minlength):
    a = torch.tensor([0,0,1,1,1,0,2,1,2,0])
    t = lib.pt.bincount(a, weights=w, minlength=minlength)
    
    if w.ndim == 2:
        arr = []
        for wi in w:
            arr.append(np.bincount(a.numpy(), weights=wi.detach().numpy(), minlength=minlength))
        arr = np.stack(arr)
    elif w.ndim == 1:
        arr = np.bincount(a.numpy(), weights=w.detach().numpy(), minlength=minlength)
    else:
        raise RuntimeError()
    
    assert np.allclose(t.detach().numpy(), arr, equal_nan=True)
    
def test_bincount_nan_policy():
    x = torch.tensor([1,3,5,4,3,2,4,5,3])
    weights = torch.tensor([1.5,2.5,np.nan,0.5,-1.0,np.nan,0.5,np.nan,np.nan])
    
    torch.testing.assert_close(
        lib.pt.bincount(x, weights=weights, nan_policy='omit'),
        torch.tensor([0.0, 1.5, 0.0, 1.5, 1.0, 0.0]),
    )
    torch.testing.assert_close(
        lib.pt.bincount(x, weights=weights, nan_policy='propagate'),
        torch.tensor([0.0, 1.5, np.nan, np.nan, 1.0, np.nan]),
        equal_nan=True,
    )

@pytest.mark.parametrize('w', [torch.tensor([], requires_grad=True), torch.tensor([[], []])])
@pytest.mark.parametrize('minlength', [0, 10])
def test_bincount_zero_length(w, minlength):
    """
    Check that the bug where lib.pt.bincount raises error on inputs with length 0 is fixed.
    """
    a = torch.tensor([], dtype=torch.long)
    t = lib.pt.bincount(a, weights=w, minlength=minlength)

    if w.ndim == 2:
        arr = []
        for wi in w:
            arr.append(np.bincount(a.numpy(), weights=wi.detach().numpy(), minlength=minlength))
        arr = np.stack(arr)
    elif w.ndim == 1:
        arr = np.bincount(a.numpy(), weights=w.detach().numpy(), minlength=minlength)
    else:
        raise RuntimeError()
    
    assert np.allclose(t.detach().numpy(), arr, equal_nan=True)

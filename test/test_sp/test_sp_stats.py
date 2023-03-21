from contextlib import nullcontext as does_not_raise
import numbers

import pytest
import numpy as np
import pandas as pd
from scipy import stats

import hyclib as lib

@pytest.mark.parametrize('D, bins', [
    (1, 100),
    (2, 10),
    (2, (20,5)),
    (2, ([-2.5,-2,-1,0,1,2,3.0],5)),
    (2, ([-2.5,-2,-1,0,1,2,3.0],[-2.5,-2,-1,0,1,2,3.0])),
])
@pytest.mark.parametrize('has_nan, nan_policy, expectation', [
    (True, 'raise', pytest.raises(ValueError)),
    (True, 'omit', does_not_raise()),
    (False, 'raise', does_not_raise()),
    (False, 'omit', does_not_raise()),
])
@pytest.mark.parametrize('expand_binnumbers', [
    True,
    False,
])
def test_bin_dd(D, bins, has_nan, nan_policy, expectation, expand_binnumbers):
    M, N = 100, 100000

    arr = np.random.normal(size=(N,D))
    
    if has_nan:
        indices_0 = np.random.randint(N, size=M)
        indices_1 = np.random.randint(D, size=M)
        arr[indices_0, indices_1] = np.nan

    with expectation as exc_info:
        binnumbers, centers, edges = lib.sp.stats.bin_dd(arr, bins=bins, nan_policy=nan_policy, expand_binnumbers=expand_binnumbers)

    if not has_nan:
        ret = stats.binned_statistic_dd(arr, np.zeros(N), bins=bins, expand_binnumbers=expand_binnumbers)
        
        if D == 1 and expand_binnumbers:
            np.testing.assert_allclose(binnumbers, ret.binnumber[None,:])
        else:
            np.testing.assert_allclose(binnumbers, ret.binnumber)
            
        for i in range(D):
            np.testing.assert_allclose(edges[i], ret.bin_edges[i])
            
    elif exc_info is None and expand_binnumbers:
        # results should be as if we performed lib.sp.stats.bin on each dimension separately
        for i in range(D):
            bins_i = bins if isinstance(bins, numbers.Number) else bins[i]
            binnumbers_i, centers_i, edges_i = lib.sp.stats.bin(arr[:, i], bins=bins_i, nan_policy=nan_policy)
            np.testing.assert_allclose(binnumbers[i], binnumbers_i)
            np.testing.assert_allclose(centers[i], centers_i)
            np.testing.assert_allclose(edges[i], edges_i)

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
    nan_indices = np.random.randint(N, size=M)
    arr[nan_indices] = np.nan

    pbinnumbers, pedges = pd.cut(arr, bins=bins, retbins=True, labels=False, right=False)
    
    with expectation as exc_info:
        binnumbers, centers, edges = lib.sp.stats.bin(arr, bins=bins, nan_policy=nan_policy)

    if exc_info is None:
        pd_isnan = np.isnan(pbinnumbers)
        np.testing.assert_allclose(binnumbers[~pd_isnan], pbinnumbers[~pd_isnan] + 1) # check that binnumbers is consistent with pd.cut

        assert np.isnan(centers[0]) and np.isnan(centers[-1])
        np.testing.assert_allclose(centers[1:-1], 0.5 * (edges[1:] + edges[:-1])) # check that centers is consistent with edges
        
        if isinstance(bins, numbers.Number):
            r = np.nanmax(arr) - np.nanmin(arr)
            edges[-1] = edges[-1] + r * 0.001
        np.testing.assert_allclose(edges, pedges) # check that edges is consistent with pd.cut

        np.testing.assert_allclose(binnumbers[nan_indices], len(edges)) # check that nan samples are sorted into rightmost bin
        np.testing.assert_allclose(binnumbers[arr < np.min(edges)], 0) # check that samples less than leftmost edge are sorted into leftmost bin
        np.testing.assert_allclose(binnumbers[arr > np.max(edges)], len(edges)) # check that samples less than leftmost edge are sorted into leftmost bin

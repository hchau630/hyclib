import pytest
import numpy as np
import scipy.stats as stats

import utils

@pytest.fixture
def y():
    y = np.array(
        [[[ 0.03476371,  0.17975505, -0.54908644, -0.64903577],
          [-0.03724603,  1.81708173, -0.36515959,      np.inf],
          [ 0.41836838,      np.nan, -0.49636453,  0.18997459]],

         [[ 1.466666  ,  1.0255607 ,      np.nan, -1.31296034],
          [ 1.08370192, -0.09796155, -0.05370452, -0.34803141],
          [ 0.13195273, -1.43293285,  0.82484756,  0.83380857]]]
    )
    return y

@pytest.fixture
def yerr():
    yerr = np.array(
        [[[ 0.49388935,  0.99288461,  2.16113518,  3.01975125],
          [-1.50622019,  0.34152895,  0.2847713 ,  1.57976904],
          [-0.51299135,      np.nan, -0.3417016 , -1.43069183]],

         [[ 0.95822919,  1.41162136,      np.inf,  0.36485372],
          [-1.15548473, -0.08944152,  0.60378143,      np.nan],
          [-1.15460362, -0.007127  , -0.51625072,  0.18837434]]]
    )
    return yerr

def test_sem(y):
    assert np.allclose(utils.np.sem(y), stats.sem(y, axis=None, ddof=1), equal_nan=True)
    assert np.allclose(utils.np.sem(y, axis=1), stats.sem(y, axis=1, ddof=1), equal_nan=True)
    assert np.allclose(utils.np.sem(y, ddof=0), stats.sem(y, axis=None, ddof=0), equal_nan=True)

def test_nansem(y):
    ynan = y.copy()
    y[~np.isfinite(y)] = np.nan
    assert np.allclose(utils.np.nansem(y), np.array(stats.sem(ynan, axis=None, ddof=1, nan_policy='omit')), equal_nan=True)
    assert np.allclose(utils.np.nansem(y, axis=1), np.array(stats.sem(ynan, axis=1, ddof=1, nan_policy='omit')), equal_nan=True)
    assert np.allclose(utils.np.nansem(y, ddof=0), np.array(stats.sem(ynan, axis=None, ddof=0, nan_policy='omit')), equal_nan=True)
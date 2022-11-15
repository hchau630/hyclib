import pytest

import numpy as np

import utils

@pytest.mark.parametrize('x, axis, expected', [
    (np.array(2), None, True),
    (np.array([]), None, True),
    (np.array([]), 0, True),
    (np.array([1.0]), None, True),
    (np.array([1.0]), 0, True),
    (np.array([[1.0]]), None, True),
    (np.array([[1.0]]), 0, True),
    (np.array([[1.0]]), 1, True),
    (np.array([[1.0,1.0,1.0]]), None, True),
    (np.array([[1.0,0.5,1.0]]), None, False),
    (np.array([[1.0,1.0,1.0],
               [2.0,2.0,2.0]]), None, False),
    (np.array([[1.0,1.0,1.0],
               [2.0,2.0,2.0]]), 0, False),
    (np.array([[1.0,1.0,1.0],
               [2.0,2.0,2.0]]), 1, True),
    (np.array([[1.0,1.0,1.0],
               [2.0,2.0,2.0]]), (0,1), False),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), None, False),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), 0, True),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), 1, False),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), -1, True),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (0,1), False),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (1,0), False),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (0,2), True),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (2,0), True),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (0,-1), True),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (-1,0), True),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (1,2), False),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (2,1), False),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (-1,-2), False),
    (np.array([[[1.0,1.0,1.0],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,2.0,2.0]]]), (-2,-1), False),
    (np.array([[[1.0,1.0,0.5],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,0.5,2.0]]]),
     0,
     np.array([[True, True, False],
               [True, False, True]])),
    (np.array([[[1.0,1.0,0.5],
                [1.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,1.0,2.0]]]),
     1,
     np.array([[True, False, False],
               [False, True, False]])),
    (np.array([[[1.0,1.0,0.5],
                [2.0,2.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,1.0,2.0]]]),
     2,
     np.array([[False, True],
               [True, False]])),
    (np.array([[[1.0,1.0,0.5],
                [2.0,1.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,1.0,2.0]]]),
     (0,1),
     np.array([False, True, False])),
    (np.array([[[1.0,1.0,1.0],
                [0.5,1.0,2.0]],
               [[1.0,1.0,1.0],
                [2.0,1.0,2.0]]]),
     (0,2),
     np.array([True, False])),
    (np.array([[[1.0,1.0,1.0],
                [0.5,1.0,2.0]],
               [[1.0,1.0,1.0],
                [1.0,1.0,1.0]]]),
     (1,2),
     np.array([False, True])),
])
def test_isconst(x, axis, expected):
    assert np.all(utils.np.isconst(x, axis=axis) == expected)
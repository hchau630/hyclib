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
    
def test_meshgrid_dd():
    a, b, c = np.random.normal(size=(3,5,2)), np.random.normal(size=(4,3)), np.random.normal(size=(5,7,2,4))
    a_, b_, c_ = utils.np.meshgrid_dd(a, b, c)
    assert a_.shape == (3,5,4,5,7,2,2) and b_.shape == (3,5,4,5,7,2,3) and c_.shape == (3,5,4,5,7,2,4)
    assert utils.np.isconst(a_, axis=(2,3,4,5)).all() and (a_[:,:,0,0,0,0,:] == a).all()
    assert utils.np.isconst(b_, axis=(0,1,3,4,5)).all() and (b_[0,0,:,0,0,0,:] == b).all()
    assert utils.np.isconst(c_, axis=(0,1,2)).all() and (c_[0,0,0,:,:,:,:] == c).all()
    
def test_meshgrid():
    a, b, c = np.random.normal(size=(3,5)), np.random.normal(size=(4,)), np.random.normal(size=(5,7,2))
    a_, b_, c_ = utils.np.meshgrid(a, b, c)
    assert a_.shape == (3,5,4,5,7,2) and b_.shape == (3,5,4,5,7,2) and c_.shape == (3,5,4,5,7,2)
    assert utils.np.isconst(a_, axis=(2,3,4,5)).all() and (a_[:,:,0,0,0,0] == a).all()
    assert utils.np.isconst(b_, axis=(0,1,3,4,5)).all() and (b_[0,0,:,0,0,0] == b).all()
    assert utils.np.isconst(c_, axis=(0,1,2)).all() and (c_[0,0,0,:,:,:] == c).all()
    
    a, b, c = np.random.normal(size=3), np.random.normal(size=4), np.random.normal(size=5)
    a1, b1, c1 = utils.np.meshgrid(a, b, c, indexing='ij')
    a2, b2, c2 = np.meshgrid(a, b, c, indexing='ij')
    np.testing.assert_allclose(a1, a2)
    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(c1, c2)
    
    a, b, c = np.random.normal(size=3), np.random.normal(size=4), np.random.normal(size=5)
    a1, b1, c1 = utils.np.meshgrid(a, b, c, indexing='xy')
    a2, b2, c2 = np.meshgrid(a, b, c, indexing='xy')
    np.testing.assert_allclose(a1, a2)
    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(c1, c2)
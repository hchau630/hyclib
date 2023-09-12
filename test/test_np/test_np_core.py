import pytest

import numpy as np

import hyclib as lib

@pytest.mark.parametrize('x, axis, expected', [
    (np.array(2), None, True),
    (np.array(2), (), np.array(True)),
    (np.array([]), None, True),
    (np.array([]), 0, True),
    (np.array([1.0]), None, True),
    (np.array([1.0]), 0, True),
    (np.array([1.0]), (), np.array([True])),
    (np.array([[1.0]]), None, True),
    (np.array([[1.0]]), 0, True),
    (np.array([[1.0]]), 1, True),
    (np.array([[1.0]]), (), np.array([[True]])),
    (np.array([[1.0,1.0,1.0]]), None, True),
    (np.array([[1.0,0.5,1.0]]), None, False),
    (np.array([[1.0,0.5,1.0]]), (), np.array([[True, True, True]])),
    (np.array([[1.0,1.0,1.0],
               [2.0,2.0,2.0]]), None, False),
    (np.array([[1.0,1.0,1.0],
               [2.0,2.0,2.0]]), 0, False),
    (np.array([[1.0,1.0,1.0],
               [2.0,2.0,2.0]]), 1, True),
    (np.array([[1.0,1.0,1.0],
               [2.0,2.0,2.0]]), (0,1), False),
    (
        np.array([[1.0,1.0,1.0],
                  [2.0,2.0,2.0]]),
        (),
        np.array([[True,True,True],
                  [True,True,True]])
    ),
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
    assert np.all(lib.np.isconst(x, axis=axis) == expected)
    
@pytest.mark.parametrize('indexing, ndims, shared_shape, post_shapes, reduce_dims, indices', [
    (
        'ij',
        None,
        (3, 5, 2, 4, 3, 5, 7, 2, 4),
        [(), (), ()],
        [(3, 4, 5, 6, 7, 8), (0, 1, 2, 5, 6, 7, 8), (0, 1, 2, 3, 4)],
        [np.s_[:, :, :, 0, 0, 0, 0, 0, 0], np.s_[0, 0, 0, :, :, 0, 0, 0, 0,], np.s_[0, 0, 0, 0, 0, :, :, :, :]]
    ),
    (
        'ij',
        0,
        (),
        [(3, 5, 2), (4, 3), (5, 7, 2, 4)],
        [(), (), ()],
        [(Ellipsis,), (Ellipsis,), (Ellipsis,)]
    ),
    (
        'ij',
        -1,
        (3, 5, 4, 5, 7, 2),
        [(2,), (3,), (4,)],
        [(2, 3, 4, 5), (0, 1, 3, 4, 5), (0, 1, 2)],
        [np.s_[:, :, 0, 0, 0, 0], np.s_[0, 0, :, 0, 0, 0], np.s_[0, 0, 0, :, :, :]],
    ),
    (
        'ij',
        2,
        (3, 5, 4, 3, 5, 7),
        [(2,), (), (2, 4)],
        [(2, 3, 4, 5), (0, 1, 4, 5), (0, 1, 2, 3)],
        [np.s_[:, :, 0, 0, 0, 0], np.s_[0, 0, :, :, 0, 0], np.s_[0, 0, 0, 0, :, :]],
    ),
    (
        'ij',
        [2, -1, 0],
        (3, 5, 4),
        [(2,), (3,), (5, 7, 2, 4)],
        [(2,), (0, 1), ()],
        [np.s_[:, :, 0], np.s_[0, 0, :], (Ellipsis,)],
    ),
    (
        'xy',
        None,
        (4, 3, 3, 5, 2, 5, 7, 2, 4),
        [(), (), ()],
        [(0, 1, 5, 6, 7, 8), (2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4)],
        [np.s_[0, 0, :, :, :, 0, 0, 0, 0], np.s_[:, :, 0, 0, 0, 0, 0, 0, 0,], np.s_[0, 0, 0, 0, 0, :, :, :, :]]
    ),
    (
        'xy',
        0,
        (),
        [(3, 5, 2), (4, 3), (5, 7, 2, 4)],
        [(), (), ()],
        [(Ellipsis,), (Ellipsis,), (Ellipsis,)]
    ),
    (
        'xy',
        -1,
        (4, 3, 5, 5, 7, 2),
        [(2,), (3,), (4,)],
        [(0, 3, 4, 5), (1, 2, 3, 4, 5), (0, 1, 2)],
        [np.s_[0, :, :, 0, 0, 0], np.s_[:, 0, 0, 0, 0, 0], np.s_[0, 0, 0, :, :, :]],
    ),
    (
        'xy',
        2,
        (4, 3, 3, 5, 5, 7),
        [(2,), (), (2, 4)],
        [(0, 1, 4, 5), (2, 3, 4, 5), (0, 1, 2, 3)],
        [np.s_[0, 0, :, :, 0, 0], np.s_[:, :, 0, 0, 0, 0], np.s_[0, 0, 0, 0, :, :]],
    ),
    (
        'xy',
        [2, -1, 0],
        (4, 3, 5),
        [(2,), (3,), (5, 7, 2, 4)],
        [(0,), (1, 2), ()],
        [np.s_[0, :, :], np.s_[:, 0, 0], (Ellipsis,)],
    ),
])
def test_meshgrid(indexing, ndims, shared_shape, post_shapes, reduce_dims, indices):
    sizes = ((3, 5, 2), (4, 3), (5, 7, 2, 4))
    in_tensors = [np.random.rand(*size) for size in sizes]
    out_tensors = lib.np.meshgrid(*in_tensors, indexing=indexing, ndims=ndims)
    
    assert all(out_tensor.shape == shared_shape + post_shape for out_tensor, post_shape in zip(out_tensors, post_shapes))
    assert all(lib.np.isconst(out_tensor, axis=reduce_dim).all() for out_tensor, reduce_dim in zip(out_tensors, reduce_dims))
    assert all(
        (out_tensor[tuple(index)] == in_tensor).all()
        for in_tensor, out_tensor, index in zip(in_tensors, out_tensors, indices)
    )

@pytest.mark.parametrize('ndims', [
    10,
    -10,
    (3, 3, 4),
    (-1, -3, 2),
    (1, 2, 1, 0),
    (1, 2),
])
def test_meshgrid_invalid(ndims):
    sizes = ((3, 5, 2), (4, 3), (5, 7, 2, 4))
    in_tensors = [np.random.rand(*size) for size in sizes]
    with pytest.raises(ValueError):
        lib.np.meshgrid(*in_tensors, ndims=ndims)
    
@pytest.mark.parametrize('indexing', ['ij', 'xy'])
def test_meshgrid_1d(indexing):
    a, b, c = np.random.rand(3), np.random.rand(4), np.random.rand(5)
    a1, b1, c1 = lib.np.meshgrid(a, b, c, indexing=indexing)
    a2, b2, c2 = np.meshgrid(a, b, c, indexing=indexing)
    np.testing.assert_allclose(a1, a2)
    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(c1, c2)

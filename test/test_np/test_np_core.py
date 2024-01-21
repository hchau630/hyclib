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

def test_isconst_float_dtype_check():
    """
    Fix bug where float dtypes are not detected correctly and hence
    np.isclose was not used for floats.
    """
    x = np.array([
        [1.0, 1.0 + 1.0e-9, 1.0 - 1.0e-9],
        [2.0, 2.0 + 1.0e-3, 2.0 - 1.0e-3]
    ])
    expected = np.array([True, False])
    assert np.all(lib.np.isconst(x, axis=1) == expected)
    
@pytest.mark.parametrize('indexing, axes, shared_shape, pre_shapes, post_shapes, inserted_axes', [
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
def test_meshgrid(indexing, axes, sparse, shared_shape, pre_shapes, post_shapes, inserted_axes):
    if pre_shapes is None:
        pre_shapes = [()] * 3
    if post_shapes is None:
        post_shapes = [()] * 3
    
    sizes = ((3, 5, 2), (4, 3), (5, 7, 2, 4))
    in_arrays = [np.random.rand(*size) for size in sizes]
    out_arrays = lib.np.meshgrid(*in_arrays, indexing=indexing, axes=axes, sparse=sparse)

    masks, indices = [], []
    for out_array, inserted_axis in zip(out_arrays, inserted_axes):
        mask = np.zeros(out_array.ndim, dtype=bool)
        mask[list(inserted_axis)] = True
        index = np.array([slice(None) for _ in range(out_array.ndim)])
        index[mask] = 0
        masks.append(mask)
        indices.append(tuple(index))

    for out_array, pre_shape, post_shape, mask in zip(out_arrays, pre_shapes, post_shapes, masks):
        expected_shape = pre_shape + shared_shape + post_shape
        if sparse:
            expected_shape = np.array(expected_shape)
            expected_shape[mask] = 1
            expected_shape = tuple(expected_shape.tolist())
        assert out_array.shape == expected_shape

    assert all(lib.np.isconst(out_array, axis=inserted_axis).all() for out_array, inserted_axis in zip(out_arrays, inserted_axes))
    assert all((out_array[index] == in_array).all() for in_array, out_array, index in zip(in_arrays, out_arrays, indices))

@pytest.mark.parametrize('axes', [
    0.5,
    (0, 1, 2),
    [(0, 1), (0, 1)],
    [(0, 1, 2), (0, 1, 2), (0, 1, 2)],
])
def test_meshgrid_invalid(axes):
    sizes = ((3, 5, 2), (4, 3), (5, 7, 2, 4))
    in_arrays = [np.random.rand(*size) for size in sizes]
    with pytest.raises(TypeError):
        lib.np.meshgrid(*in_arrays, axes=axes)
    
@pytest.mark.parametrize('indexing', ['ij', 'xy'])
def test_meshgrid_1d(indexing):
    a, b, c = np.random.rand(3), np.random.rand(4), np.random.rand(5)
    a1, b1, c1 = lib.np.meshgrid(a, b, c, indexing=indexing)
    a2, b2, c2 = np.meshgrid(a, b, c, indexing=indexing)
    np.testing.assert_allclose(a1, a2)
    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(c1, c2)

def test_meshndim():
    arrs = (np.random.normal(size=(3, 4, 5)), np.random.normal(size=(6, 7, 8, 9)))
    output = (arr[idx] for arr, idx in zip(arrs, lib.np.meshndim(2, 3)))
    expected = lib.np.meshgrid(*arrs, axes=[(None, 2), (None, 3)])
    assert all((a == b).all() for a, b in zip(output, expected))

@pytest.mark.parametrize('dtype', [np.int64, np.float32])
def test_repeat(dtype):
    arr = np.array([0, 5, 2, 10, 11, 20, 21, 22, 23], dtype=dtype)

    chunks = np.array([3, 2, 4])
    repeats = np.array([1, 3, 2])
    
    out = lib.np.repeat(arr, repeats, chunks=chunks)
    expected = np.array([0, 5, 2, 10, 11, 10, 11, 10, 11, 20, 21, 22, 23, 20, 21, 22, 23], dtype=dtype)

    np.testing.assert_allclose(out, expected)

@pytest.mark.parametrize('dtype', [np.int64, np.float32])
def test_repeat_empty(dtype):
    arr = np.array([], dtype=dtype)

    chunks = np.array([], dtype=int)
    repeats = np.array([], dtype=int)
    
    out = lib.np.repeat(arr, repeats, chunks=chunks)
    expected = np.array([], dtype=dtype)

    np.testing.assert_allclose(out, expected)

@pytest.mark.xfail(reason='0 repeats currently not supported')
@pytest.mark.parametrize('dtype', [np.int64, np.float32])
@pytest.mark.parametrize('repeats, expected', [
    ([3, 0, 2], [0, 5, 2, 0, 5, 2, 0, 5, 2, 20, 21, 22, 23, 20, 21, 22, 23]),
    ([0, 3, 2], [10, 11, 10, 11, 10, 11, 20, 21, 22, 23, 20, 21, 22, 23]),
    ([3, 2, 0], [0, 5, 2, 0, 5, 2, 0, 5, 2, 10, 11, 10, 11]),
])
def test_repeat_zero_repeat(repeats, expected, dtype):
    """Fix bug in edge case where there are zeros in repeats."""
    arr = np.array([0, 5, 2, 10, 11, 20, 21, 22, 23], dtype=dtype)

    chunks = np.array([3, 2, 4])
    repeats = np.array(repeats)
    
    out = lib.np.repeat(arr, repeats, chunks=chunks)
    expected = np.array(expected, dtype=dtype)

    np.testing.assert_allclose(out, expected)

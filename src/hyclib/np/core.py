import collections
import itertools

import numpy as np

__all__ = [
    'isconst',
    'meshshape',
    'meshndim',
    'meshgrid',
    'inv_perm',
    'unique_rows',
    'intersect_rows',
    'repeat',
]

def isconst(x, axis=None, **kwargs):
    x = np.asanyarray(x)
    
    if axis is None:
        x = x.reshape(-1)
    else:
        if isinstance(axis, int):
            axis = [axis]
        if len(axis) == 0:
            return np.ones(x.shape, dtype=bool)
        axis = sorted([d % x.ndim for d in axis])[::-1]
        for d in axis:
            x = np.moveaxis(x, d,-1)
        x = x.reshape(*x.shape[:-len(axis)],-1)
        
    if np.issubdtype(x.dtype, np.floating):
        return np.isclose(x[...,:-1], x[...,1:], **kwargs).all(axis=-1)
    return (x[...,:-1] == x[...,1:]).all(axis=-1)

def meshshape(*shapes, indexing='ij'):
    """
    The fundamental step of meshgrid, which is useful on its own as a standalone
    function. Given n shapes
    (s^1_1, ..., s^1_{D_1}), (s^2_1, ..., s^2_{D_2}), ..., (s^n_1, ..., s^n_{D_n}),
    Yields shapes
    (s^1_1, ..., s^1_{D_1}, 1, ..............., 1, ..., 1, ..............., 1),
    (1, ..............., 1, s^2_1, ..., s^2_{D_2}, ..., 1, ..............., 1),
    ...
    (1, ..............., 1, 1, ..............., 1, ..., s^n_1, ..., s^n_{D_n}),
    If indexing == 'xy', yields shapes
    (1, ..............., 1, s^1_1, ..., s^1_{D_1}, ..., 1, ..............., 1),
    (s^2_1, ..., s^2_{D_2}, 1, ..............., 1, ..., 1, ..............., 1),
    ...
    (1, ..............., 1, 1, ..............., 1, ..., s^n_1, ..., s^n_{D_n}),
    """
    if not indexing in {'ij', 'xy'}:
        raise ValueError(f"indexing must 'ij' or 'xy', but got {indexing}.")

    if indexing == 'xy':
        if len(shapes) < 2:
            raise ValueError(f"Must have at least 2 shapes if indexing == 'xy', but got {len(shapes)=}.")
        
        shapes = (shapes[1], shapes[0], *shapes[2:])
        out = list(meshshape(*shapes, indexing='ij'))
        out[1], out[0] = out[0], out[1]
        yield from out
    
    sumndim = sum(len(shape) for shape in shapes)
    cumndim = 0
    for shape in shapes:
        ndim = len(shape)
        yield (1,) * cumndim + shape + (1,) * (sumndim - cumndim - ndim)
        cumndim += ndim

def meshndim(*ndims, indexing='ij'):
    """
    Same as meshshape, but yields indices instead.
    Note that reshaping using indices is faster than reshaping using shapes in numpy,
    but the opposite is true in pytorch.
    
    f1 = lambda a, n, m: a.reshape((1,) * n  + a.shape + (1,) * m)
    f2 = lambda a, n, m: a[(None,) * n + (slice(None),) * a.ndim + (None,) * m]
    shape, n, m = (3,4,100,2,6), 5, 7
    
    a = np.random.normal(shape=shape)
    %timeit f1(a, n, m)
    489 ns ± 9.55 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    %timeit f2(a, n, m)
    472 ns ± 7.58 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

    a = torch.randn(shape)
    %timeit f1(a, n, m)
    2.62 µs ± 16.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    %timeit f2(a, n, m)
    10.8 µs ± 102 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    """
    if not indexing in {'ij', 'xy'}:
        raise ValueError(f"indexing must 'ij' or 'xy', but got {indexing}.")

    if indexing == 'xy':
        if len(shapes) < 2:
            raise ValueError(f"Must have at least 2 shapes if indexing == 'xy', but got {len(shapes)=}.")
        
        shapes = (shapes[1], shapes[0], *shapes[2:])
        out = list(meshndim(*shapes, indexing='ij'))
        out[1], out[0] = out[0], out[1]
        yield from out
    
    sumndim = sum(ndims)
    cumndim = 0
    for ndim in ndims:
        # we don't use Ellipsis because it doesn't allow for possible additional batch dimensions
        yield (None,) * cumndim + (slice(None),) * ndim + (None,) * (sumndim - cumndim - ndim)
        cumndim += ndim

def meshgrid(*arrays, indexing='ij', axes=None, sparse=False):
    """
    The most general version of np.meshgrid. Meshes together arrays of shapes
    (*00, n0[axes[0][0]], ..., n0[axes[0][1]], *01),
    (*10, n1[axes[1][0]], ..., n1[axes[1][1]], *11),
    ..., 
    (*P0, nP[axes[P][0]], ..., nP[axes[P][1]], *P1),
    indexing must be either 'ij' or 'xy'. Defaults to 'ij' like pytorch but unlike numpy.
    If axes is None, meshes over all dimensions. axes can be None, an int,
    or a length-P list/tuple of length-2 list/tuple of ints or None, where P = len(arrays).
    Negative axis has the same semantic meaning as slicing with a negative index.
    If sparse, new dimensions are of length one instead of the broadcasted length.
    
    If indexing == 'ij', returns arrays of shapes 
    (*00, n0[axes[0][0]], ..., n0[axes[0][1]], n1[axes[1][0]], ..., n1[axes[1][1]], ..., nP[axes[P][0]], ..., nP[axes[P][1]], *01),
    (*10, n0[axes[0][0]], ..., n0[axes[0][1]], n1[axes[1][0]], ..., n1[axes[1][1]], ..., nP[axes[P][0]], ..., nP[axes[P][1]], *11),
    ...,
    (*P0, n0[axes[0][0]], ..., n0[axes[0][1]], n1[axes[1][0]], ..., n1[axes[1][1]], ..., nP[axes[P][0]], ..., nP[axes[P][1]], *P1),
    
    Otherwise, returns arrays of shapes
    (*00, n1[axes[1][0]], ..., n1[axes[1][1]], n0[axes[0][0]], ..., n0[axes[0][1]], ..., nP[axes[P][0]], ..., nP[axes[P][1]], *01),
    (*10, n1[axes[1][0]], ..., n1[axes[1][1]], n0[axes[0][0]], ..., n0[axes[0][1]], ..., nP[axes[P][0]], ..., nP[axes[P][1]], *11),
    ...,
    (*P0, n1[axes[1][0]], ..., n1[axes[1][1]], n0[axes[0][0]], ..., n0[axes[0][1]], ..., nP[axes[P][0]], ..., nP[axes[P][1]], *P1),
    
    IMPORTANT: Data is NOT copied just like pytorch, unlike numpy which copies by default.
    """
    if not indexing in {'ij', 'xy'}:
        raise ValueError(f"indexing must 'ij' or 'xy', but got {indexing}.")
    
    if axes is None or isinstance(axes, int):
        axes = [(None, axes)] * len(arrays)
    elif not (
        isinstance(axes, (list, tuple)) and len(axes) == len(arrays) and 
        all(
            (
                isinstance(axis, (list, tuple)) and len(axis) == 2 and 
                all(d is None or isinstance(d, int) for d in axis)
            )
            for axis in axes
        )
    ):
        raise TypeError(
            f"""axes must be None, an int, or a length-{len(arrays)}
            list/tuple of length-2 list/tuple of None or int, but {axes=}."""
        )

    pre_shapes, shapes, post_shapes = zip(*(
        (
            () if axis[0] is None else array.shape[:axis[0]],
            array.shape[axis[0]:axis[1]],
            () if axis[1] is None else array.shape[axis[1]:],
        )
        for array, axis in zip(arrays, axes)
    ))
    shapes = list(meshshape(*shapes, indexing=indexing))
    arrays = (
        array.reshape(pre_shape + shape + post_shape)
        for array, shape, pre_shape, post_shape in zip(arrays, shapes, pre_shapes, post_shapes)
    )
    if not sparse:
        shape = np.broadcast_shapes(*shapes)
        arrays = (
            np.broadcast_to(array, pre_shape + shape + post_shape)
            for array, pre_shape, post_shape in zip(arrays, pre_shapes, post_shapes)
        )
    
    return list(arrays)


def inv_perm(p):
    """
    Code taken from: https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
    Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def unique_rows(arr, sorted=True, return_index=False, return_inverse=False, return_counts=False):
    """
    Code modified from https://github.com/scikit-image/scikit-image/blob/main/skimage/util/unique.py
    Much faster than np.unique(arr, axis=0)
    See https://github.com/numpy/numpy/issues/11136
    Set sorted=False for maximum speed.
    In this implementation, equal_nan=True.
    
    For speed comparison:
    N, M = 100000,100
    arr = np.random.randint(M, size=(N,2)).astype(float)
    
    %timeit unique_rows(arr, sorted=False, return_index=True, return_inverse=True)
    25.9 ms ± 386 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    
    %timeit np.unique(arr, return_index=True, return_inverse=True, axis=0)
    85.8 ms ± 240 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    if arr.ndim != 2:
        raise ValueError("unique_rows() only makes sense for 2D arrays, "
                         f"got {arr.ndim}")
    # the view in the next line only works if the array is C-contiguous
    arr = np.ascontiguousarray(arr)
    # np.unique() finds identical items in a raveled array. To make it
    # see each row as a single item, we create a view of each row as a
    # byte string of length itemsize times number of columns in `ar`
    arr_row_view = arr.view(f"|S{arr.itemsize * arr.shape[1]}")
    out = np.unique(arr_row_view, return_index=True, return_inverse=return_inverse, return_counts=return_counts)
    out = list(out)
    out[0] = arr[out[1]]
    
    if sorted:
        idx = np.lexsort(out[0].T[::-1])
        out[0] = out[0][idx]
        if return_index:
            out[1] = out[1][idx]
        if return_inverse:
            out[2] = inv_perm(idx)[out[2]]
        if return_counts:
            i = 3 if return_inverse else 2
            out[i] = out[i][idx]
    
    if not return_index:
        out.pop(1)

    if len(out) == 1:
        return out[0]
    return out

def intersect_rows(larr, rarr, return_indices=False, **kwargs):
    """
    intersect1d but for 2D arrays. Uses the same trick as unique_rows.
    """
    if larr.ndim != 2 or rarr.ndim != 2:
        raise ValueError("intersect_rows() only makes sense for 2D arrays, "
                         f"got {larr.ndim=}, {rarr.ndim=}")
    if larr.shape[1] != rarr.shape[1]:
        raise ValueError(f"larr and rarr must both have same number of columns, but {larr.shape[1]=}, {rarr.shape[1]=}.")
        
    N_cols = larr.shape[1]
    larr, rarr = np.ascontiguousarray(larr), np.ascontiguousarray(rarr)
    larr_row_view, rarr_row_view = larr.view(f"|S{larr.itemsize * N_cols}"), rarr.view(f"|S{rarr.itemsize * N_cols}")
    
    out = np.intersect1d(larr_row_view, rarr_row_view, return_indices=True, **kwargs)
    out = list(out)
    out[0] = larr[out[1]]
    
    if not return_indices:
        return out[0]
    return out
    
def repeat(arr, repeats, chunks=None, validate=True):
    """
    Generalized np.repeat
    Copied from @MadPhysicist's solution: https://stackoverflow.com/questions/63510977/repeat-but-in-variable-sized-chunks-in-numpy
    
    Example due to @MadPhysicist in the same link:
    arr = np.array([0, 1, 2, 10, 11, 20, 21, 22, 23])
    #               >     <  >    <  >            <
    chunks = np.array([3, 2, 4])
    repeats = np.array([1, 3, 2])
    
    repeat(arr, repeats, chunks=chunks)
    >>> np.array([0, 1, 2, 10, 11, 10, 11, 10, 11, 20, 21, 22, 23, 20, 21, 22, 23])
    # repeats:    >  1  <  >         3          <  >              2             <
    """
    if chunks is None:
        return np.repeat(arr, repeats)
    
    arr, repeats, chunks = np.asanyarray(arr), np.asanyarray(repeats), np.asanyarray(chunks)

    if validate:
        if arr.ndim != 1 or repeats.ndim != 1 or chunks.ndim != 1:
            raise ValueError(f"arr, repeats, and chunks must all be 1D, but {arr.ndim=}, {repeats.ndim=} and {chunks.ndim=}.")
        if len(repeats) != len(chunks):
            raise ValueError(f"repeats and chunks must have the same length, but {len(repeats)=} and {len(chunks)=}.")
        if chunks.sum() != len(arr):
            raise ValueError(f"sum of chunks must be the length of arr, but {chunks.sum()=} and {len(arr)=}.")
        if len(repeats) > 0 and repeats.min() < 0:
            raise ValueError(f"repeats must be non-negative, but {repeats.min()=}.")
        if len(repeats) > 0 and repeats.min() == 0:
            raise NotImplementedError(f"Currently does not support zero repeats, but {repeats.min()=}.")

    if len(repeats) == 0:
        return np.empty((0,), dtype=arr.dtype)

    # # deal with zero repeats (still buggy, and this is inefficient)
    # iszero = repeats == 0
    # if iszero.any():
    #     head_indices = np.concatenate([[0], chunks]).cumsum()[:-1]
    #     index = np.arange(len(arr) - chunks[iszero].sum())
    #     offsets = np.zeros_like(index)
    #     offsets[head_indices[:-1][iszero[:-1]]] -= chunks[:-1][iszero[:-1]]
    #     index -= offsets.cumsum()
    #     return repeat(arr[index], repeats[~iszero], chunks[~iszero], validate=False)

    regions = chunks * repeats
    index = np.arange(regions.sum())

    segments = np.repeat(chunks, repeats)
    resets = np.cumsum(segments[:-1])
    offsets = np.zeros_like(index)
    offsets[resets] = segments[:-1]
    offsets[np.cumsum(regions[:-1])] -= chunks[:-1]

    index -= np.cumsum(offsets)

    out = arr[index]
    
    return out

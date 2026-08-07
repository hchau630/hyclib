import builtins
from operator import index

import numpy as np


def _bin_edges(sample, bins=None, range=None):
    """Create edge arrays"""
    _, Ndim = sample.shape

    nbin = np.empty(Ndim, int)  # Number of bins in each dimension
    edges = Ndim * [None]  # Bin edges for each dim (will be 2D array)
    dedges = Ndim * [None]  # Spacing between edges (will be 2D array)

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        smin = np.atleast_1d(np.array(np.nanmin(sample, axis=0), float))
        smax = np.atleast_1d(np.array(np.nanmax(sample, axis=0), float))
    else:
        if len(range) != Ndim:
            raise ValueError(
                f"range given for {len(range)} dimensions; {Ndim} required"
            )
        smin = np.empty(Ndim)
        smax = np.empty(Ndim)
        for i in builtins.range(Ndim):
            if range[i][1] < range[i][0]:
                raise ValueError(
                    "In {}range, start must be <= stop".format(
                        f"dimension {i + 1} of " if Ndim > 1 else ""
                    )
                )
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in builtins.range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - 0.5
            smax[i] = smax[i] + 0.5

    # Preserve sample floating point precision in bin edges
    edges_dtype = sample.dtype if np.issubdtype(sample.dtype, np.floating) else float

    # Create edge arrays
    for i in builtins.range(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1, dtype=edges_dtype)
        else:
            edges[i] = np.asarray(bins[i], edges_dtype)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])

    nbin = np.asarray(nbin)

    return nbin, edges, dedges


def _bin_numbers(sample, nbin, edges, dedges, expand_binnumbers=False):
    """Compute the bin number each sample falls into, in each dimension"""
    _, Ndim = sample.shape

    sampBin = [np.digitize(sample[:, i], edges[i]) for i in range(Ndim)]

    # Using `digitize`, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    for i in range(Ndim):
        # Find the rounding precision
        dedges_min = dedges[i].min()
        if dedges_min == 0:
            raise ValueError("The smallest edge difference is numerically 0.")
        decimal = int(-np.log10(dedges_min)) + 6
        # Find which points are on the rightmost edge.
        on_edge = np.where(
            (sample[:, i] >= edges[i][-1])
            & (np.around(sample[:, i], decimal) == np.around(edges[i][-1], decimal))
        )[0]
        # Shift these points one bin to the left.
        sampBin[i][on_edge] -= 1

    if expand_binnumbers:
        binnumbers = np.stack(sampBin)
    else:
        # Compute the sample indices in the flattened statistic matrix.
        binnumbers = np.ravel_multi_index(sampBin, nbin)

    return binnumbers


def bin_dd(sample, bins=10, range=None, expand_binnumbers=True, nan_policy="raise"):
    """
    Bins N-dimensional data. Arguments have the same meaning as in
    scipy.stats.binned_statistic_dd, except that here expand_binnumbers=True by
    default, and that if expand_binnumbers=True and N = 1, then binnumbers is 2D
    instead of 1D as in scipy.stats.binned_statistic_dd. nan_policy can be 'raise' or
    'omit'.
    If nan_policy='raise', then ValueError is raised if sample contains any NaNs (this
    is slightly different from the default behavior of scipy.stats.binned_statistic_dd)
    If nan_policy='omit', NaNs are sorted into the rightmost bin.
    """

    if nan_policy not in ["raise", "omit"]:
        raise ValueError(f"nan_policy must be 'raise' or 'omit', but {nan_policy=}.")

    try:
        bins = index(bins)
    except TypeError:
        # bins is not an integer
        pass
    # If bins was an integer-like object, now it is an actual Python int.

    # NOTE: for _bin_edges(), see e.g. gh-11365
    if nan_policy == "raise" and not np.isfinite(sample).all():
        raise ValueError(f"{sample!r} contains non-finite values.")

    # `Ndim` is the number of dimensions (e.g. `2` for `binned_statistic_2d`)
    # This code is based on np.histogramdd
    try:
        # `sample` is an ND-array.
        _, Ndim = sample.shape
    except (AttributeError, ValueError):
        # `sample` is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        _, Ndim = sample.shape

    try:
        M = len(bins)
        if M != Ndim:
            raise AttributeError(
                "The dimension of bins must be equal to the dimension of the sample x."
            )
    except TypeError:
        bins = Ndim * [bins]

    nbin, edges, dedges = _bin_edges(sample, bins, range)
    binnumbers = _bin_numbers(
        sample, nbin, edges, dedges, expand_binnumbers=expand_binnumbers
    )

    centers = [
        np.array([np.nan] + list(0.5 * (e[:-1] + e[1:])) + [np.nan]) for e in edges
    ]
    return binnumbers, centers, edges


def bin(sample, bins=10, range=None, nan_policy="raise"):
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1:
        bins = [np.asarray(bins, float)]

    if range is not None and len(range) == 2:
        range = [range]

    bin_nums, centers, edges = bin_dd(
        sample, bins=bins, range=range, nan_policy=nan_policy
    )
    return bin_nums[0], centers[0], edges[0]

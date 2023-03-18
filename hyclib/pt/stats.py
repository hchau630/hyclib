from operator import index
import builtins

import torch
import numpy as np

def nanmax(t, dim=None):
    t = t.nan_to_num(nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
    if dim is None:
        return t.max()
    return t.max(dim=dim)

def nanmin(t, dim=None):
    t = t.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf)
    if dim is None:
        return t.min()
    return t.min(dim=dim)

def _bin_edges(sample, bins=None, range=None):
    """ 
    Create edge arrays
    """
    Dlen, Ndim = sample.shape # Dlen is batch size, Ndim is number of dimensions
    device = sample.device

    edges = Ndim * [None]         # Bin edges for each dim (will be 2D array)
    dedges = Ndim * [None]        # Spacing between edges (will be 2D array)

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        smin = np.atleast_1d(nanmin(sample, dim=0).values.detach().cpu().numpy())
        smax = np.atleast_1d(nanmax(sample, dim=0).values.detach().cpu().numpy())
    else:
        if len(range) != Ndim:
            raise ValueError(
                f"range given for {len(range)} dimensions; {Ndim} required")
        smin = np.empty(Ndim)
        smax = np.empty(Ndim)
        for i in builtins.range(Ndim):
            if range[i][1] < range[i][0]:
                raise ValueError(
                    "In {}range, start must be <= stop".format(
                        f"dimension {i + 1} of " if Ndim > 1 else ""))
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in builtins.range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Preserve sample floating point precision in bin edges
    edges_dtype = (sample.dtype if torch.is_floating_point(sample)
                   else torch.float)

    # Create edge arrays
    for i in builtins.range(Ndim):
        bins_i = torch.as_tensor(bins[i])
        if bins_i.ndim == 0: # scalar, interpret as number of bins
            edges[i] = torch.linspace(smin[i], smax[i], bins_i.item() + 1,
                                      dtype=edges_dtype, device=device)
        else:
            edges[i] = torch.as_tensor(bins_i, dtype=edges_dtype, device=device)
        dedges[i] = torch.diff(edges[i])

    return edges, dedges


def _bin_numbers(sample, edges, dedges):
    """Compute the bin number each sample falls into, in each dimension
    """
    Dlen, Ndim = sample.shape

    sampBin = [
        torch.bucketize(sample[:, i], edges[i], right=True) # `right` argument has opposite meaning in torch.bucketize vs np.digitize
        for i in range(Ndim)
    ]

    # Using `bucketize`, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    for i in range(Ndim):
        # Find the rounding precision
        dedges_min = dedges[i].min()
        if dedges_min == 0:
            raise ValueError('The smallest edge difference is numerically 0.')
        if dedges_min < 0:
            raise ValueError('Edges must be monotonically increasing.')
            
        decimal = -torch.log10(dedges_min).long().item() + 6
        # Find which points are on the rightmost edge.
        on_edge = ((sample[:, i] >= edges[i][-1]) &
                   (torch.round(sample[:, i], decimals=decimal) ==
                    torch.round(edges[i][-1], decimals=decimal))).nonzero().squeeze()
        # Shift these points one bin to the left.
        sampBin[i][on_edge] -= 1

    return torch.stack(sampBin)

def bin_dd(sample, bins=10, range=None, nan_policy='raise'):
    """
    Bins N-dimensional data. Arguments have the same meaning as in scipy.stats.binned_statistic_dd,
    except that here expand_binnumbers=True by default, and that if expand_binnumbers=True and N = 1,
    then binnumbers is 2D instead of 1D as in scipy.stats.binned_statistic_dd.
    nan_policy can be 'raise' or 'omit'. 
    If nan_policy='raise', then ValueError is raised if sample contains any NaNs (this is slightly different
    from the default behavior of scipy.stats.binned_statistic_dd).
    If nan_policy='omit', NaNs are sorted into the rightmost bin.
    """
    
    if nan_policy not in ['raise', 'omit']:
        raise ValueError(f"nan_policy must be 'raise' or 'omit', but {nan_policy=}.")
    
    try:
        bins = index(bins)
    except TypeError:
        # bins is not an integer
        pass
    # If bins was an integer-like object, now it is an actual Python int.
    
    # `Ndim` is the number of dimensions (e.g. `2` for `binned_statistic_2d`)
    # `Dlen` is the length of elements along each dimension.
    # This code is based on np.histogramdd
    try:
        # `sample` is an ND-array.
        Dlen, Ndim = sample.shape
    except (AttributeError, ValueError):
        # `sample` is a sequence of 1D arrays.
        sample = torch.stack(sample).t()
        Dlen, Ndim = sample.shape

    # NOTE: for _bin_edges(), see e.g. gh-11365
    if nan_policy == 'raise' and not torch.isfinite(sample).all():
        raise ValueError('%r contains non-finite values.' % (sample,))
        
    try:
        M = len(bins)
        if M != Ndim:
            raise AttributeError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        bins = Ndim * [bins]
        
    edges, dedges = _bin_edges(sample, bins, range)
    binnumbers = _bin_numbers(sample, edges, dedges)
        
    centers = []
    for e in edges:
        c = torch.empty(len(e)+1, dtype=e.dtype, device=e.device)
        c[0] = torch.nan
        c[-1] = torch.nan
        c[1:-1] = 0.5 * (e[1:] + e[:-1])
        centers.append(c)
    
    return binnumbers, centers, edges

def bin(sample, bins=10, range=None, **kwargs):
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1:
        bins = [torch.as_tensor(bins, dtype=torch.float)]

    if range is not None:
        if len(range) == 2:
            range = [range]
            
    binnumbers, centers, edges = bin_dd([sample], bins=bins, range=range, **kwargs)
    return binnumbers[0], centers[0], edges[0]
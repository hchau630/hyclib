import numpy as np

from ..itertools import flatten_seq

__all__ = [
    'isconst',
    'meshgrid_dd',
    'meshgrid',
]

def isconst(x, axis=None, **kwargs):
    if axis is None:
        x = x.reshape(-1)
    else:
        if isinstance(axis, int):
            axis = [axis]
        axis = sorted([d % x.ndim for d in axis])[::-1]
        for d in axis:
            x = np.moveaxis(x, d,-1)
        x = x.reshape(*x.shape[:-len(axis)],-1)
        
    return np.isclose(x[...,:-1], x[...,1:], **kwargs).all(axis=-1)

def meshgrid_dd(*arrs):
    """
    Generalized np.meshgrid
    Mesh together list of arrays of shapes (n_1_1,...,n_1_{M_1},N_1), (n_2_1,...,n_2_{M_2},N_2), ..., (n_P_1, ..., n_P_{M_P},N_P)
    Returns arrays of shapes 
    (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,n_P_1, ..., n_P_{M_P},N_1),
    (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,n_P_1, ..., n_P_{M_P},N_2),
    ...
    (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,n_P_1, ..., n_P_{M_P},N_P)
    """
    sizes = [list(arr.shape[:-1]) for arr in arrs] # [[n_1,...,n_{M_1}],[n_1,...,.n_{M_2}],...]
    Ms = np.array([arr.ndim - 1 for arr in arrs]) # [M_1, M_2, ...]
    M_befores = np.cumsum(np.insert(Ms[:-1],0,0))
    M_afters = np.sum(Ms) - np.cumsum(Ms)
    Ns = [arr.shape[-1] for arr in arrs]
    shapes = [[1]*M_befores[i]+sizes[i]+[1]*M_afters[i]+[Ns[i]] for i, arr in enumerate(arrs)]
    expanded_arrs = [np.broadcast_to(arr.reshape(shapes[i]), flatten_seq(sizes)+[Ns[i]]) for i, arr in enumerate(arrs)]
    return expanded_arrs

def meshgrid(*arrs, **kwargs):
    """
    Generalized np.meshgrid
    Mesh together list of arrays of shapes (n_1_1,...,n_1_{M_1}), (n_2_1,...,n_2_{M_2}), ..., (n_P_1, ..., n_P_{M_P})
    Returns arrays of shapes 
    (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,n_P_1, ..., n_P_{M_P}),
    (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,n_P_1, ..., n_P_{M_P}),
    ...
    (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,n_P_1, ..., n_P_{M_P})
    
    IMPORTANT: By default, indexing='ij' rather than 'xy' as in np.meshgrid. This follows the pytorch convention.
    """
    default_kwargs = {
        'indexing': 'ij',
    }
    kwargs = default_kwargs | kwargs
    invalid_keys = set(kwargs.keys()) - {'indexing'}
    if len(invalid_keys) > 0:
        raise TypeError(f"meshgrid() got an unexpected keyword argument '{invalid_keys.pop()}'")
    indexing = kwargs['indexing']    
        
    arrs = (arr[...,None] for arr in arrs)
    arrs = meshgrid_dd(*arrs)
    arrs = [arr.squeeze() for arr in arrs]
    
    if indexing == 'xy':
        arrs = [np.swapaxes(arr, 0, 1) for arr in arrs]
        
    return arrs
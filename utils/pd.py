import functools

import numpy as np
import torch
import pandas as pd

from .np import meshgrid_dd

def _formatter(x, float_format=None, verbose=False):
    if isinstance(x, np.ndarray):
        if verbose:
            return f"np.ndarray {x.shape} ({x.dtype})"
        return f"{x.shape}"
    if isinstance(x, torch.Tensor):
        if verbose:
            return f"torch.Tensor {tuple(x.shape)} ({x.dtype})"
        return f"{tuple(x.shape)}"
    if isinstance(x, tuple):
        if verbose:
            return f"tuple ({len(x)}) {tuple([type(e).__name__ for e in x])}"
        return f"tuple ({len(x)})"
    if isinstance(x, list):
        if verbose:
            return f"list ({len(x)}) {[type(e).__name__ for e in x]}"
        return f"list ({len(x)})"
    if isinstance(x, dict):
        if verbose:
            d = {k: type(v).__name__ for k, v in x.items()}
            return f"dict ({len(x)}) {d}"
        return f"dict ({len(x)})"
    if isinstance(x, np.inexact):
        if float_format is None:
            return f'{x:.4f}'
        return float_format(x)
    return str(x)

def display(df, float_format=None, verbose=False, **kwargs):
    default_kwargs = {
        'max_rows': 6,
        'show_dimensions': True,
        'formatters': {k: functools.partial(_formatter, float_format=float_format, verbose=verbose) for k in df.columns},
    }
    default_kwargs.update(kwargs)

    try:
        get_ipython
        from IPython.display import display as ipy_display
        from IPython.core.display import HTML
    except:
        raise RuntimeError("display only works in jupyter notebook")
        
    ipy_display(HTML(df.to_html(**default_kwargs)))
    
def revert_dtypes(df):
    dtypes = {}
    for k, v in df.dtypes.items():
        v = str(v)
        if 'Int' in v:
            if df[k].isnull().any():
                dtypes[k] = 'float'
            else:
                dtypes[k] = v.lower()
        elif 'Float' in v:
            dtypes[k] = v.lower()
        elif 'boolean' in v:
            dtypes[k] = 'bool'
        elif pd.api.types.is_numeric_dtype(v): # could be complex, for example
            dtypes[k] = v
        else:
            dtypes[k] = 'object'
            
    return df.astype(dtypes)

def cross_join(*dfs, maintain_dtypes=True):
    """
    Efficient cross/cartesian product of numeric dataframes. Should be faster than df.join()
    If maintain_dtypes=True, will ensure that the resulting DataFrame has the same dtypes as the original DataFrames.
    However, maintain_dtypes=True is very slow in general.
    """
    columns = [column for df in dfs for column in df.columns]
    dtypes = {k: v for df in dfs for k, v in df.dtypes.items()}
    dfs = meshgrid_dd(*(revert_dtypes(df).to_numpy() for df in dfs)) # reverting dtypes makes meshgrid faster if all columns are numeric
    df = np.concatenate([df.reshape(-1,df.shape[-1]) for df in dfs], axis=-1)
    df = pd.DataFrame(df, columns=columns)
    if maintain_dtypes:
        return df.astype(dtypes)
    return df
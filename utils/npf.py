import numbers
import functools

import numpy as np

class ndarray(np.ndarray):
    def __array_finalize__(self, obj):
        if obj is None: return
        self.op_expr = getattr(obj, 'op_expr', '{}')
    
    def __call__(self, x):
        if isinstance(x, numbers.Number):
            x_shape = tuple()
            x = np.array([x])
        elif isinstance(x, np.ndarray):
            x_shape = x.shape
            x = x.reshape(-1)
        else:
            raise NotImplementedError(f"Currently only accepts scalar or np.ndarray inputs, but got {type(x)}")
        
        res = []
        for idx in np.ndindex(self.shape):
            res.append(self[idx](x))
        res = np.moveaxis(np.array(res), -1, 0).reshape((*x_shape, *self.shape))
        return res
    
    def __matmul__(self, g):
        result = matmul(self, g)
        result.op_expr = '({} @ {})'
        return result
    
    def __repr__(self):
        op_expr = self.op_expr.format(*[f'f{i}' for i in range(self.op_expr.count('{}'))])
        return f"npf.ndarray(shape={self.shape}, op_expr={op_expr})"
    
    def __str__(self):
        return repr(self)
    
def array(*args, **kwargs):
    arr = np.array(*args, **kwargs).view(ndarray)
    return arr

def empty(*args, dtype=object, **kwargs):
    arr = np.empty(*args, dtype=dtype, **kwargs).view(ndarray)
    return arr

### npf.ndarray ufuncs ###

def sum(x, **kwargs):
    x = array(x)
    add_ufunc = get_ufunc('__add__')
    result = add_ufunc.reduce(x, **kwargs)
    result.op_expr = f'sum{x.op_expr}'
    return result

@functools.partial(np.vectorize, signature='(a,b),(b,c)->(a,c)')
def _matmul(x, y):
    assert x.ndim == 2 and y.ndim == 2 and x.shape[1] == y.shape[0]
    z = empty((x.shape[0], y.shape[1]))
    for i, j in np.ndindex(z.shape):
        z[i,j] = sum(x[i,:] * y[:,j])
    return z

def matmul(x, y):
    squeeze_x, squeeze_y = False, False
    if x.ndim == 1:
        x = x[None,:]
        squeeze_x = True
    if y.ndim == 1:
        y = y[:,None]
        squeeze_y = True
    result = _matmul(x, y)
    if squeeze_x:
        result = result.squeeze(-2)
    if squeeze_y:
        result = result.squeeze(-1)
    return result

def trace(A):
    return sum(np.diagonal(A, axis1=-2, axis2=-1))

@functools.partial(np.vectorize, signature='(m,n)->()')
def det(A):
    assert A.ndim == 2 and A.shape[0] == A.shape[1], f'A ({shape=}) is not a square matrix'
    N = A.shape[0]
    
    if N == 1:
        return array(A.item())
    
    return sum([(-1)**i * array(A[0,i]) * _minor(A, 0, i) for i in range(N)])

def _minor(A, i, j):
    assert A.ndim == 2 and A.shape[0] == A.shape[1], f'A ({shape=}) is not a square matrix'
    
    return det(np.delete(np.delete(A, i, axis=0), j, axis=1))

@functools.partial(np.vectorize, signature='(m,n)->(m,n)')
def adj(A):
    assert A.ndim == 2 and A.shape[0] == A.shape[1], f'A ({shape=}) is not a square matrix'
    N = A.shape[0]
    
    return array([[(-1)**(i+j) * _minor(A, i, j) for j in range(N)] for i in range(N)]).T

def inv(A):
    return adj(A)/det(A)

def get_ufunc(op_name):
    def ufunc(f, g):
        if not callable(f):
            raise NotImplementedError(f'f must be callable, but {type(f)} found.')
        if isinstance(g, numbers.Number):
            h = lambda x: getattr(f(x), op_name)(g)
        elif callable(g):
            h = lambda x: getattr(f(x), op_name)(g(x))
        else:
            raise NotImplementedError(f'g must be callable or a numbers.Number, but {type(g)} found.')
        return h
    
    return np.frompyfunc(ufunc,2,1)

### add methods for common mathematical operators to npf.ndarray ###

OP_EXPR_DICT = {
    '__add__': '+',
    '__sub__': '-',
    '__mul__': '*',
    '__truediv__': '/',
    '__pow__': '**',
    '__radd__': '+',
    '__rsub__': '-',
    '__rmul__': '*',
    '__rtruediv__': '/',
    '__rpow__': '**',
}

def get_array_op(op_name):
    ufunc = get_ufunc(op_name)
    
    def array_op(self, g):
        result = ufunc(self, g)
        if isinstance(g, ndarray):
            result.op_expr = f'({self.op_expr} {OP_EXPR_DICT[op_name]} {g.op_expr})'
        else:
            if op_name.startswith('__r'):
                result.op_expr = f'({g} {OP_EXPR_DICT[op_name]} {self.op_expr})'
            else:
                result.op_expr = f'({self.op_expr} {OP_EXPR_DICT[op_name]} {g})'
        return result
    
    return array_op

for op_name in OP_EXPR_DICT.keys():
    setattr(ndarray, op_name, get_array_op(op_name))
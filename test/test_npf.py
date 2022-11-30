import numpy as np

import utils

def check_npf_array(arr, expected_func_arr):
    expected_func_arr = np.array(expected_func_arr)
    expected_shape = expected_func_arr.shape
    assert isinstance(arr, utils.npf.ndarray)
    assert arr.shape == expected_shape
    xs = np.array([-2.0,-1.0,0.0,1.0,2.0,-2.0j,-1.0j,1.0j,2.0j,2.0-2.0j,2.0-1.0j,2.0+0.0j,2.0+1.0j,2.0+2.0j])
    for x in xs:
        res = arr(x)
        assert res.shape == expected_shape
        assert np.allclose(res.reshape(-1), np.array([func(x) for func in expected_func_arr.reshape(-1)]))
    res = arr(xs)
    assert res.shape == (*xs.shape, *expected_shape)
    assert np.allclose(res.reshape(-1), np.array([[func(x) for func in expected_func_arr.reshape(-1)] for x in xs]).reshape(-1))
    xs = xs.reshape(2,7)
    assert arr(xs).shape == (*xs.shape, *expected_shape)

def test():
    f1, f2 = lambda x: 2*x, lambda x: 3*x
    g1, g2 = lambda x: 4*x, lambda x: 5*x
    a = utils.npf.array([[f1, f2]])
    b = utils.npf.array([g1, g2])
    
    print(a, b)
    check_npf_array(a, [[f1, f2]])
    check_npf_array(b, [g1, g2])
    
    c = a+b
    print(c)
    check_npf_array(c, [[lambda x: f1(x)+g1(x), lambda x: f2(x)+g2(x)]])
    
    d = c[:,:1]
    print(d)
    check_npf_array(d, [[lambda x: f1(x)+g1(x)]])
    
    a[0] = lambda x: x
    b[0] = lambda x: x
    print(a, b)
    check_npf_array(a, [[lambda x: x, lambda x: x]])
    check_npf_array(b, [lambda x: x, g2])
    check_npf_array(a+b, [[lambda x: x+x, lambda x: x+g2(x)]])
    check_npf_array(c, [[lambda x: f1(x)+g1(x), lambda x: f2(x)+g2(x)]])
    check_npf_array(d, [[lambda x: f1(x)+g1(x)]])
    
    a = utils.npf.array([[f1, f2]])
    b = utils.npf.array([g1, g2])
    
    c = b+a
    print(c)
    check_npf_array(c, [[lambda x: f1(x)+g1(x), lambda x: f2(x)+g2(x)]])
    
    c = a-b
    print(c)
    check_npf_array(c, [[lambda x: f1(x)-g1(x), lambda x: f2(x)-g2(x)]])
    
    c = a*b
    print(c)
    check_npf_array(c, [[lambda x: f1(x)*g1(x), lambda x: f2(x)*g2(x)]])
    
    g1, g2 = lambda x: 4*x+3, lambda x: 5*x+4
    b = utils.npf.array([g1, g2])
    c = a/b
    print(c)
    check_npf_array(c, [[lambda x: f1(x)/g1(x), lambda x: f2(x)/g2(x)]])
    
    c = a**b
    print(c)
    check_npf_array(c, [[lambda x: f1(x)**g1(x), lambda x: f2(x)**g2(x)]])
    
    c = b**((a+b)*b-a)
    print(c)
    
    b = np.pi
    print(a+b)
    print(b/a)
    
def test_linalg():
    a = utils.npf.array([[lambda x: 2*x, lambda x: 3*x],[lambda x: 4*x, lambda x: 5*x]])
    b = utils.npf.array([[lambda x: 6*x, lambda x: 7*x],[lambda x: 8*x, lambda x: 9*x]])
    x = np.linspace(1,2,num=3)
    
    result, expected = (a @ b)(x), a(x) @ b(x)
    assert (result.shape == expected.shape) and np.allclose(result, expected)
    
    result, expected = utils.npf.trace(a)(x), np.trace(a(x), axis1=-2, axis2=-1)
    assert (result.shape == expected.shape) and np.allclose(result, expected)
    
    result, expected = utils.npf.det(a)(x), np.linalg.det(a(x))
    assert (result.shape == expected.shape) and np.allclose(result, expected)
    
    result, expected = utils.npf.adj(a)(x), np.linalg.inv(a(x))*np.linalg.det(a(x))[:,None,None]
    assert (result.shape == expected.shape) and np.allclose(result, expected)
    
    result, expected = utils.npf.inv(a)(x), np.linalg.inv(a(x))
    assert (result.shape == expected.shape) and np.allclose(result, expected)

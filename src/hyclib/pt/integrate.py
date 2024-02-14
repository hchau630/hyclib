from scipy.integrate._quadrature import _cached_roots_legendre


def fixed_quad(func, a, b, args=(), n=5):
    """
    Like scipy.integrate.fixed_quad, but for tensor inputs a, b

    Args:
        func: Callable[[torch.Tensor, ...], torch.Tensor]. Integrand.
        a, b: torch.Tensor. Lower and upper limits of integral.
        args (optional): tuple. Additional arguments passed to func.
        n (optional): int. Order of Gauss-Legendre quadrature.
    """
    x, w = _cached_roots_legendre(n)
    x = x.real

    d = (b - a) / 2.0
    return d * sum(wi * func(d * (xi + 1) + a, *args) for xi, wi in zip(x, w)), None
    
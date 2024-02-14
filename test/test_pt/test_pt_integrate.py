import pytest
import torch
from scipy import integrate

import hyclib as lib


@pytest.mark.parametrize('n', list(range(1, 7)))
def test_fixed_quad(n):
    f = lambda x: x**1.5
    a = torch.linspace(1, 10, 10)
    b = torch.linspace(2, 11, 10)

    out = lib.pt.integrate.fixed_quad(f, a, b, n=n)[0]
    expected = torch.tensor([integrate.fixed_quad(f, ai.item(), bi.item(), n=n)[0].item() for ai, bi in zip(a, b)])

    torch.testing.assert_close(out, expected)
    
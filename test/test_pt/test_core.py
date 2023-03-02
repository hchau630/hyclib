import torch
import numpy as np
import pytest

import utils

@pytest.mark.parametrize('w', [
    torch.tensor([[ 0.6455,  torch.nan, -0.9379, -0.0359, -0.1765, -0.3408, -1.9616,  0.1720,
                    0.4691, -0.5099],
                  [ 1.2700,  0.1971, -0.8720, -0.6560,  1.4493,  0.3326, -0.9554,  0.7140,
                   -1.5305, -1.1244]], requires_grad=True),
    torch.tensor([[ 0.6455,  torch.nan, -0.9379, -0.0359, -0.1765, -0.3408, -1.9616,  0.1720,
                    0.4691, -0.5099],
                  [ 1.2700,  0.1971, -0.8720, -0.6560,  1.4493,  0.3326, -0.9554,  0.7140,
                   -1.5305, -1.1244]]),
    torch.tensor([ 0.6455,  torch.nan, -0.9379, -0.0359, -0.1765, -0.3408, -1.9616,  0.1720,
                   0.4691, -0.5099]),
])
def test_bincount(w):
    a = torch.tensor([0,0,1,1,1,0,2,1,2,0])
    t = utils.pt.bincount(a, weights=w)
    
    if w.ndim == 2:
        arr = []
        for wi in w:
            arr.append(np.bincount(a.numpy(), weights=wi.detach().numpy()))
        arr = np.stack(arr)
    elif w.ndim == 1:
        arr = np.bincount(a.numpy(), weights=w.detach().numpy())
    else:
        raise RuntimeError()
    
    assert np.allclose(t.detach().numpy(), arr, equal_nan=True)
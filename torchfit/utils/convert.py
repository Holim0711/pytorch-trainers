import torch
from collections import abc


def _convert(x, device):
    """ default conversion function """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, abc.Mapping):
        return {k: _convert(x[k], device) for k in x}
    if isinstance(x, tuple) and hasattr(x, '_fields'):
        return type(x)(*(_convert(v, device) for v in x))
    if isinstance(x, abc.Sequence) and not isinstance(x, str):
        return type(x)(_convert(v, device) for v in x)
    return x

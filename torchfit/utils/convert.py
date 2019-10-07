import torch
from collections import abc


def _convert(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, tuple) and hasattr(x, '_fields'):
        return x.__class__(*(_convert(v, device) for v in x))
    if isinstance(x, abc.Sequence) and not isinstance(x, str):
        return x.__class__(_convert(v, device) for v in x)
    if isinstance(x, abc.Mapping):
        return x.__class__((k, _convert(x[k], device)) for k in x)
    return x

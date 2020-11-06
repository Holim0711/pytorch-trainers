import torch
from collections import abc


def convert(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, abc.Mapping):
        return {k: convert(v, device) for k, v in x.items()}
    if isinstance(x, abc.Sequence) and not isinstance(x, str):
        return [convert(v, device) for v in x]
    return x

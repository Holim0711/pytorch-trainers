from torch import Tensor
from collections.abc import Sequence, Mapping
from tqdm import tqdm


def convert(x, device):
    if isinstance(x, Tensor):
        return x.to(device)
    if isinstance(x, tuple) and hasattr(x, '_fields'):
        return x.__class__(*(convert(v, device) for v in x))
    if isinstance(x, Sequence) and not isinstance(x, str):
        return x.__class__(convert(v, device) for v in x)
    if isinstance(x, Mapping):
        return x.__class__((k, convert(x[k], device)) for k in x)
    return x


def iterate(dataloader, device, verbose):
    if verbose:
        dataloader = tqdm(dataloader)
    for x in dataloader:
        yield convert(x, device)

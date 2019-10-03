from torch import Tensor
from inspect import getargspec


__all__ = [
    '_convert_data',
    '_check_named_params',
]


def _convert_data(x, device):
    if isinstance(x, Tensor):
        return x.to(device)
    elif isinstance(x, tuple):
        return tuple(_convert_data(v, device) for v in x)
    elif isinstance(x, list):
        return list(_convert_data(v, device) for v in x)
    elif isinstance(x, set):
        return set(_convert_data(v, device) for v in x)
    elif isinstance(x, dict):
        return {k: _convert_data(v, device) for k, v in x.items()}
    return x


def _check_named_params(func, param_names):
    args = getargspec(func).args
    miss = [x for x in param_names if x not in args]
    if miss:
        msg = f'{func.__name__}() missing {len(miss)} named parameter'
        msg += 's: ' if len(miss) > 1 else ': '
        raise TypeError(msg + ', '.join(miss))

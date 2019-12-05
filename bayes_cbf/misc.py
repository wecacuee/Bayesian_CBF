"""
Home for functions/classes that haven't find a home of their own
"""
import torch
import inspect
from functools import wraps, partial

t_hstack = partial(torch.cat, dim=-1)
"""
Similar to np.hstack
"""

t_vstack = partial(torch.cat, dim=-2)
"""
Similar to np.vstack
"""


def to_numpy(x):
    return x.detach().cpu().double().numpy()


def t_jac(f_x, x):
    if f_x.ndim:
        return torch.cat(
            [torch.autograd.grad(f_x[i], x, retain_graph=True)[0].unsqueeze(0)
            for i in range(f_x.shape[0])], dim=0)
    else:
        return torch.autograd.grad(f_x, x, retain_graph=True)[0]


def store_args(method):
    argspec = inspect.getfullargspec(method)
    @wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if argspec.defaults is not None:
          for name, val in zip(argspec.args[::-1], argspec.defaults[::-1]):
              setattr(self, name, val)
        if argspec.kwonlydefaults and args.kwonlyargs:
            for name, val in zip(argspec.kwonlyargs, argspec.kwonlydefaults):
                setattr(self, name, val)
        for name, val in zip(argspec.args, args):
            setattr(self, name, val)
        for name, val in kwargs.items():
            setattr(self, name, val)

        method(self, *args, **kwargs)

    return wrapped_method

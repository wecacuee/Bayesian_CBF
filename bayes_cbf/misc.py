"""
Home for functions/classes that haven't find a home of their own
"""
import math
from functools import wraps, partial
from itertools import zip_longest
from abc import ABC, abstractmethod
from contextlib import contextmanager
import inspect

import torch

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


def t_jac(f_x, x, retain_graph=False):
    if f_x.ndim:
        return torch.cat(
            [torch.autograd.grad(f_x[i], x, retain_graph=True)[0].unsqueeze(0)
             for i in range(f_x.shape[0])], dim=0)
    else:
        return torch.autograd.grad(f_x, x, retain_graph=retain_graph)[0]


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


def torch_kron(A, B):
    """
    >>> B = torch.rand(5,3,3)
    >>> A = torch.rand(5,2,2)
    >>> AB = torch_kron(A, B)
    >>> torch.allclose(AB[1, :3, :3] , A[1, 0,0] * B[1, ...])
    True
    >>> BA = torch_kron(B, A)
    >>> torch.allclose(BA[1, :2, :2] , B[1, 0,0] * A[1, ...])
    True
    """
    b = B.shape[0]
    assert A.shape[0] == b
    B_shape = sum([[1, si] for si in B.shape[1:]], [])
    A_shape = sum([[si, 1] for si in A.shape[1:]], [])
    kron_shape = [a*b for a, b in zip_longest(A.shape[1:], B.shape[1:], fillvalue=1)]
    return (A.reshape(b, *A_shape) * B.reshape(b, *B_shape)).reshape(b, *kron_shape)


class DynamicsModel(ABC):
    """
    Represents mode of the form:

    ẋ = f(x) + g(x)u
    """
    @property
    @abstractmethod
    def ctrl_size(self):
        """
        Dimension of ctrl
        """

    @property
    @abstractmethod
    def state_size(self):
        """
        Dimension of state
        """

    @abstractmethod
    def f_func(self, X):
        """
        ẋ = f(x) + g(x)u

        @param: X : d x self.state_size vector or self.state_size vector
        @returns: f(X)
        """

    @abstractmethod
    def g_func(self, X):
        """
        ẋ = f(x) + g(x)u

        @param: X : d x self.state_size vector or self.state_size vector
        @returns: g(X)
        """

    def normalize_state(self, X_in):
        return X_in

class ZeroDynamicsModel(DynamicsModel):
    def __init__(self, m, n):
        self.m = m
        self.n = n

    @property
    def ctrl_size(self):
        return self.m

    @property
    def state_size(self):
        return self.n

    def f_func(self, X):
        return (torch.zeros((self.n,))
                if X.dim() <= 1
                else torch.zeros(X.shape))

    def g_func(self, X):
        return torch.zeros((*X.shape, self.m))

@contextmanager
def variable_required_grad(x):
    """
    creates context for x requiring gradient
    """
    old_x_requires_grad = x.requires_grad
    try:
        x.requires_grad_(True)
        yield x
    finally:
        x.requires_grad_(old_x_requires_grad)


def t_hessian(f, x, xp, grad_check=True):
    """
    Computes second derivative, Hessian
    """
    with variable_required_grad(x):
        with variable_required_grad(xp):
            grad_k_func = lambda xs, xt: torch.autograd.grad(
                f(xs, xt), xs, create_graph=True)[0]
            Hxx_k = t_jac(grad_k_func(x, xp), xp)
    return Hxx_k


def gradgradcheck(f2, x):
    xp = x.detach().clone()

    # assuming first analytical derivative is correct
    grad_k_func_1 = lambda i, xs, xt: torch.autograd.grad(
        f2(xs, xt), xs, create_graph=True)[0][i]

    with variable_required_grad(x):
        with variable_required_grad(xp):
            for i in range(x.shape[0]):
                torch.autograd.gradcheck(
                    partial(grad_k_func_1, i, x), xp)

def epsilon(i, interpolate={0: 1, 1000: 0.01}):
    """
    """
    ((si,sv), (ei, ev)) = list(interpolate.items())
    return math.exp((i-si)/(ei-si)*(math.log(ev)-math.log(sv)) + math.log(sv))


"""
Home for functions/classes that haven't find a home of their own
"""
from datetime import datetime
import math
from functools import wraps, partial
from itertools import zip_longest
from abc import ABC, abstractmethod, abstractproperty
from contextlib import contextmanager
import inspect
import io
import subprocess
import os
import os.path as osp

from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto

import numpy as np
import torch
import bayes_cbf


t_hstack = partial(torch.cat, dim=-1)
"""
Similar to np.hstack
"""

t_vstack = partial(torch.cat, dim=-2)
"""
Similar to np.vstack
"""


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().double().numpy()
    else:
        return x


def t_jac(f_x, x, retain_graph=False, **kw):
    if f_x.ndim:
        return torch.cat(
            [torch.autograd.grad(f_x[i], x, retain_graph=True, **kw)[0].unsqueeze(0)
             for i in range(f_x.shape[0])], dim=0)
    else:
        return torch.autograd.grad(f_x, x, retain_graph=retain_graph, **kw)[0]


def store_args(method, skip=[]):
    argspec = inspect.getfullargspec(method)
    @wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if argspec.defaults is not None:
          for name, val in zip(argspec.args[::-1], argspec.defaults[::-1]):
              if name not in skip:
                  setattr(self, name, val)
        if argspec.kwonlydefaults and args.kwonlyargs:
            for name, val in zip(argspec.kwonlyargs, argspec.kwonlydefaults):
                if name not in skip:
                    setattr(self, name, val)
        for name, val in zip(argspec.args[1:], args):
            if name not in skip:
                setattr(self, name, val)
        for name, val in kwargs.items():
            if name not in skip:
                setattr(self, name, val)

        method(self, *args, **kwargs)

    return wrapped_method


def torch_kron(A, B, batch_dims=1):
    """
    >>> B = torch.rand(5,3,3)
    >>> A = torch.rand(5,2,2)
    >>> AB = torch_kron(A, B)
    >>> torch.allclose(AB[1, :3, :3] , A[1, 0,0] * B[1, ...])
    True
    >>> BA = torch_kron(B, A)
    >>> torch.allclose(BA[1, :2, :2] , B[1, 0,0] * A[1, ...])
    True
    >>> B = torch.rand(3,2)
    >>> A = torch.rand(2,3)
    >>> AB = torch_kron(A, B, batch_dims=0)
    >>> AB.shape = (6, 6)
    True
    """
    assert A.ndim == B.ndim
    b = B.shape[0:batch_dims]
    #assert A.shape[0:batch_dims] == b
    a = A.shape[0:batch_dims]
    B_shape = sum([[1, si] for si in B.shape[batch_dims:]], [])
    A_shape = sum([[si, 1] for si in A.shape[batch_dims:]], [])
    kron_shape = [a*b for a, b in zip_longest(A.shape[batch_dims:],
                                              B.shape[batch_dims:], fillvalue=1)]
    kron = (A.reshape(*a, *A_shape) * B.reshape(*b, *B_shape))
    k = kron.shape[:batch_dims]
    return kron.reshape(*k, *kron_shape)


class DynamicsModel(ABC):
    """
    Represents mode of the form:

    ẋ = f(x) + g(x)u
    """
    def __init__(self):
        self._state = None

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

    def forward(self, x, u):
        if x.ndim == 1:
            X_b = x.unsqueeze(0)
        else:
            X_b = x

        if u.ndim == 1:
            U_b = u.unsqueeze(0).unsqueeze(-1)
        elif u.ndim == 2:
            U_b = u.unsqueeze(0)
        else:
            U_b = u

        Xdot_b = self.f_func(X_b) + self.g_func(X_b).bmm(U_b).squeeze(-1)
        if x.ndim == 1:
            xdot = Xdot_b.squeeze(0)
        else:
            xdot = Xdot_b

        return xdot

    def step(self, u, dt):
        x = self._state
        xdot = self.forward(x, u)
        xtp1 = self.normalize_state(x + xdot * dt)
        self._state = xtp1
        return dict(x=xtp1, xdot=xdot)

    def set_init_state(self, x0):
        self._state = x0.clone()

    def F_func(self, X):
        return torch.cat([self.f_func(X).unsqueeze(-1), self.g_func(X)], dim=-1)

class BayesianDynamicsModel(DynamicsModel):
    @abstractmethod
    def fu_func_gp(self, U):
        """
        return a GaussianProcessBase
        """

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
                else torch.zeros(X.shape)) * X

    def g_func(self, X):
        return torch.zeros((*X.shape, self.m)) * X.unsqueeze(-1)


def isleaf(x):
    return x.grad_fn is None

@contextmanager
def variable_required_grad(x):
    """
    creates context for x requiring gradient
    """
    old_x_requires_grad = x.requires_grad
    if isleaf(x):
        xleaf = x
    else:
        xleaf = x.detach().clone()
    try:
        yield xleaf.requires_grad_(True)
    finally:
        if isleaf(x):
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


def get_affine_terms(func, x):
    with variable_required_grad(x):
        f_x = func(x)
        linear = torch.autograd.grad(f_x, x, create_graph=True)[0]
    with torch.no_grad():
        const = f_x - linear @ x
    return linear, const


def get_quadratic_terms(func, x):
    with variable_required_grad(x):
        f_x = func(x)
        linear_more = torch.autograd.grad(f_x, x, create_graph=True)[0]
        quad = t_jac(linear_more, x) / 2
    with torch.no_grad():
        linear = linear_more - 2 * quad @ x
        const = f_x - x.T @ quad @ x - linear @ x
    return quad, linear, const

def clip(x, min_, max_):
    return torch.max(torch.min(x, max_), min_)

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = Image.open(buf)
    # Add the batch dimension
    return torch.from_numpy(np.asarray(image))

def create_summary_writer(run_dir='data/runs/', exp_tags=[]):
    timestamp_tag = datetime.now().strftime("%m%d-%H%M")
    return SummaryWriter(run_dir + '/'
                         + '_'.join(
                             exp_tags + [timestamp_tag]))


def random_psd(m):
    M =  torch.rand(m, m)
    return M @ M.T


def normalize_radians(theta):
    return (theta + math.pi) % (2*math.pi) - math.pi

def make_tensor_summary(name, nparray):
    tensor_pb = TensorProto(dtype='DT_FLOAT',
                         float_val=nparray.reshape(-1).tolist(),
                         tensor_shape=TensorShapeProto(
                             dim=[TensorShapeProto.Dim(size=s)
                                  for s in nparray.shape]))
    return Summary(value=[Summary.Value(tag=name, tensor=tensor_pb)])


def add_tensors(summary_writer, tag, var_dict, t):
    for k, v in var_dict.items():
        summary_writer._get_file_writer().add_summary(
            make_tensor_summary("/".join((tag, k)),
                                v),
            t
        )


def gitdescribe(f):
    return subprocess.run("git describe --always".split(),
                          cwd=os.path.dirname(f) or '.',
                          stdout=subprocess.PIPE).stdout.decode('utf-8').strip()

def stream_tensorboard_scalars(event_file):
    loader = event_file_loader.EventFileLoader(event_file)
    for event in loader.Load():
        t = event.step
        if event.summary is not None and len(event.summary.value):
            val = event.summary.value[0]
            tag = val.tag
            value = val.simple_value or np.array(val.tensor.float_val).reshape(
                                                 [d.size for d in val.tensor.tensor_shape.dim])
            yield t, tag, value


def load_tensorboard_scalars(event_file):
    groupby_tag = dict()
    for t, tag, value in stream_tensorboard_scalars(event_file):
        groupby_tag.setdefault(tag, []).append( (t, value))
    return groupby_tag


class Logger(ABC):
    @abstractproperty
    def experiment_logs_dir(self):
        return "/tmp"

    @abstractmethod
    def add_scalars(self, tag, var_dict, t):
        pass

    @abstractmethod
    def add_tensors(self, tag, var_dict, t):
        pass

class NoLogger(Logger):
    @property
    def experiment_logs_dir(self):
        return "/tmp"

    def add_scalars(self, tag, var_dict, t):
        pass

    def add_tensors(self, tag, var_dict, t):
        pass

class TBLogger(Logger):
    def __init__(self, exp_tags, runs_dir='data/runs'):
        self.exp_tags = exp_tags
        self.runs_dir = runs_dir
        self.exp_dir = osp.join(runs_dir,
                                '_'.join(exp_tags + [bayes_cbf.__version__]))
        self.summary_writer = SummaryWriter(self.exp_dir)

    @property
    def experiment_logs_dir(self):
        return self.exp_dir

    def add_scalars(self, tag, var_dict, t):
        if not osp.exists(self.exp_dir): osp.makedirs(self.exp_dir)
        for k, v in var_dict.items():
            self.summary_writer.add_scalar("/".join((tag, k)), v, t)

    def add_tensors(self, tag, var_dict, t):
        if not osp.exists(self.exp_dir): osp.makedirs(self.exp_dir)
        add_tensors(self.summary_writer, tag, var_dict, t)

def ensuredirs(fpath):
    fdir = osp.dirname(fpath)
    if not osp.exists(fdir):
        os.makedirs(fdir)
    return fpath

def mkdir_savefig(fig, figpath):
    fig.savefig(ensuredirs(figpath))

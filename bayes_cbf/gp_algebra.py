from functools import partial
from abc import ABC, abstractmethod, abstractproperty
from importlib import import_module

import torch
from torch.distributions import MultivariateNormal
tgradcheck = import_module("torch.autograd.gradcheck") # global variable conflicts with module name

from bayes_cbf.misc import t_jac, variable_required_grad, t_hessian


class GaussianProcessBase(ABC):
    def __init__(self, mean, knl):
        self._mean = mean
        self._knl = knl

    @abstractproperty
    def shape(self):
        pass

    @abstractmethod
    def mean(self, x):
        return self._mean(x)

    @abstractmethod
    def knl(self, x, xp):
        return self._knl(x, xp)

    @abstractmethod
    def covar(self, Z, x, xp):
        pass

    def sample(self, x, sample_shape=torch.Size([])):
        return MultivariateNormal(self.mean(x), self.knl(x, x)).sample(sample_shape)

    def __add__(self, Y):
        return GaussianProcessAddExpr(self, Y)

    def __mul__(self, α):
        return GaussianProcessMulExpr(self, α)

    def __div__(self, α):
        return GaussianProcessMulExpr(self, 1/α)

    def __matmul__(self, Y):
        if isinstance(self, DeterministicGP):
            return GaussianProcessDetMatmulExpr(self, Y)
        else:
            return GaussianProcessMatmulExpr(self, Y)

    def t(self):
        return GaussianProcessTranspose(self)

class GaussianProcessLeaf(GaussianProcessBase):
    @classmethod
    def isleaf(cls):
        return True

    def name(self):
        return self._name

    def __str__(self):
        return "GaussianProcessLeaf(name={self._name})".format(self=self)

class GaussianProcessExpr(GaussianProcessBase):
    @classmethod
    def isleaf(cls):
        return False

class DeterministicGP(GaussianProcessLeaf):
    def __init__(self, mean, shape, name="{mean}"):
        self._mean = mean
        assert len(shape) <= 2
        assert len(shape) == 1 or min(shape) == 1
        self._shape = shape
        self._name = name.format(mean=mean)

    @property
    def shape(self):
        return self._shape

    def mean(self, x):
        m = self._mean(x)
        return m

    def knl(self, x, xp):
        msize = max(self._shape)
        return x.new_zeros(msize, msize)

    def covar(self, Z, x, xp):
        assert isinstance(Z, GaussianProcessBase)
        if isinstance(Z, DeterministicGP):
            return x.new_zeros(
                max(self._shape),
                max(Z.shape))
        else:
            return Z.covar(self, x, xp).t()

    def sample(self, x, sample_shape=torch.Size([])):
        return self.mean(x).expand(*sample_shape, -1)

    def __repr__(self):
        return "DeterministicGP(mean={self._mean})".format(self=self)

    def __str__(self):
        return "DeterministicGP(name={self._name})".format(self=self)


class GaussianProcessAddExpr(GaussianProcessExpr):
    def __init__(self, X, Y):
        assert isinstance(X, GaussianProcessBase)
        assert isinstance(Y, GaussianProcessBase)
        assert X.shape == Y.shape
        self.lhs = X
        self.rhs = Y

    @property
    def shape(self):
        return self.lhs.shape

    def mean(self, x):
        return self.lhs.mean(x) + self.rhs.mean(x)

    def knl(self, x, xp):
        X, Y = self.lhs, self.rhs
        return X.knl(x, xp) + Y.knl(x, xp) + Y.covar(X, x, xp) + X.covar(Y, x, xp)

    def covar(self, Z, x, xp):
        assert isinstance(Z, GaussianProcessBase)
        return self.lhs.covar(Z, x, xp) + self.rhs.covar(Z, x, xp)


class GaussianProcessMatmulExpr(GaussianProcessExpr):
    def __init__(self, X, Y):
        assert isinstance(X, GaussianProcessBase)
        assert isinstance(Y, GaussianProcessBase)
        assert X.shape[-1] == Y.shape[0]
        self.lhs = X.t()
        self.rhs = Y

    @property
    def shape(self):
        return (1,)

    def mean(self, x):
        X = self.lhs
        Y = self.rhs
        return (X.mean(x).t() @ Y.mean(x)
                + 0.5 * X.covar(Y, x, x).trace()
                + 0.5 * Y.covar(X, x, x).trace())

    def knl(self, x, xp):
        X = self.lhs
        Y = self.rhs
        return (2 * X.covar(Y, x, xp).trace()**2
                + Y.mean(x).t() @ X.knl(x, xp) @ Y.mean(xp)
                + X.mean(x).t() @ Y.knl(x, xp) @ X.mean(xp)
                # FIXME: This should be Y.mean(x) @ X.covar(Y, x, xp) @ X.mean(xp).t()
                + 2 * Y.mean(x).t() @ Y.covar(X, x, xp) @ X.mean(xp))

    def covar(self, Z, x, xp):
        X = self.lhs
        Y = self.rhs
        assert isinstance(Z, GaussianProcessBase)
        return X.mean(x).t() @ Y.covar(Z, x, xp)  + Y.mean(x).t() @ X.covar(Z, x, xp)

    def __str__(self):
        return "{self.lhs!s} @ {self.rhs!s}".format(self=self)

class GaussianProcessDetMatmulExpr(GaussianProcessExpr):
    def __init__(self, X, Y):
        assert isinstance(X, DeterministicGP)
        assert isinstance(Y, GaussianProcessBase)
        assert X.t().shape == Y.shape
        self.lhs = X.t()
        self.rhs = Y

    @property
    def shape(self):
        return (1,)

    def mean(self, x):
        X = self.lhs
        Y = self.rhs
        return X.mean(x).t() @ Y.mean(x)

    def knl(self, x, xp):
        X = self.lhs
        Y = self.rhs
        return X.mean(x).t() @ Y.knl(x, xp) @ X.mean(xp)

    def covar(self, Z, x, xp):
        X = self.lhs
        Y = self.rhs
        assert isinstance(Z, GaussianProcessBase)
        return X.mean(x).t() @ Y.covar(Z, x, xp)

    def __str__(self):
        return "{self.lhs!s} @ {self.rhs!s}".format(self=self)

class GaussianProcessMulExpr(GaussianProcessExpr):
    def __init__(self, X, α):
        assert isinstance(X, GaussianProcessBase)
        assert isinstance(α, (float, int, torch.Tensor))
        self.rhs = X
        self.α = α

    @property
    def shape(self):
        return self.rhs.shape

    def mean(self, x):
        return self.α * self.rhs.mean(x)

    def knl(self, x, xp):
        return (self.α ** 2) * self.rhs.knl(x, xp)

    def covar(self, Z, x, xp):
        assert isinstance(Z, GaussianProcessBase)
        return self.α * self.rhs.covar(Z, x, xp)

    def __str__(self):
        return "{self.α!s} * {self.rhs!s}".format(self=self)

class GaussianProcessTranspose(GaussianProcessExpr):
    def __init__(self, gp):
        assert isinstance(gp, GaussianProcessBase)
        assert len(gp.shape) <= 2
        assert len(gp.shape) >= 1
        self.gp = gp

    @property
    def shape(self):
        assert len(self.gp.shape) <= 2
        if len(self.gp.shape) == 2:
            assert self.gp.shape[0] == 1
            return (self.gp.shape[1],)
        else:
            return (1, self.gp.shape[0])

    def mean(self, x):
        return self.gp.mean(x).t()

    def knl(self, x, xp):
        return self.gp.knl(x, xp)

    def covar(self, Y, x, xp):
        assert isinstance(Y, GaussianProcessBase)
        return self.gp.covar(Y, x, xp).t()

    def t(self):
        return self.gp

    def __str__(self):
        return "{self.gp!s}.t()".format(self=self)


class GaussianProcess(GaussianProcessLeaf):
    def __init__(self, mean, knl, shape, assume_independence=False, name="{mean}"):
        self._mean = mean
        self._knl = knl
        self._shape = shape
        self._covars = dict()
        self.register_covar(self, self.knl) # Covariance with self is variance
        self.assume_independence = assume_independence
        self._name = name.format(mean=mean)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        if hasattr(self._mean, '__self__') and hasattr(self._mean.__self__, 'dtype'):
            return self._mean.__self__.dtype

    def to(self, dtype):
        if hasattr(self._mean, '__self__') and hasattr(self._mean.__self__, 'to'):
            self.mean.__self__.to(dtype=dtype)
        if hasattr(self._knl, '__self__') and hasattr(self._knl.__self__, 'to'):
            self._knl.__self__.to(dtype=dtype)

    def mean(self, x):
        return self._mean(x)

    def knl(self, x, xp):
        return self._knl(x, xp)

    def covar(self, Z, x, xp):
        assert isinstance(Z, GaussianProcessBase)
        if isinstance(Z, GaussianProcess):
            if id(Z) in self._covars:
                return self._covars[id(Z)](x, xp)
            else:
                if self.assume_independence:
                    return x.new_zeros(max(self.shape), max(Z.shape))
                else:
                    raise ValueError("No covariance registered among two leaf GaussianProcesses: "
                                     + self + " and " + Z
                                     + ". Throwing error")
        elif isinstance(Z, DeterministicGP):
            return x.new_zeros(max(self.shape), max(Z.shape))
        else:
            return Z.covar(self, x, xp).t()

    def register_covar(self, gp, covar_func):
        assert isinstance(gp, GaussianProcess)
        self._covars[id(gp)] = covar_func
        gp._covars[id(self)] = covar_func

    def __repr__(self):
        return "GaussianProcess(mean={self._mean}, knl={self._knl}, covars={self._covars}, shape={self.shape})".format(self=self)

    def __str__(self):
        return "GaussianProcess(name={self._name})".format(self=self)

EPS = 2e-3

class GradientGP(GaussianProcessExpr):
    """
    return ∇f, Hₓₓ k_f, ∇ covar_fg
    """
    def __init__(self, f, x_shape, grad_check=False, analytical_hessian=True):
        self.gp = f
        self.x_shape = x_shape
        self.grad_check = grad_check
        self.analytical_hessian = analytical_hessian

    @property
    def shape(self):
        return self.x_shape

    @property
    def dtype(self):
        return self.gp.dtype

    def to(self, dtype):
        self.gp.to(dtype)

    def mean(self, x):
        f = self.gp

        if self.grad_check:
            old_dtype = self.dtype
            self.to(torch.float64)
            with variable_required_grad(x):
                torch.autograd.gradcheck(f.mean, x.double())
            self.to(dtype=old_dtype)
        with variable_required_grad(x):
            return torch.autograd.grad(f.mean(x), x)[0]

    def knl(self, x, xp, eigeps=EPS):
        f = self.gp
        if xp is x:
            xp = xp.detach().clone()

        grad_k_func = lambda xs, xt: torch.autograd.grad(
            f.knl(xs, xt), xs, create_graph=True)[0]
        if self.grad_check:
            old_dtype = self.dtype
            self.to(torch.float64)
            f_knl_func = lambda xt: f.knl(xt, xp.double())
            with variable_required_grad(x):
                torch.autograd.gradcheck(f_knl_func, x.double())
            torch.autograd.gradgradcheck(lambda x: f.knl(x, x), x.double())
            with variable_required_grad(x):
                with variable_required_grad(xp):
                    torch.autograd.gradcheck(
                        lambda xp: grad_k_func(x.double(), xp)[0], xp.double())
            self.to(dtype=old_dtype)

        analytical = self.analytical_hessian
        if analytical:
            Hxx_k = t_hessian(f.knl, x, xp)
        else:
            with variable_required_grad(x):
                with variable_required_grad(xp):
                    old_dtype = self.dtype
                    self.to(torch.float64)
                    Hxx_k = tgradcheck.get_numerical_jacobian(
                        partial(grad_k_func, x.double()), xp.double())
                    self.to(dtype=old_dtype)
                    Hxx_k = Hxx_k.to(old_dtype)
        if torch.allclose(x, xp):
            eigenvalues, eigenvectors = torch.eig(Hxx_k, eigenvectors=False)
            assert (eigenvalues[:, 0] > -eigeps).all(),  " Hessian must be positive definite"
            small_neg_eig = ((eigenvalues[:, 0] > -eigeps) & (eigenvalues[:, 0] < 0))
            if small_neg_eig.any():
                eigenvalues, eigenvectors = torch.eig(Hxx_k, eigenvectors=True)
                evalz = eigenvalues[:, 0]
                evalz[small_neg_eig] = 0
                Hxx_k = eigenvectors.T @ torch.diag(evalz) @ eigenvectors
        return Hxx_k

    def covar(self, G, x, xp):
        """
        returns covar(∇f, g) given covar(f, g)
        """
        f = self.gp
        with variable_required_grad(x):
            J_covar_fg = t_jac(self.gp.covar(G, x, xp), x)
        return J_covar_fg.t()

    def __str__(self):
        return "∇ {self.gp!s}".format(self=self)


# stable pendulum
import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

import warnings
import sys
import io
import tempfile
import inspect
from collections import namedtuple
from functools import partial, wraps
import pickle
import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc as mplibrc
mplibrc('text', usetex=True)

from bayes_cbf.control_affine_model import ControlAffineRegressor, LOG as CALOG
CALOG.setLevel(logging.WARNING)

from bayes_cbf.plotting import plot_results, plot_learned_2D_func, plt_savefig_with_data
from bayes_cbf.sampling import sample_generator_trajectory, controller_sine


def t_vstack(xs):
    torch.cat(xs, dim=-2)


def control_trivial(xi, m=1, mass=None, length=None, gravity=None):
    assert mass is not None
    assert length is not None
    assert gravity is not None
    theta, w = xi
    u = mass * gravity * torch.sin(theta)
    return torch.tensor([u])


def control_random(xi, m=1, mass=None, length=None, gravity=None):
    assert mass is not None
    assert length is not None
    assert gravity is not None
    return control_trivial(
        xi, mass=mass, length=length, gravity=gravity
    ) * torch.abs(torch.rand(1)) + torch.rand(1)


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



class Rel1PendulumModel(DynamicsModel):
    def __init__(self, m, n, mass=1, gravity=10, length=1, deterministic=True,
                 model_noise=0):
        self.m = m
        self.n = n
        self.mass = mass
        self.gravity = gravity
        self.length = length
        self.model_noise = model_noise

    @property
    def ctrl_size(self):
        return self.m

    @property
    def state_size(self):
        return self.n

    def f_func(self, X):
        m = self.m
        n = self.n
        mass = self.mass
        gravity = self.gravity
        length = self.length
        X = torch.tensor(X)
        theta_old, omega_old = X[..., 0:1], X[..., 1:2]
        noise = torch.random.normal(scale=self.model_noise) if self.model_noise else 0
        return noise + torch.cat(
            [omega_old,
             - (gravity/length)*torch.sin(theta_old)], axis=-1)

    def g_func(self, x):
        m = self.m
        n = self.n
        mass = self.mass
        gravity = self.gravity
        length = self.length
        size = x.shape[0] if x.ndim == 2 else 1
        noise = torch.random.normal(scale=self.model_noise) if self.model_noise else 0
        return noise + torch.repeat_interleave(
            torch.tensor([[[1/(mass*length), 0], [1/(mass*length), 0]]]), size, axis=0)



class MeanPendulumDynamicsModel(DynamicsModel):
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
        return torch.zeros((self.n,))

    def g_func(self, X):
        return torch.zeros((self.n, self.m))



class PendulumDynamicsModel(DynamicsModel):
    def __init__(self, m, n, mass=1, gravity=10, length=1, deterministic=True,
                 model_noise=0):
        self.m = m
        self.n = n
        self.mass = mass
        self.gravity = gravity
        self.length = length
        self.model_noise = model_noise

    @property
    def ctrl_size(self):
        return self.m

    @property
    def state_size(self):
        return self.n

    def f_func(self, X):
        m = self.m
        n = self.n
        mass = self.mass
        gravity = self.gravity
        length = self.length
        theta_old, omega_old = X[..., 0:1], X[..., 1:2]
        noise = torch.random.normal(scale=self.model_noise) if self.model_noise else 0
        return noise + torch.cat(
            [omega_old,
             - (gravity/length)*torch.sin(theta_old)], axis=-1)

    def g_func(self, x):
        m = self.m
        n = self.n
        mass = self.mass
        gravity = self.gravity
        length = self.length
        size = x.shape[0] if x.ndim == 2 else 1
        noise = torch.random.normal(scale=self.model_noise) if self.model_noise else 0
        return noise + torch.repeat_interleave(
            torch.tensor([[[0], [1/(mass*length)]]]), size, axis=0)


def sampling_pendulum(dynamics_model, numSteps,
                      x0=None,
                      dt=0.01,
                      controller=control_trivial,
                      plot_every_n_steps=100,
                      axs=None):
    m, g, l = (dynamics_model.mass, dynamics_model.gravity,
               dynamics_model.length)
    tau = dt
    f_func, g_func = dynamics_model.f_func, dynamics_model.g_func
    theta0, omega0 = x0

    # initialize vectors
    time_vec = torch.zeros(numSteps)
    theta_vec = torch.zeros(numSteps)
    omega_vec = torch.zeros(numSteps)
    u_vec = torch.zeros(numSteps)
    #damage indicator
    damage_vec = torch.zeros(numSteps)

    # set initial conditions

    theta = theta0
    omega = omega0
    time = 0

    # begin time-stepping

    for i in range(numSteps):
        time_vec[i] = tau*i
        theta_vec[i] = theta
        omega_vec[i] = omega
        u= controller(torch.tensor((theta, omega)))
        u_vec[i] = u

        if 0<theta_vec[i]<math.pi/4:
            damage_vec[i]=1

        omega_old = omega
        theta_old = theta
        # update the values
        omega_direct = omega_old - (g/l)*torch.sin(theta_old)*tau+(u[0]/(m*l))*tau
        theta_direct = theta_old + omega_old * tau
        # Update as model
        Xold = torch.tensor([[theta_old, omega_old]])
        Xdot = f_func(Xold) + g_func(Xold) @ u
        theta_prop, omega_prop = ( Xold + Xdot * tau ).flatten()
        #assert torch.allclose(omega_direct, omega_prop, atol=1e-6, rtol=1e-4)
        #assert torch.allclose(theta_direct, theta_prop, atol=1e-6, rtol=1e-4)
        LOG.debug("Diff: {}".format(torch.abs(theta_direct - theta_prop)))
        theta, omega = theta_prop, omega_prop

        # theta, omega = theta_direct, omega_direct
        # record the values
        #record and normalize theta to be in -pi to pi range
        theta = (((theta+math.pi) % (2*math.pi)) - math.pi)
        if i % plot_every_n_steps == 0:
            axs = plot_results(np.arange(i+1),
                               omega_vec[:i+1].detach().cpu().numpy(),
                               theta_vec[:i+1].detach().cpu().numpy(),
                               u_vec[:i+1].detach().cpu().numpy(),
                               axs=axs)
            plt.pause(0.001)

    assert torch.all((theta_vec <= math.pi) & (-math.pi <= theta_vec))
    damge_perc=damage_vec.sum() * 100/numSteps
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)


def sampling_pendulum_data(dynamics_model, D=100, dt=0.01, **kwargs):
    tau = dt
    (damge_perc,time_vec,theta_vec,omega_vec,u_vec) = sampling_pendulum(
        dynamics_model, numSteps=D+1, dt=tau, **kwargs)

    # X.shape = Nx2
    X = t_vstack((theta_vec, omega_vec)).T
    # XU.shape = Nx3
    U = u_vec.reshape(-1, 1)
    XU = torch.hstack((X, u_vec.reshape(-1, 1)))

    # compute discrete derivative
    # dxₜ₊₁ = xₜ₊₁ - xₜ / dt
    dX = (X[1:, :] - X[:-1, :]) / tau

    assert torch.all((X[:, 0] <= math.pi) & (-math.pi <= X[:, 0]))
    return dX, X, U


def rad2deg(rad):
    return rad / math.pi * 180


def run_pendulum_experiment(#parameters
        theta0=5*math.pi/6,
        omega0=-0.01,
        tau=0.01,
        mass=1,
        gravity=10,
        length=1,
        numSteps=10000,
        ground_truth_model=True,
        controller=control_trivial,
        pendulum_dynamics_class=PendulumDynamicsModel,
        plotfile='plots/run_pendulum_experiment{suffix}.pdf'):
    if ground_truth_model:
        controller = partial(controller, mass=mass, gravity=gravity, length=length)
    damge_perc,time_vec,theta_vec,omega_vec,u_vec = sampling_pendulum(
        pendulum_dynamics_class(m=1, n=2, mass=mass, gravity=gravity,
                              length=length),
        numSteps, x0=(theta0,omega0), controller=controller, dt=tau)
    plot_results(time_vec, omega_vec, theta_vec, u_vec)

    for i in plt.get_fignums():
        suffix='_%d' % i if i > 0 else ''
        plt_savefig_with_data(plt.figure(i), plotfile.format(suffix=suffix))
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)


def learn_dynamics(
        theta0=5*math.pi/6,
        omega0=-0.01,
        tau=0.01,
        mass=1,
        gravity=10,
        length=1,
        max_train=200,
        numSteps=1000,
        pendulum_dynamics_class=PendulumDynamicsModel):
    #from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    #from bayes_cbf.affine_kernel import AffineScaleKernel
    #from sklearn.gaussian_process import GaussiatorchrocessRegressor

    # kernel_x = 1.0 * RBF(length_scale=torch.tensor([100.0, 100.0]),
    #                      length_scale_bounds=(1e-2, 1e3)) \
    #     + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    # kernel_xu = AffineScaleKernel(kernel_x, 2)

    # xₜ₊₁ = F(xₜ)[1 u]
    # where F(xₜ) = [f(xₜ), g(xₜ)]

    pend_env = pendulum_dynamics_class(m=1, n=2, mass=mass, gravity=gravity,
                                       length=length)
    dX, X, U = sampling_pendulum_data(
        pend_env, D=numSteps, x0=(theta0,omega0),
        dt=tau,
        controller=partial(control_random, mass=mass, gravity=gravity,
                           length=length))

    UH = torch.hstack((torch.ones((U.shape[0], 1), dtype=U.dtype), U))

    # Do not need the full dataset . Take a small subset
    N = min(numSteps-1, max_train)
    shuffled_range = torch.randint(numSteps - 1, size=N)
    XdotTrain = dX[shuffled_range, :]
    Xtrain = X[shuffled_range, :]
    Utrain = U[shuffled_range, :]
    #gp = GaussiatorchrocessRegressor(kernel=kernel_xu,
    #                              alpha=1e6).fit(Z_shuffled, Y_shuffled)
    dgp = ControlAffineRegressor(Xtrain.shape[-1], Utrain.shape[-1])
    dgp.fit(Xtrain, Utrain, XdotTrain, training_iter=50, lr=0.01)
    dgp.save()

    # Plot the pendulum trajectory
    plot_results(torch.arange(U.shape[0]), omega_vec=X[:, 0],
                 theta_vec=X[:, 1], u_vec=U[:, 0])
    fig = plot_learned_2D_func(Xtrain.detach().cpu().numpy(), dgp.f_func, pend_env.f_func,
                               axtitle="f(x)[{i}]")
    plt_savefig_with_data(fig, 'plots/f_learned_vs_f_true.pdf')
    fig = plot_learned_2D_func(Xtrain.detach().cpu().numpy(), dgp.g_func, pend_env.g_func,
                               axtitle="g(x)[{i}]")
    plt_savefig_with_data(fig, 'plots/g_learned_vs_g_true.pdf')

    # within train set
    FX_98, FXcov_98 = dgp.predict(X[98:99,:], return_cov=True)
    dX_98 = FX_98[0, ...].T @ UH[98, :]
    #dXcov_98 = UH[98, :] @ FXcov_98 @ UH[98, :]
    if not torch.allclose(dX[98], dX_98, rtol=0.05, atol=0.05):
        print("Train sample: expected:{}, got:{}, cov:{}".format(dX[98], dX_98, FXcov_98))

    # out of train set
    FXNp1, FXNp1cov = dgp.predict(X[N+1:N+2,:], return_cov=True)
    dX_Np1 = FXNp1[0, ...].T @ UH[N+1, :]
    if not torch.allclose(dX[N+1], dX_Np1, rtol=0.05, atol=0.05):
        print("Test sample: expected:{}, got:{}, cov:{}".format( dX[N+1], dX_Np1, FXNp1cov))

    return dgp, dX, U


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, solver=None,
                    initvals=None, **kwargs):
    import cvxopt
    from cvxopt import matrix
    #P = (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
    else:
        args.extend([None, None])
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    else:
        args.extend([None, None])
    args.extend([initvals, solver])
    solvers = cvxopt.solvers
    old_options = solvers.options.copy()
    solvers.options.update(kwargs)
    try:
        sol = cvxopt.solvers.qp(*args)
    except ValueError:
        return None
    finally:
        solvers.options.update(old_options)
    if 'optimal' not in sol['status']:
        return None
    return np.asarray(sol['x']).reshape((P.shape[1],))



class NamedAffineFunc(ABC):
    @property
    def __name__(self):
        """
        Name used for plots
        """
        return self.name

    @abstractmethod
    def value(self, x):
        """
        Scalar value function
        """

    @abstractmethod
    def b(self, x):
        """
        A(x) @ u - b(x)
        """

    @abstractmethod
    def A(self, x):
        """
        A(x) @ u - b(x)
        """

    def __call__(self, x, u):
        """
        A(x) @ u - b(x)
        """
        return self.A(x) @ u - self.b(x)


class NamedFunc:
    def __init__(self, func, name):
        self.__name__ = name
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


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

class EnergyCLF(NamedAffineFunc):
    @store_args
    def __init__(self, model,
                 clf_c=1,
                 name="clf"):
        self.model = model

    def V_clf(self, x, pkg=torch):
        (θ, ω) = x
        g, l = self.gravity, self.length
        return l*ω**2 / 2 + g*(1-pkg.cos(θ))

    value = V_clf

    def __call__ (self, x, u):
        return self.A(x) @ u - self.b(x)

    def grad_V_clf(self, x):
        (theta, w) = x
        g, l = self.gravity, self.length
        numpy_val = torch.tensor([ g*torch.sin(theta), l * w])
        x_with_grad = x.clone().detach().requires_grad_(True)
        x_with_grad.requires_grad = True
        V = self.V_clf(x_with_grad, pkg=torch)
        V.backward()
        torch_val = x_with_grad.grad
        assert torch.allclose(numpy_val, torch_val, rtol=1e-3, atol=1e-3)
        return torch_val

    def A(self, x):
        (θ, ω) = x
        abstract = self.grad_V_clf(x) @ self.g_func(x)
        if isinstance(self.model, PendulumDynamicsModel):
            m, l, g = self.mass, self.length, self.gravity
            direct = torch.tensor([l*ω])
            assert torch.allclose(direct , abstract, rtol=1e-2, atol=1e-4)
        return abstract

    def b(self, x):
        (θ, ω) = x
        c = self.clf_c
        abstract = - self.grad_V_clf(x) @ self.f_func(x) - c * self.V_clf(x)
        if isinstance(self.model, PendulumDynamicsModel):
            m, l, g = self.mass, self.length, self.gravity
            direct = torch.tensor([-c*(l*ω**2 / 2 + g*(1-torch.cos(θ)))])
            assert torch.allclose(direct , abstract, rtol=1e-2, atol=1e-4)
        return abstract

    def __getattr__(self, name):
        return getattr(self.model, name)


class RadialCBF(NamedAffineFunc):
    @store_args
    def __init__(self, model,
                 cbf_col_gamma=1,
                 cbf_col_K_alpha=[1., 1.],
                 cbf_col_delta=math.pi/8,
                 cbf_col_theta=math.pi/4,
                 theta_c=math.pi/4,
                 gamma_col=1,
                 delta_col=math.pi/8,
                 name="cbf"):
        self.model = model

    def h_col(self, x, pkg=torch):
        (theta, w) = x
        delta_col = self.cbf_col_delta
        theta_c = self.cbf_col_theta
        return (math.cos(delta_col) - pkg.cos(theta - theta_c))*(w**2+1)

    value = h_col

    def __call__ (self, x, u):
        return self.A(x) @ u - self.b(x)

    def grad_h_col(self, x):
        (θ, ω) = x
        θ_c = self.cbf_col_theta
        Δ_c = self.cbf_col_delta
        return torch.tensor([torch.sin(θ - θ_c)*(ω**2+1),
                         2*ω*(torch.cos(Δ_c) - torch.cos(θ - θ_c))])

    def A(self, x):
        (θ, ω) = x
        Δ_c = self.cbf_col_delta
        θ_c = self.cbf_col_theta
        abstract = -self.grad_h_col(x) @ self.g_func(x)
        if isinstance(self.model, PendulumDynamicsModel):
            m, l, g = self.mass, self.length, self.gravity
            direct = torch.tensor([-(2*ω*(torch.cos(Δ_c)-torch.cos(θ-θ_c)))/(m*l)])
            assert torch.allclose(direct, abstract, rtol=1e-2, atol=1e-4)
        return abstract

    def b(self, x):
        (θ, ω) = x
        γ_c = self.cbf_col_gamma
        Δ_c = self.cbf_col_delta
        θ_c= self.cbf_col_theta
        abstract = self.grad_h_col(x) @ self.f_func(x) + γ_c * self.h_col(x)
        if isinstance(self.model, PendulumDynamicsModel):
            m, l, g = self.mass, self.length, self.gravity
            b = (γ_c*(torch.cos(Δ_c)-torch.cos(θ-θ_c))*(ω**2+1)
                + (ω**3+ω)*torch.sin(θ-θ_c)
                - (2*g*ω*torch.sin(θ)*(torch.cos(Δ_c)-torch.cos(θ-θ_c)))/l)
            direct = torch.tensor([b])
            assert torch.allclose(direct, abstract, rtol=1e-2, atol=1e-4)
        return abstract

    def __getattr__(self, name):
        return getattr(self.model, name)


class RadialCBFRelDegree2(NamedAffineFunc):
    @store_args
    def __init__(self, model,
                 cbf_col_gamma=1,
                 cbf_col_K_alpha=[1., 1.],
                 cbf_col_delta=math.pi/8,
                 cbf_col_theta=math.pi/4,
                 theta_c=math.pi/4,
                 gamma_col=1,
                 delta_col=math.pi/8,
                 name="cbf-r2"):
        self.model = model

    def h2_col(self, x):
        (theta, w) = x
        delta_col = self.cbf_col_delta
        theta_c = self.cbf_col_theta
        return math.cos(delta_col) - torch.cos(theta - theta_c)

    value = h2_col

    def __call__ (self, x, u):
        return self.A(x) @ u - self.b(x)

    def grad_h2_col(self, x):
        (θ, ω) = x
        θ_c = self.cbf_col_theta
        return torch.tensor([torch.sin(θ - θ_c), 0])

    def lie_f_h2_col(self, x):
        return self.grad_h2_col(x) @ self.f_func(x)

    def lie2_f_h_col(self, x):
        (θ, ω) = x
        m, l, g = self.mass, self.length, self.gravity
        Δ_c = self.cbf_col_delta
        θ_c = self.cbf_col_theta
        return - ω**2 * torch.cos(θ - θ_c) - (g / l) * torch.sin(θ - θ_c) * torch.sin(θ)

    def lie_g_lie_f_h_col(self, x):
        (θ, ω) = x
        m, l, g = self.mass, self.length, self.gravity
        Δ_c = self.cbf_col_delta
        θ_c = self.cbf_col_theta
        return (1/(m*l)) * torch.sin(θ - θ_c)

    def A(self, x):
        return torch.tensor([- self.lie_g_lie_f_h_col(x)])

    def b(self, x):
        K_α = torch.tensor(self.cbf_col_K_alpha)
        η_b_x = torch.tensor([self.h2_col(x), self.lie_f_h2_col(x)])
        return torch.tensor([self.lie2_f_h_col(x) + K_α @ η_b_x])

    def __getattr__(self, name):
        return getattr(self.model, name)


class PendulumCBFCLFDirect:
    @store_args
    def __init__(self, mass=None, length=None, gravity=None,
                 axes=None,
                 constraint_hists=[],
                 pendulum_dynamics_class=PendulumDynamicsModel,
    ):
        self.set_model_params(mass=mass, length=length, gravity=gravity)

    def set_model_params(self, **kwargs):
        self.model = self.pendulum_dynamics_class(m=1, n=2, **kwargs)
        self.aff_constraints = [
            #NamedAffineFunc(self.A_clf, self.b_clf, "clf"),
            EnergyCLF(self.model),
            RadialCBFRelDegree2(self.model),
            #NamedAffineFunc(self.A_col, self.b_col, "col")
            RadialCBFRelDegree2(self.model),
        ]

    def f_func(self, x):
        return self.model.f_func(torch.tensor(x))

    def g_func(self, x):
        return self.model.g_func(torch.tensor(x))

    def _A_sr(self, x):
        # UNUSED
        warnings.warn_once("DEPRECATED")
        (theta, w) = x
        m, l, g = self.model.mass, self.model.length, self.model.gravity
        return torch.tensor([l*w])

    def _b_sr(self, x):
        # UNUSED
        warnings.warn_once("DEPRECATED")
        (theta, w) = x
        m, l, g = self.model.mass, self.model.length, self.model.gravity
        cbf_sr_gamma=1
        cbf_sr_delta=10
        gamma_sr = cbf_sr_gamma
        delta_sr = cbf_sr_delta
        return torch.tensor([gamma_sr*(delta_sr-w**2)+(2*g*torch.sin(theta)*w)/(l)])

    def plot_constraints(self, funcs, x, u):
        axs = self.axes
        nfuncs = len(funcs)
        if axs is None:
            nplots = nfuncs
            shape = ((math.ceil(nplots / 2), 2) if nplots >= 2 else (nplots,))
            fig, axs = plt.subplots(*shape)
            fig.subplots_adjust(wspace=0.35, hspace=0.5)
            self.axes = axs.flatten() if hasattr(axs, "flatten") else torch.tensor([axs])

        if len(self.constraint_hists) < nfuncs:
            self.constraint_hists = self.constraint_hists + [
                list() for _ in range(
                nfuncs - len(self.constraint_hists))]

        for i, af in enumerate(funcs):
            self.constraint_hists[i].append(af(x, u))

        if torch.rand(1) < 1e-2:
            for i, (ch, af) in enumerate(zip(self.constraint_hists, funcs)):
                axs[i].clear()
                axs[i].plot(ch)
                axs[i].set_ylabel(af.__name__)
                axs[i].set_xlabel("time")
                plt.pause(0.0001)


    def control(self, xi, mass=None, gravity=None, length=None):
        assert mass is not None
        assert length is not None
        assert gravity is not None
        self.set_model_params(mass=mass, gravity=gravity, length=length)

        u = control_QP_cbf_clf(
            xi,
            ctrl_aff_constraints=self.aff_constraints,
            constraint_margin_weights=[100])
        self.plot_constraints(
            self.aff_constraints + [
                NamedFunc(lambda x, u: f.value(x), r"\verb!%s!" % f.__name__)
                for f in self.aff_constraints],
            xi, u)
        return u


class ControlCBFCLFLearned(PendulumCBFCLFDirect):
    @store_args
    def __init__(self,
                 x_dim=2,
                 u_dim=1,
                 gamma_sr=1,
                 delta_sr=10,
                 train_every_n_steps=50,
                 mean_dynamics_model_class=MeanPendulumDynamicsModel
    ):
        self.Xtrain = []
        self.Utrain = []
        self.dgp = ControlAffineRegressor(x_dim, u_dim)
        self.mean_dynamics_model = mean_dynamics_model_class(m=u_dim, n=x_dim)
        self.ctrl_aff_constraints=[EnergyCLF(self),
                                   RadialCBF(self)]

    def f_g(self, x):
        X = torch.tensor([x])
        FXTmean = self.dgp.predict(X, return_cov=False)
        fx = FXTmean[0, 0, :]
        gx = FXTmean[0, 1:, :].T
        return fx, gx

    def f_func(self, x):
        return self.f_g(x)[0] + self.mean_dynamics_model.f_func(x)

    def g_func(self, x):
        return self.f_g(x)[1] + self.mean_dynamics_model.g_func(x)

    def train(self):
        if not len(self.Xtrain):
            return
        assert len(self.Xtrain) == len(self.Utrain), "Call train when Xtrain and Utrain are balanced"
        Xtrain = torch.tensor(self.Xtrain)
        Utrain = torch.tensor(self.Utrain)
        XdotTrain = Xtrain[1:, :] - Xtrain[:-1, :]
        XdotMean = self.mean_dynamics_model.f_func(Xtrain) + (self.mean_dynamics_model.g_func(Xtrain) @ Utrain.T).T
        XdotError = XdotTrain - XdotMean[1:, :]
        axs = plot_results(torch.arange(Utrain.shape[0]), omega_vec=Xtrain[:, 0],
                     theta_vec=Xtrain[:, 1], u_vec=Utrain[:, 0])
        plt_savefig_with_data(
            axs[0,0].figure,
            'plots/pendulum_data_{}.pdf'.format(Xtrain.shape[0]))
        assert torch.all((Xtrain[:, 0] <= math.pi) & (-math.pi <= Xtrain[:, 0]))
        LOG.info("Training model with datasize {}".format(XdotTrain.shape[0]))
        try:
            self.dgp.fit(Xtrain[:-1, :], Utrain[:-1, :], XdotTrain)
        except AssertionError:
            train_data = (Xtrain[:-1, :], Utrain[:-1, :], XdotTrain)
            filename = hashlib.sha224(pickle.dumps(train_data)).hexdigest()
            filepath = 'tests/data/{}.pickle'.format(filename)
            pickle.dump(train_data, open(filepath, 'wb'))
            raise

    def control(self, xi):
        if len(self.Xtrain) % self.train_every_n_steps == 0:
            # train every n steps
            LOG.info("Training GP with dataset size {}".format(len(self.Xtrain)))
            self.train()

        assert torch.all((xi[0] <= math.pi) & (-math.pi <= xi[0]))
        self.Xtrain.append(xi)
        u = control_QP_cbf_clf(xi,
                               ctrl_aff_constraints=self.ctrl_aff_constraints,
                               constraint_margin_weights=[1, 10000])
        self.Utrain.append(u)
        return u


def control_QP_cbf_clf(x,
                       ctrl_aff_constraints,
                       constraint_margin_weights=[]):
    """
    Args:
          A_cbfs: A tuple of CBF functions
          b_cbfs: A tuple of CBF functions
          constraint_margin_weights: Add a margin constant to the constraint
                                     that is maximized.

    """
    #import ipdb; ipdb.set_trace()
    clf_idx = 0
    A_total = np.vstack([af.A(x).detach().numpy()
                         for af in ctrl_aff_constraints])
    b_total = np.vstack([af.b(x).detach().numpy()
                         for af in ctrl_aff_constraints]).flatten()
    D_u = A_total.shape[1]
    N_const = A_total.shape[0]

    # u0 = l*g*sin(theta)
    # uopt = 0.1*g
    # contraints = A_total.dot(uopt) - b_total
    # assert contraints[0] <= 0
    # assert contraints[1] <= 0
    # assert contraints[2] <= 0


    # [A, I][ u ]
    #       [ ρ ] ≤ b for all constraints
    #
    # minimize
    #         [ u, ρ1, ρ2 ] [ 1,     0] [  u ]
    #                       [ 0,   100] [ ρ2 ]
    #         [A_cbf, 1] [ u, -ρ ] ≤ b_cbf
    #         [A_clf, 1] [ u, -ρ ] ≤ b_clf
    N_slack = len(constraint_margin_weights)
    A_total_rho = np.hstack(
        (A_total,
         np.vstack((-np.eye(N_slack),
                    np.zeros((N_const - N_slack, N_slack))))
        ))
    A = A_total
    P_rho = np.eye(D_u + N_slack)
    P_rho[D_u:, D_u:] = np.diag(constraint_margin_weights)
    q_rho = np.zeros(P_rho.shape[0])
    #u_rho_init = np.linalg.lstsq(A_total_rho, b_total - 1e-1, rcond=-1)[0]
    u_rho = cvxopt_solve_qp(P_rho.astype(np.float64),
                            q_rho.astype(np.float64),
                            G=A_total_rho.astype(np.float64),
                            h=b_total.astype(np.float64),
                            show_progress=False,
                            maxiters=1000)
    if u_rho is None:
        raise RuntimeError("""QP is infeasible
        minimize
        u_rhoᵀ {P_rho} u_rho
        s.t.
        {A_total_rho} u_rho ≤ {b_total}""".format(
            P_rho=P_rho,
            A_total_rho=A_total_rho, b_total=b_total))
    # Constraints should be satisfied
    constraint = A_total_rho @ u_rho - b_total
    assert np.all((constraint <= 1e-2) | (constraint / np.abs(b_total) <= 1e-2))
    return torch.from_numpy(u_rho[:D_u]).to(dtype=x.dtype)


run_pendulum_control_trival = partial(
    run_pendulum_experiment, controller=control_trivial,
    plotfile='plots/run_pendulum_control_trival{suffix}.pdf')
"""
Run pendulum with a trivial controller.
"""


run_pendulum_control_cbf_clf = partial(
    run_pendulum_experiment, controller=PendulumCBFCLFDirect().control,
    plotfile='plots/run_pendulum_control_cbf_clf{suffix}.pdf',
    theta0=5*math.pi/12,
    tau=0.001,
    numSteps=15000)
"""
Run pendulum with a safe CLF-CBF controller.
"""


def run_pendulum_control_online_learning(numSteps=1000):
    """
    Run save pendulum control while learning the parameters online
    """
    return run_pendulum_experiment(
        ground_truth_model=False,
        plotfile='plots/run_pendulum_control_online_learning{suffix}.pdf',
        controller=ControlCBFCLFLearned().control,
        numSteps=numSteps)


if __name__ == '__main__':
    #run_pendulum_control_trival()
    run_pendulum_control_cbf_clf()
    #learn_dynamics()
    #run_pendulum_control_online_learning()

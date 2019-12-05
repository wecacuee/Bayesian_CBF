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
from bayes_cbf.controllers import cvxopt_solve_qp, control_QP_cbf_clf, controller_qcqp_gurobi
from bayes_cbf.misc import t_vstack, to_numpy
from bayes_cbf.relative_degree_2 import cbf2_quadratic_constraint


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
        u= controller(torch.tensor((theta, omega)), i=i)
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
        numSteps, x0=torch.tensor([theta0,omega0]), controller=controller, dt=tau)
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
        pend_env, D=numSteps, x0=torch.tensor([theta0,omega0]),
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
                 theta_goal=0.,
                 omega_goal=0.,
                 quad_goal_cost=[[1.0, 0],
                                 [0, 1.0]],
                 x_dim=2,
                 u_dim=1,
                 gamma_sr=1,
                 delta_sr=10,
                 train_every_n_steps=50,
                 mean_dynamics_model_class=MeanPendulumDynamicsModel,
                 egreedy_scheme=[1, 0.01],
                 iterations=1000,
                 max_unsafe_prob=0.01,
                 dt=0.001,
                 max_train=250,
    ):
        self.Xtrain = []
        self.Utrain = []
        self.model = ControlAffineRegressor(x_dim, u_dim)
        self.mean_dynamics_model = mean_dynamics_model_class(m=u_dim, n=x_dim)
        self.ctrl_aff_constraints=[EnergyCLF(self),
                                   RadialCBF(self)]
        self.cbf2 = RadialCBFRelDegree2(self.model)
        self._has_been_trained_once = False
        # These are used in the optimizer hence numpy
        self.x_goal = torch.tensor([theta_goal, omega_goal])
        self.x_quad_goal_cost = torch.tensor(quad_goal_cost)

    def train(self):
        if not len(self.Xtrain):
            return
        assert len(self.Xtrain) == len(self.Utrain), "Call train when Xtrain and Utrain are balanced"
        Xtrain = torch.cat(self.Xtrain).reshape(-1, self.x_dim)
        Utrain = torch.cat(self.Utrain).reshape(-1, self.u_dim)
        XdotTrain = Xtrain[1:, :] - Xtrain[:-1, :]
        XdotMean = self.mean_dynamics_model.f_func(Xtrain) + (self.mean_dynamics_model.g_func(Xtrain) @ Utrain.T).T
        XdotError = XdotTrain - XdotMean[1:, :]
        axs = plot_results(np.arange(Utrain.shape[0]),
                           omega_vec=to_numpy(Xtrain[:, 0]),
                           theta_vec=to_numpy(Xtrain[:, 1]),
                           u_vec=to_numpy(Utrain[:, 0]))
        plt_savefig_with_data(
            axs[0,0].figure,
            'plots/pendulum_data_{}.pdf'.format(Xtrain.shape[0]))
        assert torch.all((Xtrain[:, 0] <= math.pi) & (-math.pi <= Xtrain[:, 0]))
        LOG.info("Training model with datasize {}".format(XdotTrain.shape[0]))
        if XdotTrain.shape[0] > self.max_train:
            indices = torch.randint(XdotTrain.shape[0], self.max_train)
            self.model.fit(Xtrain[indices, :], Utrain[indices, :],
                           XdotTrain[indices, :])
        else:
            self.model.fit(Xtrain[:-1, :], Utrain[:-1, :], XdotTrain)

        self._has_been_trained_once = True

    def egreedy(self, i):
        se, ee = map(math.log, self.egreedy_scheme)
        T = self.iterations
        return math.exp( i * (ee - se) / T )

    def quad_objective(self, i, x, u0):
        x_g = self.x_goal
        P = self.x_quad_goal_cost
        R = torch.eye(self.u_dim)
        λ = self.egreedy(i)
        fx = (x + self.dt * self.model.f_func(x)
              + self.dt * self.mean_dynamics_model.f_func(x))
        Gx = (self.dt * self.model.g_func(x.unsqueeze(0)).squeeze(0)
              + self.dt * self.mean_dynamics_model.g_func(x))
        # xp = fx + Gx @ u
        # (1-λ)(xp - x_g)ᵀ P (xp - x_g) + λ (u - u₀)ᵀ R (u - u₀)
        # Quadratic term: uᵀ (λ R + (1-λ)GₓᵀPGₓ) u
        # Linear term   : - (2λRu₀ + 2(1-λ)GₓP(x_g - fx)  )ᵀ u
        # Constant term : + (1-λ)(x_g-fx)ᵀP(x_g-fx) + λ u₀ᵀRu₀


        # Quadratic term λ R + (1-λ)Gₓᵀ P Gₓ
        Q = λ * R + (1-λ) * Gx.T @ P @ Gx
        # Linear term - (2λRu₀ + 2(1-λ)Gₓ P(x_g - fx)  )ᵀ u
        c = -2.0*(λ * R @ u0 + (1-λ) * Gx.T @ P @ (x_g - fx))
        # Constant term + (1-λ)(x_g-fx)ᵀ P (x_g-fx) + λ u₀ᵀRu₀
        const = (1-λ) * (x_g-fx).T @ P @ (x_g-fx) + λ * u0.T @ R @ u0
        return list(map(to_numpy, (Q, c, const)))


    def quadtratic_contraints(self, i, x, u0):
        A, b, c = cbf2_quadratic_constraint(self.cbf2.h2_col,
                                            self.model.fu_func_gp, x, u0,
                                            self.max_unsafe_prob)
        return [list(map(to_numpy, (A, b, c)))]


    def control(self, xi, i=None):
        if len(self.Xtrain) % self.train_every_n_steps == 0:
            # train every n steps
            LOG.info("Training GP with dataset size {}".format(len(self.Xtrain)))
            self.train()

        assert torch.all((xi[0] <= math.pi) & (-math.pi <= xi[0]))

        if self._has_been_trained_once:
            u0 = torch.rand(self.u_dim, dtype=xi.dtype, device=xi.device)
            u = controller_qcqp_gurobi(to_numpy(u0),
                                       self.quad_objective(i, xi, u0),
                                       self.quadtratic_contraints(i, xi, u0))
            u = torch.from_numpy(u).to(dtype=xi.dtype, device=xi.device)
        else:
            u = torch.rand(self.u_dim)
        # record the xi, ui pair
        self.Xtrain.append(xi)
        self.Utrain.append(u)
        return u


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


def run_pendulum_control_online_learning(numSteps=15000):
    """
    Run save pendulum control while learning the parameters online
    """
    return run_pendulum_experiment(
        ground_truth_model=False,
        plotfile='plots/run_pendulum_control_online_learning{suffix}.pdf',
        controller=ControlCBFCLFLearned(dt=0.001).control,
        numSteps=numSteps,
        tau=0.001)


if __name__ == '__main__':
    #run_pendulum_control_trival()
    #run_pendulum_control_cbf_clf()
    #learn_dynamics()
    run_pendulum_control_online_learning()

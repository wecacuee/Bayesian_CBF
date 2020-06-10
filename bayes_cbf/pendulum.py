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
from bayes_cbf.controllers import (cvxopt_solve_qp, control_QP_cbf_clf,
                                   controller_qcqp_gurobi, Controller,
                                   ControlCBFLearned, NamedAffineFunc, NamedFunc)
from bayes_cbf.misc import (t_vstack, t_hstack, to_numpy, store_args,
                            DynamicsModel, variable_required_grad)
from bayes_cbf.relative_degree_2 import cbc2_quadratic_terms, cbc2_gp


class ControlTrivial(Controller):
    needs_ground_truth = True
    @store_args
    def __init__(self, m=1, mass=None, length=None, gravity=None, dt=None,
                 true_model=None):
        pass

    def control(self, xi, i=None):
        mass, gravity, length = self.mass, self.gravity, self.length
        theta, w = xi
        u = mass * gravity * torch.sin(theta)
        return torch.tensor([u])


class ControlRandom(Controller):
    needs_ground_truth = True
    @store_args
    def __init__(self, **kwargs):
        self.control_trivial = ControlTrivial(**kwargs)

    def control(self, xi, i=None):
        return self.control_trivial.control(
            xi, i=i
        ) * torch.abs(torch.rand(1)) + torch.rand(1)


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
    ground_truth = True
    def __init__(self, m, n, mass=1, gravity=10, length=1, deterministic=True,
                 model_noise=0,
                 dtype=torch.get_default_dtype()):
        self.m = m
        self.n = n
        self.mass = mass
        self.gravity = gravity
        self.length = length
        self.model_noise = model_noise
        self.dtype = dtype

    def to(self, dtype):
        self.dtype = dtype

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
        noise = torch.random.normal(scale=self.model_noise, dtype=self.dtype) if self.model_noise else 0
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
        noise = torch.random.normal(scale=self.model_noise, dtype=self.dtype) if self.model_noise else 0
        if x.ndim == 2:
            return noise + torch.repeat_interleave(
                torch.tensor([[[0], [1/(mass*length)]]], dtype=self.dtype), size, axis=0)
        else:
            return noise + torch.tensor([[0], [1/(mass*length)]], dtype=self.dtype)


def sampling_pendulum(dynamics_model, numSteps,
                      controller=None,
                      x0=None,
                      dt=0.01,
                      plot_every_n_steps=20,
                      axs=None,
                      plotfile='plots/pendulum_data_{i}.pdf'):
    assert controller is not None, 'Surprise !! Changed interface to make controller a required argument'
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
            plt_savefig_with_data(axs.flatten()[0].figure, plotfile.format(i=i))
            plt.pause(0.001)

    assert torch.all((theta_vec <= math.pi) & (-math.pi <= theta_vec))
    damge_perc=damage_vec.sum() * 100/numSteps
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)


def sampling_pendulum_data(dynamics_model, D=100, dt=0.01, **kwargs):
    tau = dt
    (damge_perc,time_vec,theta_vec,omega_vec,u_vec) = sampling_pendulum(
        dynamics_model, numSteps=D+1, dt=tau, **kwargs)

    # X.shape = Nx2
    X = t_vstack((theta_vec.unsqueeze(0), omega_vec.unsqueeze(0))).T
    # XU.shape = Nx3
    U = u_vec.reshape(-1, 1)
    XU = t_hstack((X, u_vec.reshape(-1, 1)))

    # compute discrete derivative
    # dxₜ₊₁ = xₜ₊₁ - xₜ / dt
    dX = (X[1:, :] - X[:-1, :]) / tau

    assert torch.all((X[:, 0] <= math.pi) & (-math.pi <= X[:, 0]))
    return dX, X, U


def rad2deg(rad):
    return rad * 180. / math.pi

def deg2rad(deg):
    return deg * math.pi / 180.


def run_pendulum_experiment(#parameters
        theta0=5*math.pi/6,
        omega0=-0.01,
        tau=0.01,
        mass=1,
        gravity=10,
        length=1,
        numSteps=10000,
        controller_class=ControlTrivial,
        pendulum_dynamics_class=PendulumDynamicsModel,
        plotfile='plots/run_pendulum_experiment{suffix}.pdf',
        dtype=torch.float32):
    torch.set_default_dtype(dtype)
    pendulum_model = pendulum_dynamics_class(m=1, n=2, mass=mass, gravity=gravity,
                                             length=length, dtype=dtype)
    if controller_class.needs_ground_truth:
        controller = controller_class(mass=mass, gravity=gravity,
                                      length=length, dt=tau,
                                      true_model=pendulum_model,
                                      plotfile=plotfile.format(suffix='_ctrl_{suffix}'),
                                      dtype=dtype
        ).control
    else:
        controller = controller_class(dt=tau, true_model=pendulum_model,
                                      plotfile=plotfile.format(suffix='_ctrl_{suffix}'),
                                      dtype=dtype

        ).control
    damge_perc,time_vec,theta_vec,omega_vec,u_vec = sampling_pendulum(
        pendulum_model,
        numSteps, x0=torch.tensor([theta0,omega0]), controller=controller, dt=tau,
        plotfile=plotfile.format(suffix='_trajectory_{i}'))
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
        controller=ControlRandom(mass=mass, gravity=gravity, length=length).control)

    UH = t_hstack((torch.ones((U.shape[0], 1), dtype=U.dtype), U))

    # Do not need the full dataset . Take a small subset
    N = min(numSteps-1, max_train)
    shuffled_range = torch.randint(numSteps - 1, size=(N,))
    XdotTrain = dX[shuffled_range, :]
    Xtrain = X[shuffled_range, :]
    Utrain = U[shuffled_range, :]
    #gp = GaussiatorchrocessRegressor(kernel=kernel_xu,
    #                              alpha=1e6).fit(Z_shuffled, Y_shuffled)
    dgp = ControlAffineRegressor(Xtrain.shape[-1], Utrain.shape[-1])
    dgp.fit(Xtrain, Utrain, XdotTrain, training_iter=50)
    #dgp.save()

    # Plot the pendulum trajectory
    Xtrain_numpy = Xtrain.detach().cpu().numpy()
    plot_results(torch.arange(U.shape[0]), omega_vec=X[:, 0],
                 theta_vec=X[:, 1], u_vec=U[:, 0])
    axs = plot_learned_2D_func(Xtrain_numpy, dgp.f_func,
                               pend_env.f_func,
                               axtitle="f(x)[{i}]")
    plt_savefig_with_data(axs.flatten()[0].figure,
                          'plots/f_orig_learned_vs_f_true.pdf')
    axs = plot_learned_2D_func(Xtrain_numpy, dgp.f_func_mean,
                               pend_env.f_func,
                               axtitle="f(x)[{i}]")
    plt_savefig_with_data(axs.flatten()[0].figure,
                          'plots/f_custom_learned_vs_f_true.pdf')
    axs = plot_learned_2D_func(Xtrain_numpy,
                               dgp.g_func,
                               pend_env.g_func,
                               axtitle="g(x)[{i}]")
    plt_savefig_with_data(axs.flatten()[0].figure,
                          'plots/g_learned_vs_g_true.pdf')

    # within train set
    dX_98, _ = dgp.predict_flatten(X[98:99,:], U[98:99, :])
    #dX_98 = FX_98[0, ...].T @ UH[98, :]
    #dXcov_98 = UH[98, :] @ FXcov_98 @ UH[98, :]
    if not torch.allclose(dX[98], dX_98, rtol=0.4, atol=0.1):
        print("Test failed: Train sample: expected:{}, got:{}, cov".format(dX[98], dX_98))

    # out of train set
    dX_Np1, _ = dgp.predict_flatten(X[N+1:N+2,:], U[N+1:N+2,:])
    #dX_Np1 = FXNp1[0, ...].T @ UH[N+1, :]
    if not torch.allclose(dX[N+1], dX_Np1, rtol=0.4, atol=0.1):
        print("Test failed: Test sample: expected:{}, got:{}, cov".format( dX[N+1], dX_Np1))

    true_h_func = RadialCBFRelDegree2(pend_env)
    learned_h_func = RadialCBFRelDegree2(dgp)
    def learned_cbc2(X):
        l_cbc2 = torch.zeros(X.shape[0], 2)
        for i in range(X.shape[0]):
            cbc2 = cbc2_gp( learned_h_func.h2_col, dgp, U[N+1, :])
            l_cbc2[i, 0] = cbc2.mean(X[i, :])
        return l_cbc2
    #true_cbc2 = - true_h_func.A(X[N+1, :]) @ U[N+1,:] + true_h_func.b(X[N+1, :])
    def true_cbc2(X):
        t_cbc2 = torch.zeros(X.shape[0], 2)
        for i in range(X.shape[0]):
            t_cbc2[i, 0] = - true_h_func.A(X[i, :]) @ U[N+1,:] + true_h_func.b(X[i, :])
        return t_cbc2
    axs = plot_learned_2D_func(Xtrain_numpy,
                               learned_cbc2,
                               true_cbc2,
                               axtitle="cbc2[{i}]"
    )
    plt_savefig_with_data(axs.flatten()[0].figure,
                          'plots/cbc_learned_vs_cbc_true.pdf')
    return dgp, dX, U


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
                 cbf_col_K_alpha=[1., 3.],
                 cbf_col_delta=math.pi/8,
                 cbf_col_theta=math.pi/4,
                 theta_c=math.pi/4,
                 gamma_col=1,
                 delta_col=math.pi/8,
                 name="cbf-r2",
                 dtype=torch.get_default_dtype()):
        self.model = model

    def to(self, dtype):
        self.dtype = dtype
        self.model.to(dtype=dtype)

    def h2_col(self, x):
        (theta, w) = x
        delta_col = self.cbf_col_delta
        theta_c = self.cbf_col_theta
        return math.cos(delta_col) - torch.cos(theta - theta_c)

    value = h2_col

    def __call__ (self, x, u):
        return self.A(x) @ u - self.b(x)

    def grad_h2_col(self, X_in):
        if X_in.ndim == 1:
            X = X_in.unsqueeze(0)

        θ_c = self.cbf_col_theta
        grad_h2_x = torch.cat((torch.sin(X[:, 0:1] - θ_c),
                               X.new_zeros(X.shape[0],1)),
                              dim=-1)
        if X_in.ndim == 1:
            grad_h2_x = grad_h2_x.squeeze(0)
        return grad_h2_x

    def lie_f_h2_col(self, x):
        (θ, ω) = x
        θ_c = self.cbf_col_theta
        direct = ω * torch.sin(θ-θ_c)
        abstract = self.grad_h2_col(x) @ self.f_func(x)
        assert torch.allclose(direct, abstract, atol=1e-4)
        return direct


    def grad_lie_f_h2_col(self, x):
        (θ, ω) = x
        θ_c = self.cbf_col_theta
        direct = torch.tensor([ω * torch.cos(θ-θ_c), torch.sin(θ-θ_c)], dtype=self.dtype)
        with variable_required_grad(x):
            abstract = torch.autograd.grad(self.lie_f_h2_col(x), x)[0]
        assert torch.allclose(direct, abstract, atol=1e-4)
        return direct

    def lie2_f_h_col(self, x):
        (θ, ω) = x
        m, l, g = self.mass, self.length, self.gravity
        Δ_c = self.cbf_col_delta
        θ_c = self.cbf_col_theta
        direct =  ω**2 * torch.cos(θ - θ_c) - (g / l) * torch.sin(θ - θ_c) * torch.sin(θ)
        abstract = self.grad_lie_f_h2_col(x) @ self.f_func(x)
        assert torch.allclose(direct, abstract, atol=1e-4)
        return direct

    def lie_g_lie_f_h_col(self, x):
        (θ, ω) = x
        m, l, g = self.mass, self.length, self.gravity
        Δ_c = self.cbf_col_delta
        θ_c = self.cbf_col_theta
        direct = (1/(m*l)) * torch.sin(θ - θ_c)
        abstract = self.grad_lie_f_h2_col(x) @ self.g_func(x)
        assert torch.allclose(direct, abstract, atol=1e-4)
        return direct

    def lie2_fu_h_col(self, x, u):
        grad_L1h = self.grad_lie_f_h2_col(x)
        return grad_L1h @ (self.f_func(x) + self.g_func(x) @ u)

    def A(self, x):
        return - self.lie_g_lie_f_h_col(x).unsqueeze(0)

    def b(self, x):
        K_α = torch.tensor(self.cbf_col_K_alpha, dtype=self.dtype)
        η_b_x = torch.cat([self.h2_col(x).unsqueeze(0), self.lie_f_h2_col(x).unsqueeze(0)])
        return (self.lie2_f_h_col(x) + K_α @ η_b_x)

    def __getattr__(self, name):
        return getattr(self.model, name)


class CBFSr(NamedAffineFunc):
    # UNUSED
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


class ConstraintPlotter:
    @store_args
    def __init__(self,
                 axes=None,
                 constraint_hists=[],
                 plotfile='plots/constraint_hists_{i}.pdf'):
        pass

    def plot_constraints(self, funcs, x, u, i=None):
        axs = self.axes
        nfuncs = len(funcs)
        if axs is None:
            nplots = nfuncs
            shape = ((math.ceil(nplots / 2), 2) if nplots >= 2 else (nplots,))
            fig, axs = plt.subplots(*shape)
            fig.subplots_adjust(wspace=0.35, hspace=0.5)
            axs = self.axes = axs.flatten() if hasattr(axs, "flatten") else np.array([axs])

        if len(self.constraint_hists) < nfuncs:
            self.constraint_hists = self.constraint_hists + [
                list() for _ in range(
                nfuncs - len(self.constraint_hists))]

        for i, af in enumerate(funcs):
            self.constraint_hists[i].append(af(x, u))

        if torch.rand(1) <= 0.2:
            for i, (ch, af) in enumerate(zip(self.constraint_hists, funcs)):
                axs[i].clear()
                axs[i].plot(ch)
                axs[i].set_ylabel(af.__name__)
                axs[i].set_xlabel("time")
                plt.pause(0.0001)
            if i is not None:
                plt_savefig_with_data(axs[i].figure, self.plotfile.format(i=i))
        return axs


class PendulumCBFCLFDirect(Controller):
    needs_ground_truth = True
    @store_args
    def __init__(self, mass=None, length=None, gravity=None, dt=None,
                 constraint_plotter_class=ConstraintPlotter,
                 pendulum_dynamics_class=PendulumDynamicsModel,
                 true_model=None,
                 plotfile=None,
                 dtype=torch.get_default_dtype()
    ):
        self.set_model_params(mass=mass, length=length, gravity=gravity)
        self.constraint_plotter = constraint_plotter_class(plotfile=plotfile.format(suffix='_constraint_{i}'))

    def set_model_params(self, **kwargs):
        self.model = self.pendulum_dynamics_class(m=1, n=2, **kwargs)
        self.aff_constraints = [
            #NamedAffineFunc(self.A_clf, self.b_clf, "clf"),
            EnergyCLF(self.model),
            RadialCBFRelDegree2(self.model, dtype=self.dtype),
            #NamedAffineFunc(self.A_col, self.b_col, "col")
        ]

    def f_func(self, x):
        return self.model.f_func(torch.tensor(x))

    def g_func(self, x):
        return self.model.g_func(torch.tensor(x))

    def control(self, xi, i=None):
        plotfile = self.plotfile
        u = control_QP_cbf_clf(
            xi,
            ctrl_aff_constraints=self.aff_constraints,
            constraint_margin_weights=[100])
        self.constraint_plotter.plot_constraints(
            self.aff_constraints + [
                NamedFunc(lambda x, u: f.value(x), r"\verb!%s!" % f.__name__)
                for f in self.aff_constraints],
            xi, u)
        return u

class ControlPendulumCBFLearned(ControlCBFLearned):
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
                 train_every_n_steps=10,
                 mean_dynamics_model_class=MeanPendulumDynamicsModel,
                 egreedy_scheme=[1, 0.01],
                 iterations=100,
                 max_unsafe_prob=0.01,
                 dt=0.001,
                 max_train=200,
                 #gamma_length_scale_prior=[1/deg2rad(0.1), 1],
                 gamma_length_scale_prior=None,
                 constraint_plotter_class=ConstraintPlotter,
                 true_model=None,
                 plotfile='plots/ctrl_cbf_learned_{suffix}.pdf',
                 dtype=torch.get_default_dtype(),
                 use_ground_truth_model=False
    ):
        self.Xtrain = []
        self.Utrain = []
        if self.use_ground_truth_model:
            self.model = self.true_model
        else:
            self.model = ControlAffineRegressor(
                x_dim, u_dim,
                gamma_length_scale_prior=gamma_length_scale_prior)
        self.mean_dynamics_model = mean_dynamics_model_class(m=u_dim, n=x_dim)
        self.ctrl_aff_constraints=[EnergyCLF(self),
                                   RadialCBF(self)]
        self.cbf2 = RadialCBFRelDegree2(self.model, dtype=dtype)
        self.ground_truth_cbf2 = RadialCBFRelDegree2(self.true_model, dtype=dtype)
        self._has_been_trained_once = False
        # These are used in the optimizer hence numpy
        self.x_goal = torch.tensor([theta_goal, omega_goal])
        self.x_quad_goal_cost = torch.tensor(quad_goal_cost)
        self.axes = [None, None]
        self.constraint_plotter = constraint_plotter_class(
            plotfile=plotfile.format(suffix='_constraints_{i}'))

    def debug_train(self, Xtrain, Utrain, XdotError):
        XdotErrorGot_train_mean, _ = self.model.predict_flatten(Xtrain[:-1], Utrain[:-1])
        assert torch.allclose(XdotErrorGot_train_mean, XdotError, rtol=0.4, atol=0.1), """
            Train data check using original flatten predict """
        print("hat f(x; u)[0] ∈ [{}, {}]".format(XdotErrorGot_train_mean[:, 0].min(),
                                                   XdotErrorGot_train_mean[:, 0].max()))
        print("hat f(x; u)[1] ∈ [{}, {}]".format(XdotErrorGot_train_mean[:, 1].min(),
                                                   XdotErrorGot_train_mean[:, 1].max()))
        print("f(x; u)[0] ∈ [{}, {}]".format(XdotError[:, 0].min(),
                                               XdotError[:, 0].max()))
        print("f(x; u)[1] ∈ [{}, {}]".format(XdotError[:, 1].min(),
                                               XdotError[:, 1].max()))

        fx = self.model.f_func(Xtrain[:-1])
        print("hat f(x)[0] ∈ [{}, {}]".format(fx[:, 0].min(),
                                                fx[:, 0].max()))
        print("hat f(x)[1] ∈ [{}, {}]".format(fx[:, 1].min(),
                                                fx[:, 1].max()))
        fxtrue = self.true_model.f_func(Xtrain[:-1])
        print("f(x)[0] ∈ [{}, {}]".format(fxtrue[:, 0].min(),
                                            fxtrue[:, 0].max()))
        print("f(x)[1] ∈ [{}, {}]".format(fxtrue[:, 1].min(),
                                            fxtrue[:, 1].max()))

        gxu = self.model.gu_func(Xtrain[:-1], Utrain[:-1])
        print("hat g(x; u)[0] ∈ [{}, {}]".format(gxu[:, 0].min(),
                                                gxu[:, 0].max()))
        print("hat g(x; u)[1] ∈ [{}, {}]".format(gxu[:, 1].min(),
                                                gxu[:, 1].max()))
        gxutrue = self.true_model.g_func(Xtrain[:-1]).bmm(Utrain[:-1].unsqueeze(-1)).squeeze(-1)
        print("g(x; u)[0] ∈ [{}, {}]".format(gxutrue[:, 0].min(),
                                               gxutrue[:, 0].max()))
        print("g(x; u)[1] ∈ [{}, {}]".format(gxutrue[:, 1].min(),
                                               gxutrue[:, 1].max()))

        fxu = fx + gxu
        print("hat f(x; u)[0] ∈ [{}, {}]".format(fxu[:, 0].min(),
                                                   fxu[:, 0].max()))
        print("hat f(x; u)[1] ∈ [{}, {}]".format(fxu[:, 1].min(),
                                                   fxu[:, 1].max()))
        fxutrue = fxtrue + gxutrue
        print("f(x; u)[0] ∈ [{}, {}]".format(fxutrue[:, 0].min(),
                                               fxutrue[:, 0].max()))
        print("f(x; u)[1] ∈ [{}, {}]".format(fxutrue[:, 1].min(),
                                               fxutrue[:, 1].max()))



run_pendulum_control_trival = partial(
    run_pendulum_experiment, controller_class=ControlTrivial,
    plotfile='plots/run_pendulum_control_trival{suffix}.pdf')
"""
Run pendulum with a trivial controller.
"""


run_pendulum_control_cbf_clf = partial(
    run_pendulum_experiment, controller_class=PendulumCBFCLFDirect,
    plotfile='plots/run_pendulum_control_cbf_clf{suffix}.pdf',
    theta0=5*math.pi/12,
    tau=0.002,
    numSteps=15000)
"""
Run pendulum with a safe CLF-CBF controller.
"""

class ControlCBFCLFGroundTruth(ControlPendulumCBFLearned):
    """
    Controller that avoids learning but uses the ground truth model
    """
    needs_ground_truth = False
    def __init__(self, *a, **kw):
        assert kw.pop("use_ground_truth_model", False) is False
        super().__init__(*a, use_ground_truth_model=True, **kw)


run_pendulum_control_online_learning = partial(
    run_pendulum_experiment,
    plotfile='plots/run_pendulum_control_online_learning{suffix}.pdf',
    controller_class=ControlPendulumCBFLearned,
    numSteps=3000,
    theta0=5*math.pi/12,
    tau=2e-3,
    dtype=torch.float64)
"""
Run save pendulum control while learning the parameters online
"""

run_pendulum_control_ground_truth = partial(
    run_pendulum_experiment,
    plotfile='plots/run_pendulum_control_ground_truth{suffix}.pdf',
    controller_class=ControlCBFCLFGroundTruth,
    numSteps=2500,
    theta0=5*math.pi/12,
    tau=1e-2)
"""
Run save pendulum control with ground_truth model
"""



if __name__ == '__main__':
    #run_pendulum_control_trival()
    #run_pendulum_control_cbf_clf()
    #learn_dynamics()
    # run_pendulum_control_ground_truth()
    run_pendulum_control_online_learning()

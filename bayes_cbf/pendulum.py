# stable pendulum
import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

import timeit
import random
import warnings
import sys
import io
import tempfile
import inspect
from collections import namedtuple, OrderedDict
from functools import partial, wraps
import pickle
import hashlib
import math
from abc import ABC, abstractmethod
import os.path as osp
import glob
import os
import subprocess

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc as mplibrc
mplibrc('text', usetex=True)

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_file_loader

from bayes_cbf.control_affine_model import (ControlAffineRegressor, LOG as CALOG,
                                            ControlAffineRegressorExact,
                                            ControlAffineRegressorVector,
                                            ControlAffineRegMatrixDiag,
                                            ControlAffineRegVectorDiag,
                                            is_psd)
CALOG.setLevel(logging.WARNING)

from bayes_cbf.plotting import (plot_results, plot_learned_2D_func_from_data,
                                plt_savefig_with_data, plot_2D_f_func,
                                speed_test_matrix_vector_plot)
from bayes_cbf.sampling import (sample_generator_trajectory, controller_sine,
                                Visualizer, VisualizerZ, uncertainity_vis_kwargs)
from bayes_cbf.controllers import (Controller, ControlCBFLearned,
                                   NamedAffineFunc, NamedFunc, ConstraintPlotter)
from bayes_cbf.misc import (t_vstack, t_hstack, to_numpy, store_args,
                            DynamicsModel, ZeroDynamicsModel, variable_required_grad,
                            epsilon, add_tensors, gitdescribe, TBLogger,
                            load_tensorboard_scalars, ensuredirs)
from bayes_cbf.cbc2 import cbc2_quadratic_terms, cbc2_gp, RelDeg2Safety


class ControlTrivial(Controller):
    needs_ground_truth = True
    @store_args
    def __init__(self, m=1, mass=None, length=None, gravity=None, dt=None,
                 true_model=None):
        pass

    def control(self, xi, t=None):
        mass, gravity, length = self.mass, self.gravity, self.length
        theta, w = xi
        u = mass * gravity * torch.sin(theta)
        return torch.tensor([u])


class ControlRandom(Controller):
    needs_ground_truth = True
    @store_args
    def __init__(self, **kwargs):
        self.control_trivial = ControlTrivial(**kwargs)

    def control(self, xi, t=None):
        return self.control_trivial.control(
            xi, t=t
        ) * (torch.rand(1)*0.8+0.60)



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
        shape = x.shape[:-1] if x.ndim == 2 else 1
        noise = torch.random.normal(scale=self.model_noise, dtype=self.dtype) if self.model_noise else 0
        gx = torch.tensor([[0], [1/(mass*length)]], dtype=self.dtype)
        if x.ndim >= 2:
            return noise + torch.ones(*x.shape[:-1], 1, 1) * gx
        else:
            return noise + gx


class PendulumVisualizer(Visualizer):
    def __init__(self, plotfile='data/plots/pendulum_data_{t}.pdf',
                 plot_every_n_steps=20):
        self.plotfile = plotfile
        self.plot_every_n_steps = plot_every_n_steps

        self._reset()

    def _reset(self):
        # initializations
        self.omega_vec = []
        self.theta_vec = []
        self.u_vec = []
        self.axs = None

    def setStateCtrl(self, x, u, t, **kw):
        self.theta_vec.append(float(x[0]))
        self.omega_vec.append(float(x[1]))
        self.u_vec.append(float(u))

        if t % self.plot_every_n_steps == 0:
            self.axs = plot_results(np.arange(t+1),
                                    np.asarray(self.omega_vec),
                                    np.asarray(self.theta_vec),
                                    np.asarray(self.u_vec),
                                    axs=self.axs)
            plt_savefig_with_data(self.axs.flatten()[0].figure,
                                ensuredirs(self.plotfile.format(t=t)))
            plt.pause(0.001)


def sampling_pendulum(dynamics_model, numSteps,
                      controller=None,
                      x0=None,
                      dt=0.01,
                      plot_every_n_steps=20,
                      axs=None,
                      visualizer=None,
                      visualizer_class=PendulumVisualizer,
                      plotfile='data/plots/pendulum_data_{t}.pdf'):
    if visualizer is None:
        visualizer = visualizer_class(plotfile=plotfile,
                                      plot_every_n_steps=plot_every_n_steps)
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

    for t in range(numSteps):
        time_vec[t] = tau*t
        theta_vec[t] = theta
        omega_vec[t] = omega
        u= controller(torch.tensor((theta, omega)), t=t)
        u_vec[t] = u

        if 0<theta_vec[t]<math.pi/4:
            damage_vec[t]=1

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
        visualizer.setStateCtrl(
            Xold[0], u, t=t,
            **uncertainity_vis_kwargs(controller, Xold[0], u, tau))

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


class PendulumVisualizer(Visualizer):
    @store_args
    def __init__(self, length, unsafe_c, unsafe_delta, plotfile='data/plots/visualizer/{t:04d}.png'):
        self.fig, self.axes = plt.subplots(1,1)
        self.fig.suptitle('Pendulum')
        self.count = 0

    def setStateCtrl(self, x, u, t=0, xtp1=None, xtp1_var=None):
        ax = self.axes
        ax.clear()
        ax.set_aspect('equal')
        ax.set_axis_off()
        l = self.length
        ax.set_xlim(-.05*l, 1.05*l)
        ax.set_ylim(-1.05*l, 1.05*l)
        c = self.unsafe_c - np.pi/2
        Δ = self.unsafe_delta
        θ = x[0] - np.pi/2
        ax.plot([0, l*math.cos(θ)],
                [0, l*math.sin(θ)], 'b-o', linewidth=2, markersize=10)
        ax.fill([0, l*math.cos(c + Δ), l*math.cos(c - Δ)],
                [0, l*math.sin(c + Δ), l*math.sin(c - Δ)], 'r')
        if xtp1 is not None and xtp1_var is not None:
            xtp1_theta = xtp1[0] - np.pi/2
            xtp1_theta_var = xtp1_var[0,0]
            ax.plot([0, l*math.cos(xtp1_theta)],
                    [0, l*math.sin(xtp1_theta)], 'b-o', linewidth=1, markersize=5)
            ax.fill([0, l*math.cos(xtp1_theta + xtp1_theta_var), l*math.cos(xtp1_theta - xtp1_theta_var)],
                    [0, l*math.sin(xtp1_theta + xtp1_theta_var), l*math.sin(xtp1_theta - xtp1_theta_var)],
                    'g--')

        self.fig.savefig(self.plotfile.format(t=self.count))
        self.count += 1
        plt.draw()


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
        plotfile='data/plots/run_pendulum_experiment{suffix}.pdf',
        dtype=torch.float32):
    torch.set_default_dtype(dtype)
    pendulum_model = pendulum_dynamics_class(m=1, n=2, mass=mass, gravity=gravity,
                                             length=length, dtype=dtype)
    if controller_class.needs_ground_truth:
        controller_object = controller_class(mass=mass, gravity=gravity,
                                             length=length, dt=tau,
                                             true_model=pendulum_model,
                                             plotfile=plotfile.format(suffix='_ctrl_{suffix}'),
                                             dtype=dtype,
                                             numSteps=numSteps
        )
        controller = controller_object.control
    else:
        controller_object = controller_class(dt=tau, true_model=pendulum_model,
                                             plotfile=plotfile.format(suffix='_ctrl_{suffix}'),
                                             dtype=dtype,
                                             numSteps=numSteps

        )
        controller = controller_object.control
    damge_perc,time_vec,theta_vec,omega_vec,u_vec = sampling_pendulum(
        pendulum_model,
        numSteps, x0=torch.tensor([theta0,omega0]), controller=controller, dt=tau,
        visualizer=PendulumVisualizer(length=length,
                                      unsafe_c=controller_object.cbf2.cbf_col_theta,
                                      unsafe_delta=controller_object.cbf2.cbf_col_delta),
        plotfile=plotfile.format(suffix='_trajectory_{t}'))
    plot_results(time_vec, omega_vec, theta_vec, u_vec)

    for i in plt.get_fignums():
        suffix='_%d' % i if i > 0 else ''
        plt_savefig_with_data(plt.figure(i), plotfile.format(suffix=suffix))
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)

def learn_dynamics_from_data(dX, X, U, pend_env, regressor_class, logger, max_train, tags=[]):
    numSteps = X.shape[0]
    UH = t_hstack((torch.ones((U.shape[0], 1), dtype=U.dtype), U))

    # Do not need the full dataset . Take a small subset
    N = min(numSteps-1, max_train)
    shuffled_range = torch.randint(numSteps - 1, size=(N,))
    XdotTrain = dX[shuffled_range, :]
    Xtrain = X[shuffled_range, :]
    Utrain = U[shuffled_range, :]
    #gp = GaussiatorchrocessRegressor(kernel=kernel_xu,
    #                              alpha=1e6).fit(Z_shuffled, Y_shuffled)
    dgp = regressor_class(Xtrain.shape[-1], Utrain.shape[-1])
    dgp.fit(Xtrain, Utrain, XdotTrain, training_iter=50)
    #dgp.save()

    # Plot the pendulum trajectory
    logger.add_tensors("train", dict(Xtrain=Xtrain, Utrain=Utrain), 0)
    Xtrain_numpy = Xtrain.detach().cpu().numpy()
    plot_results(torch.arange(U.shape[0]), omega_vec=X[:, 0],
                 theta_vec=X[:, 1], u_vec=U[:, 0])
    log_learned_model(Xtrain_numpy, dgp,
                      pend_env.F_func,
                      key="/".join(tags + ["Fx"]),
                      logger=logger)

    return dgp, Xtrain_numpy

def learn_dynamics_exp(
        theta0=5*math.pi/6,
        omega0=-0.01,
        tau=0.01,
        mass=1,
        gravity=10,
        length=1,
        max_train=200,
        numSteps=1000,
        regressor_class=ControlAffineRegressor,
        logger_class=partial(TBLogger,
                           exp_tags=['learn_dynamics'], runs_dir='data/runs'),
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

    logger = logger_class()
    pend_env = pendulum_dynamics_class(m=1, n=2, mass=mass, gravity=gravity,
                                       length=length)
    dX, X, U = sampling_pendulum_data(
        pend_env, D=numSteps, x0=torch.tensor([theta0,omega0]),
        dt=tau,
        controller=ControlRandom(mass=mass, gravity=gravity, length=length).control)
    for t,  (dx, x, u) in enumerate(zip(dX, X, U)):
        logger.add_tensors("traj", dict(dx=dx, x=x, u=u), t)

    dgp, Xtrain_numpy = learn_dynamics_from_data(dX, X, U, pend_env, regressor_class, logger,
                                                 max_train=max_train, tags=[])
    return dgp, dX, U, logger


def learn_dynamics(**kw):
    dgp, dX, U, logger = learn_dynamics_exp(**kw)
    events_file = max(
        glob.glob(osp.join(logger.experiment_logs_dir, "*.tfevents*")),
        key=lambda f: os.stat(f).st_mtime)
    learn_dynamics_plot_from_log(events_file)


def get_grid_from_Xtrain(Xtrain):
    theta_range = slice(Xtrain[:, 0].min(), Xtrain[:, 0].max(),
                        (Xtrain[:, 0].max() - Xtrain[:, 0].min()) / 20)
    omega_range = slice(Xtrain[:, 1].min(), Xtrain[:, 1].max(),
                        (Xtrain[:, 1].max() - Xtrain[:, 1].min()) / 20)

    theta_omega_grid = np.mgrid[theta_range, omega_range]
    return theta_omega_grid


def Xtest_from_theta_omega_grid(theta_omega_grid, xsample):
    # Plot true f(x)
    _, N, M = theta_omega_grid.shape
    D = xsample.shape[-1]
    Xgrid = torch.empty((N * M, D), dtype=torch.float32)
    Xgrid[:, :] = torch.from_numpy(xsample)
    Xgrid[:, :2] = torch.from_numpy(
        theta_omega_grid.transpose(1, 2, 0).reshape(-1, 2)).to(torch.float32)
    return Xgrid.reshape(N, M, D)


def evaluate_func_on_grid(theta_omega_grid, f_func, xsample):
    _, N, M = theta_omega_grid.shape
    D = xsample.shape[-1]
    Xgrid = Xtest_from_theta_omega_grid(theta_omega_grid, xsample)
    FX = f_func(Xgrid.reshape(-1, D)).reshape(N, M, D)
    return to_numpy(FX)


def log_learned_model(Xtrain, model, true_f_func,
                      key="Fx",
                      logger=None):
    theta_omega_grid = get_grid_from_Xtrain(Xtrain)
    D = Xtrain.shape[-1]
    _, N, M = theta_omega_grid.shape
    Xtest = Xtest_from_theta_omega_grid(theta_omega_grid, Xtrain[0, :])
    # b(1+m)n
    FX_learned, var_FX = model.custom_predict_fullmat(Xtest.reshape(-1, D))
    n = model.x_dim
    m = model.u_dim
    assert FX_learned.shape == (N*M*(1+m)*n,)
    FX_learned = FX_learned.reshape(N, M, (1+m), n)
    var_FX = var_FX.reshape(N, M, (1+m), n, N, M, (1+m), n)
    assert not torch.isnan(FX_learned).any()
    assert not torch.isnan(var_FX).any()
    FX_true = true_f_func(Xtest)
    assert FX_true.shape == (N, M, n, (1+m))
    FX_true = FX_true.transpose(-1, -2) # (N, M, (1+m), n)
    logger.add_tensors("/".join(("log_learned_model", key)),
                       dict(Xtrain=Xtrain,
                            theta_omega_grid=theta_omega_grid,
                            FX_learned=to_numpy(FX_learned),
                            var_FX=to_numpy(var_FX),
                            FX_true=to_numpy(FX_true)),
                       0)


def plot_learned_2D_func(Xtrain, learned_f_func, true_f_func, axtitle='f(x)[{i}]',
                         figtitle='Learned vs True'):
    theta_omega_grid = get_grid_from_Xtrain(Xtrain)
    FX_learned = evaluate_func_on_grid(theta_omega_grid, learned_f_func, Xtrain[-1, :])
    FX_true = evaluate_func_on_grid(theta_omega_grid, true_f_func, Xtrain[-1, :])

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False,
                            figsize=(5, 3.5))
    return plot_learned_2D_func_from_data(theta_omega_grid, FX_learned,
                                          FX_true, Xtrain, axtitle=axtitle,
                                          figtitle=figtitle, axs=axs)


def learn_dynamics_plot_from_log(
        figtitle='Learned vs True',
        events_file='saved-runs/learn_dynamics_v1.0.2/events.out.tfevents.1606951767.dwarf.13002.0'):
    """
    """
    logdata = load_tensorboard_scalars(events_file)
    events_dir = osp.dirname(events_file)
    fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, squeeze=False,
                            figsize=(10, 3.5))
    theta_omega_grid = logdata['plot_learned_2D_func/fx/true/theta_omega_grid'][0][1]
    FX_learned = logdata['plot_learned_2D_func/fx/learned/FX'][0][1]
    FX_true = logdata['plot_learned_2D_func/fx/true/FX'][0][1]
    Xtrain = logdata['plot_learned_2D_func/fx/Xtrain'][0][1]
    plot_learned_2D_func_from_data(theta_omega_grid, FX_learned, FX_true, Xtrain,
                                   axtitle='f(x)[{i}]', figtitle=figtitle,
                                   axs=axs[:, :2])

    GX_learned = logdata['plot_learned_2D_func/gx/learned/FX'][0][1]
    GX_true = logdata['plot_learned_2D_func/gx/true/FX'][0][1]
    plot_learned_2D_func_from_data(theta_omega_grid, GX_learned, GX_true, Xtrain,
                                   axtitle='g(x)[{i}]', figtitle=figtitle,
                                   axs=axs[:, 2:],
                                   ylabel=None)
    xmin = np.min(theta_omega_grid[0, ...])
    xmax = np.max(theta_omega_grid[0, ...])
    ymin = np.min(theta_omega_grid[1, ...])
    ymax = np.max(theta_omega_grid[1, ...])
    for ax in axs.flatten():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    fig.suptitle(figtitle)
    if hasattr(fig, "canvas") and hasattr(fig.canvas, "set_window_title"):
        fig.canvas.set_window_title(figtitle)
    fig.subplots_adjust(wspace=0.2, hspace=0.3, left=0.05, right=0.95, bottom=0.15)
    plot_file = osp.join(events_dir, 'learned_f_g_vs_true_f_g.pdf')
    fig.savefig(plot_file)
    #subprocess.run(["xdg-open", plot_file])
    return plot_file

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
                 k_alpha=[1., 1.],
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


class RadialCBFRelDegree2(RelDeg2Safety, NamedAffineFunc):
    @partial(store_args, skip=["model", "max_unsafe_prob"])
    def __init__(self, model,
                 cbf_col_gamma=1,
                 _k_alpha=[1., 3.],
                 cbf_col_delta=math.pi/8,
                 cbf_col_theta=math.pi/4,
                 theta_c=math.pi/4,
                 gamma_col=1,
                 max_unsafe_prob=0.01,
                 delta_col=math.pi/8,
                 name="cbf-r2",
                 dtype=torch.get_default_dtype()):
        self._model = model
        self._max_unsafe_prob = max_unsafe_prob

    @property
    def k_alpha(self):
        return self._k_alpha

    @property
    def model(self):
        return self._model

    @property
    def max_unsafe_prob(self):
        return self._max_unsafe_prob

    def to(self, dtype):
        self.dtype = dtype
        self.model.to(dtype=dtype)

    def cbf(self, x):
        (theta, w) = x
        delta_col = self.cbf_col_delta
        theta_c = self.cbf_col_theta
        return math.cos(delta_col) - torch.cos(theta - theta_c)

    value = cbf

    def __call__ (self, x, u):
        return self.A(x) @ u - self.b(x)

    def grad_cbf(self, X_in):
        if X_in.ndim == 1:
            X = X_in.unsqueeze(0)

        θ_c = self.cbf_col_theta
        grad_h2_x = torch.cat((torch.sin(X[:, 0:1] - θ_c),
                               X.new_zeros(X.shape[0],1)),
                              dim=-1)
        if X_in.ndim == 1:
            grad_h2_x = grad_h2_x.squeeze(0)
        return grad_h2_x

    def lie_f_cbf(self, x):
        (θ, ω) = x
        θ_c = self.cbf_col_theta
        direct = ω * torch.sin(θ-θ_c)
        abstract = self.grad_cbf(x) @ self.model.f_func(x)
        assert torch.allclose(direct, abstract, atol=1e-4)
        return direct


    def grad_lie_f_cbf(self, x):
        (θ, ω) = x
        θ_c = self.cbf_col_theta
        direct = torch.tensor([ω * torch.cos(θ-θ_c), torch.sin(θ-θ_c)], dtype=self.dtype)
        with variable_required_grad(x):
            abstract = torch.autograd.grad(self.lie_f_cbf(x), x)[0]
        assert torch.allclose(direct, abstract, atol=1e-4)
        return direct

    def lie2_f_h_col(self, x):
        (θ, ω) = x
        m, l, g = self.model.mass, self.model.length, self.model.gravity
        Δ_c = self.cbf_col_delta
        θ_c = self.cbf_col_theta
        direct =  ω**2 * torch.cos(θ - θ_c) - (g / l) * torch.sin(θ - θ_c) * torch.sin(θ)
        abstract = self.grad_lie_f_cbf(x) @ self.model.f_func(x)
        assert torch.allclose(direct, abstract, atol=1e-4)
        return direct

    def lie_g_lie_f_h_col(self, x):
        (θ, ω) = x
        m, l, g = self.model.mass, self.model.length, self.model.gravity
        Δ_c = self.cbf_col_delta
        θ_c = self.cbf_col_theta
        direct = (1/(m*l)) * torch.sin(θ - θ_c)
        abstract = self.grad_lie_f_cbf(x) @ self.model.g_func(x)
        assert torch.allclose(direct, abstract, atol=1e-4)
        return direct

    def lie2_fu_h_col(self, x, u):
        grad_L1h = self.grad_lie_f_cbf(x)
        return grad_L1h @ (self.model.f_func(x) + self.model.g_func(x) @ u)

    def A(self, x):
        return - self.lie_g_lie_f_h_col(x).unsqueeze(0)

    def b(self, x):
        K_α = torch.tensor(self.k_alpha, dtype=self.dtype)
        η_b_x = torch.cat([self.cbf(x).unsqueeze(0), self.lie_f_cbf(x).unsqueeze(0)])
        return (self.lie2_f_h_col(x) + K_α @ η_b_x)


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
        self.constraint_plotter = constraint_plotter_class(plotfile=plotfile.format(suffix='_constraint_{t}'))

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
                 mean_dynamics_model_class=partial(ZeroDynamicsModel, m=1, n=2),
                 egreedy_scheme=[1, 0.01],
                 iterations=100,
                 dt=0.001,
                 max_train=200,
                 #gamma_length_scale_prior=[1/deg2rad(0.1), 1],
                 gamma_length_scale_prior=[np.pi/100, np.pi/100],
                 constraint_plotter_class=ConstraintPlotter,
                 true_model=None,
                 plotfile='data/plots/ctrl_cbf_learned_{suffix}.pdf',
                 dtype=torch.get_default_dtype(),
                 use_ground_truth_model=False,
                 numSteps=1000,
                 ctrl_range=[-15., 15.],
                 u_quad_cost=[[1.]],
    ):
        if self.use_ground_truth_model:
            self.model = self.true_model
        else:
            self.model = ControlAffineRegressor(
                x_dim, u_dim,
                gamma_length_scale_prior=gamma_length_scale_prior)
        self.ctrl_aff_constraints=[EnergyCLF(self),
                                   RadialCBF(self)]
        self.cbf2 = RadialCBFRelDegree2(self.model, dtype=dtype)
        self.ground_truth_cbf2 = RadialCBFRelDegree2(self.true_model, dtype=dtype)
        super().__init__(model=self.model,
                         x_dim=x_dim,
                         u_dim=u_dim,
                         train_every_n_steps=train_every_n_steps,
                         mean_dynamics_model_class=mean_dynamics_model_class,
                         dt=dt,
                         constraint_plotter_class=constraint_plotter_class,
                         plotfile=plotfile,
                         ctrl_range=ctrl_range,
                         x_goal = [theta_goal, omega_goal],
                         x_quad_goal_cost = quad_goal_cost,
                         u_quad_cost = u_quad_cost,
                         numSteps = numSteps,
                         cbfs = [self.cbf2],
                         ground_truth_cbfs = [self.ground_truth_cbf2])
        # These are used in the optimizer hence numpy
        self.axes = [None, None]

    def debug_train(self, Xtrain, Utrain, XdotError):
        XdotErrorGot_train_mean = self.model.fu_func_mean(Utrain[:-1], Xtrain[:-1])
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
    plotfile='data/plots/run_pendulum_control_trival{suffix}.pdf')
"""
Run pendulum with a trivial controller.
"""


run_pendulum_control_cbf_clf = partial(
    run_pendulum_experiment, controller_class=PendulumCBFCLFDirect,
    plotfile='data/plots/run_pendulum_control_cbf_clf{suffix}.pdf',
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
        super().__init__(*a, use_ground_truth_model=True,
                         mean_dynamics_model_class=PendulumDynamicsModel,
                         **kw)


run_pendulum_control_online_learning = partial(
    run_pendulum_experiment,
    plotfile='data/plots/run_pendulum_control_online_learning{suffix}.pdf',
    controller_class=ControlPendulumCBFLearned,
    numSteps=250,
    theta0=7*math.pi/12,
    tau=0.002,
    dtype=torch.float64)
"""
Run save pendulum control while learning the parameters online
"""

def learn_dynamics_matrix_vector_exp(
        exps=dict(matrix=dict(regressor_class=ControlAffineRegressorExact),
                  vector=dict(regressor_class=ControlAffineRegressorVector)),
        theta0=5*math.pi/6,
        omega0=-0.01,
        tau=0.01,
        mass=1,
        gravity=10,
        length=1,
        max_train=200,
        numSteps=1000,
        logger_class=partial(TBLogger,
                             exp_tags=['learn_matrix_vector'],
                             runs_dir='data/runs'),
        pendulum_dynamics_class=PendulumDynamicsModel
):
    logger = logger_class()
    pend_env = pendulum_dynamics_class(m=1, n=2, mass=mass, gravity=gravity,
                                       length=length)
    dX, X, U = sampling_pendulum_data(
        pend_env, D=numSteps, x0=torch.tensor([theta0,omega0]),
        dt=tau,
        controller=ControlRandom(mass=mass, gravity=gravity, length=length).control,
        plot_every_n_steps=numSteps)
    for t,  (dx, x, u) in enumerate(zip(dX, X, U)):
        logger.add_tensors("traj", dict(dx=dx, x=x, u=u), t)

    for name, kw in exps.items():
        dgp, _ = learn_dynamics_from_data(dX, X, U, pend_env,
                                          kw['regressor_class'], logger,
                                          max_train=max_train,
                                          tags=[name])
    events_file = max(
        glob.glob(osp.join(logger.experiment_logs_dir, "*.tfevents*")),
        key=lambda f: os.stat(f).st_mtime)
    return events_file


def measure_batch_error(FX_learned, var_FX, FX_true):
    N, D = FX_learned.shape
    assert FX_true.shape == (N, D)
    assert var_FX.shape == (N, D, D)
    FX_t_diff = FX_true - FX_learned
    errors = FX_t_diff.unsqueeze(-2).bmm(
        FX_t_diff.unsqueeze(-1).solve(var_FX).solution
    )
    assert (errors > 0).all()
    sq_sum = errors.reshape(-1).sum()
    assert not torch.isnan(sq_sum).any()
    assert sq_sum > 0
    return np.sqrt(to_numpy(sq_sum) / N)

def learn_dynamics_matrix_vector_plot(
        exps,
        exp_data,
        FX_true,
        Xtrain,
        theta_omega_grid,
        xlabel=r'$\theta$',
        ylabel=r'$\omega$',
        figtitle='Comparing GP Learning methods',
        n = 2,
        m = 1,
        exp_conf=dict(
            vector=dict(rowlabel='CoGP'),
            matrix=dict(rowlabel='MVGP')),
        collabels=[r'$f(x)_{i}$', r'$g(x)_{{{{i},1}}$']
):
    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, squeeze=False,
                            figsize=(10, 6.0))
    fig.subplots_adjust(wspace=0.2, hspace=0.2, left=0.10, right=0.95,
                        bottom=0.07, top=0.90)
    csets_fx = None
    csets_gx = None
    exp_error_data = []
    for e, exp in enumerate(exp_conf.keys()):
        FX_learned = exp_data[exp]['FX_learned']
        var_FX = exp_data[exp]['var_FX']
        FX_learned_t, var_FX_t, FX_true_t = map(
            torch.from_numpy, (FX_learned, var_FX, FX_true))
        b = int(np.prod(FX_learned_t.shape[:-2]))
        var_FX_t = var_FX_t.reshape(b, (1+m)*n, b, (1+m)*n)
        var_FX_diag_t = torch.empty((b, (1+m)*n, (1+m)*n),
                                    dtype=var_FX_t.dtype,
                                    device=var_FX_t.device)
        # Extract diagonal
        for i in range(b):
            var_FX_diag_t[i, :, :] = var_FX_t[i, :, i, :]
        error = measure_batch_error(FX_learned_t.reshape(-1, (1+m)*n),
                                    var_FX_diag_t,
                                    FX_true_t.reshape(-1, (1+m)*n))
        exp_error_data.append((exp, error))
        batch_shape = FX_learned.shape[:-2]
        # FX_learned.shape = (*b, (1+m), n)
        FX_learned = FX_learned.reshape(*batch_shape, (1+m), n)
        csets_fx = plot_2D_f_func(theta_omega_grid, FX_learned[:, :, 0, :],
                                  axes_gen=lambda _: axs[e+1, :2],
                                  axtitle=None,
                                  xsample=Xtrain[-1, :],
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  contour_levels=(None
                                                  if csets_fx is None else
                                                  [c.levels for c in csets_fx])
        )
        csets_gx = plot_2D_f_func(theta_omega_grid, FX_learned[:, :, 1, :],
                                  axes_gen=lambda _: axs[e+1, 2:],
                                  axtitle=None,
                                  xsample=Xtrain[-1, :],
                                  xlabel=xlabel,
                                  ylabel=None,
                                  contour_levels=(None
                                                  if csets_gx is None else
                                                  [c.levels for c in csets_gx])
        )
        pad = 5
        axs[e+1, 0].annotate(exp_conf[exp]['rowlabel'],
                             xy=(0.0, 0.5), # loc in axes fraction
                             xytext=(-axs[e+1, 0].yaxis.labelpad - pad, 5), # padding in pts
                             xycoords=axs[e+1, 0].yaxis.label,
                             textcoords='offset points',
                             size='large', ha='right', va='center',
                             rotation=90)


    batch_shape = FX_true.shape[:-2]
    FX_true = FX_true.reshape(*batch_shape, (1+m), n)
    csets_fx = plot_2D_f_func(theta_omega_grid, FX_true[:, :, 0, :],
                              axes_gen=lambda _: axs[0, :2],
                              axtitle="$f(x)_{i}$",
                              xsample=Xtrain[-1, :],
                              xlabel=xlabel,
                              ylabel=ylabel,
                              contour_levels=[c.levels for c in csets_fx])
    csets_gx = plot_2D_f_func(theta_omega_grid, FX_true[:, :, 1, :],
                              axes_gen=lambda _: axs[0, 2:],
                              axtitle="$g(x)_{{{i},1}}$",
                              xsample=Xtrain[-1, :],
                              xlabel=xlabel,
                              ylabel=None,
                              contour_levels=[c.levels for c in csets_gx])
    axs[0, 0].annotate('Ground Truth',
                       xy=(0.0, 0.5), # loc in axes fraction
                       xytext=(-axs[0, 0].yaxis.labelpad - pad, 5), # padding in pts
                       xycoords=axs[0, 0].yaxis.label,
                       textcoords='offset points',
                       size='large', ha='right', va='center',
                       rotation=90)

    xmin = np.min(theta_omega_grid[0, ...])
    xmax = np.max(theta_omega_grid[0, ...])
    ymin = np.min(theta_omega_grid[1, ...])
    ymax = np.max(theta_omega_grid[1, ...])
    for ax in axs[1:, :].flatten():
        ax.plot(Xtrain[:, 0], Xtrain[:, 1], marker='+', linestyle='', color='r')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    fig.suptitle(figtitle, fontsize='x-large')
    if hasattr(fig, "canvas") and hasattr(fig.canvas, "set_window_title"):
        fig.canvas.set_window_title(figtitle)
    return fig, exp_error_data

def learn_dynamics_matrix_vector_vis(
        exps=['matrix', 'vector'],
        events_file='saved-runs/learn_matrix_vector_v1.1.0/events.out.tfevents.1607382261.dwarf.5274.7'):
    logdata = load_tensorboard_scalars(events_file)
    events_dir = osp.dirname(events_file)
    theta_omega_grid = logdata['log_learned_model/matrix/Fx/theta_omega_grid'][0][1]
    FX_true = logdata['log_learned_model/matrix/Fx/FX_true'][0][1]
    Xtrain = logdata['log_learned_model/matrix/Fx/Xtrain'][0][1]
    exp_data = dict()
    for e, exp in enumerate(exps):
        exp_data[exp] = dict()
        exp_data[exp]['FX_learned'] = logdata['log_learned_model/' + exp + '/Fx/FX_learned'][0][1]
        exp_data[exp]['var_FX'] = logdata['log_learned_model/' + exp + '/Fx/var_FX'][0][1]
    fig, exp_error_data = learn_dynamics_matrix_vector_plot(
        exps, exp_data, FX_true, Xtrain, theta_omega_grid)
    error_file = osp.join(events_dir,
                          'vector_matrix_learning_error.txt')
    exp_names, exp_errors = zip(*exp_error_data)
    print(exp_names, exp_errors)
    np.savetxt(error_file, [exp_errors],
               fmt='%.03f',
               header=' '.join(exp_names))

    plot_file = osp.join(events_dir, 'learned_f_g_vs_true_f_g_mat_vec_ind.pdf')
    fig.savefig(plot_file)
    subprocess.run(["xdg-open", plot_file])
    return plot_file

def learn_dynamics_matrix_vector(**kw):
    events_file = learn_dynamics_matrix_vector_exp(**kw)
    learn_dynamics_matrix_vector_vis(events_file=events_file)


def compute_errors(regressor_class, sampling_callable, pend_env,
                   ntries=5, max_train=200, test_on_grid=False, ntest=400):
    error_list = []
    # b(1+m)n
    for _ in range(ntries):
        dX, X, U = sampling_callable()

        shuffled_order = np.arange(X.shape[0]-1)

        # Test train split
        np.random.shuffle(shuffled_order)
        shuffled_order_t = torch.from_numpy(shuffled_order)

        train_indices = shuffled_order_t[:max_train]
        Xtrain = X[train_indices, :]
        Utrain = U[train_indices, :]
        XdotTrain = dX[train_indices, :]

        if test_on_grid:
            theta_omega_grid = get_grid_from_Xtrain(to_numpy(Xtrain))
            Xtest = torch.from_numpy(
                theta_omega_grid.reshape(-1, Xtrain.shape[-1])).to(
                    dtype=Xtrain.dtype,
                    device=Xtrain.device)
        else:
            Xtest = X[shuffled_order_t[-ntest:], :]

        # b, 1+m, n
        FX_true = pend_env.F_func(Xtest).transpose(-2, -1) # (b, n, 1+m) -> (b, 1+m, n)

        dgp = regressor_class(Xtrain.shape[-1], Utrain.shape[-1])


        # b(1+m)n
        FX_learned, var_FX = dgp.custom_predict_fullmat(
            Xtest.reshape(-1, Xtest.shape[-1]))
        b = Xtest.shape[0]
        n = pend_env.state_size
        m = pend_env.ctrl_size

        var_FX_t = var_FX.reshape(b, (1+m)*n, b, (1+m)*n)
        var_FX_diag_t = torch.empty((b, (1+m)*n, (1+m)*n),
                                    dtype=var_FX_t.dtype,
                                    device=var_FX_t.device)
        for i in range(b):
            var_FX_diag_t[i, :, :] = var_FX_t[i, :, i, :]
        error = measure_batch_error(
            FX_learned.reshape(-1, (1+m)*n),
            var_FX_diag_t,
            FX_true.reshape(-1, (1+m)*n).to(
                dtype=FX_learned.dtype,
                device=FX_learned.device))
        error_list.append(error)
    return error_list


def speed_test_matrix_vector_exp(
        # max_train_variations=[16, 32], # testing GPU
        max_train_variations=[256, 256+64, 256+128, 256+256], # final GPU
        # max_train_variations=[10, 25, 50, 80, 125], # CPU
        # ntimes = 20, # How many times the inference should be repeated
        ntimes = 50, # How many times the inference should be repeated
        repeat = 5,
        errorbartries = 30,
        logger_class=partial(TBLogger,
                             exp_tags=['speed_test_matrix_vector'],
                             runs_dir='data/runs'),
        exps=dict(matrix=dict(regressor_class=ControlAffineRegressorExact),
                  vector=dict(regressor_class=ControlAffineRegressorVector),
                  matrixdiag=dict(regressor_class=ControlAffineRegMatrixDiag),
                  vectordiag=dict(regressor_class=ControlAffineRegVectorDiag)),
        theta0=5*math.pi/6,
        omega0=-0.01,
        tau=0.01,
        mass=1,
        gravity=10,
        length=1,
        numSteps=2000,
        pendulum_dynamics_class=PendulumDynamicsModel,
):
    logger = logger_class()
    pend_env = pendulum_dynamics_class(m=1, n=2, mass=mass, gravity=gravity,
                                       length=length)

    dX, X, U = sampling_pendulum_data(
        pend_env, D=numSteps, x0=torch.tensor([theta0,omega0]),
        dt=tau,
        visualizer=VisualizerZ(),
        controller=ControlRandom(mass=mass, gravity=gravity, length=length).control,
        plot_every_n_steps=numSteps)
    for t,  (dx, x, u) in enumerate(zip(dX, X, U)):
        logger.add_tensors("traj", dict(dx=dx, x=x, u=u), t)

    shuffled_order = np.arange(X.shape[0]-1)

    dgp = dict()
    for max_train in max_train_variations:

        # Test train split
        np.random.shuffle(shuffled_order)
        shuffled_order_t = torch.from_numpy(shuffled_order)

        train_indices = shuffled_order_t[:max_train]
        Xtrain = X[train_indices, :]
        Utrain = U[train_indices, :]
        XdotTrain = dX[train_indices, :]

        theta_omega_grid = get_grid_from_Xtrain(to_numpy(Xtrain))
        Xtest = torch.from_numpy(
            theta_omega_grid.reshape(-1, Xtrain.shape[-1])).to(
                dtype=Xtrain.dtype,
                device=Xtrain.device)
        # b, n, 1+m
        FX_true = pend_env.F_func(Xtest).transpose(-2, -1) # (b, n, 1+m) -> (b, 1+m, n)

        for name, kw in exps.items():
            dgp = kw['regressor_class'](Xtrain.shape[-1], Utrain.shape[-1])
            dgp.fit(Xtrain, Utrain, XdotTrain, training_iter=50)
            elapsed = min(timeit.repeat(
                stmt='dgp.custom_predict_fullmat(Xtest);dgp.clear_cache()',
                repeat=repeat,
                number=ntimes,
                globals=dict(dgp=dgp,
                             Xtest=Xtest)))
            errors = compute_errors(kw['regressor_class'],
                                    partial(
                                        sampling_pendulum_data,
                                        dynamics_model=pend_env,
                                        D=numSteps,
                                        x0=torch.tensor([theta0,omega0]),
                                        dt=tau,
                                        visualizer=VisualizerZ(),
                                        controller=ControlRandom(mass=mass, gravity=gravity, length=length).control,
                                        plot_every_n_steps=numSteps),
                                    pend_env,
                                    max_train=max_train,
                                    ntries=errorbartries)
            print(name, "training: ", max_train, "time:", elapsed, "error:",
                  np.mean(errors), "+-", np.std(errors))
            logger.add_scalars(name, dict(elapsed=elapsed / ntimes), max_train)
            logger.add_tensors(name, dict(errors=np.asarray(errors)), max_train)

    events_file = max(
        glob.glob(osp.join(logger.experiment_logs_dir, "*.tfevents*")),
        key=lambda f: os.stat(f).st_mtime)
    return events_file

def speed_test_matrix_vector_vis(
        events_file='',
        exp_conf=OrderedDict(
            vectordiag=dict(label='CoGP (diag)'),
            matrixdiag=dict(label='MVGP (diag)'),
            vector=dict(label='CoGP (full)'),
            matrix=dict(label='MVGP (full)')),
        marker_rotation=['b*-', 'g+-', 'r.-', 'k^-'],
        elapsed_ylabel='Inference time (secs)',
        error_ylabel='Variance weighted error',
        #error_ylabel=r'''$ \sqrt{\frac{1}{n}\sum_{\mathbf{x} \in \mathbf{X}_{test}} \left\|\mathbf{K}^{-\frac{1}{2}}_k(\mathbf{x}, \mathbf{x}) \mbox{vec}(\mathbf{M}_k(\mathbf{x})-F_{true}(\mathbf{x})) \right\|_2^2}$''',
        xlabel='Training samples'
):
    logdata = load_tensorboard_scalars(events_file)
    events_dir = osp.dirname(events_file)
    exp_data = dict()
    for gp, gp_conf in exp_conf.items():
        training_samples, elapsed = zip(*logdata[gp + '/elapsed'])
        training_samples, errors = zip(*logdata[gp + '/errors'])
        exp_data[gp] = dict(elapsed=elapsed, errors=errors)
    fig, axes = plt.subplots(1,2, figsize=(8, 4.7))
    fig.subplots_adjust(bottom=0.1, wspace=0.20, top=0.90, right=0.95, left=0.1)
    fig.suptitle('Pendulum')
    speed_test_matrix_vector_plot(axes,
                                  training_samples,
                                  exp_data,
                                  exp_conf=exp_conf,
                                  marker_rotation=marker_rotation,
                                  xlabel=xlabel,
                                  error_ylabel=error_ylabel,
                                  elapsed_ylabel=elapsed_ylabel)
    plot_file = osp.join(events_dir, 'speed_test_mat_vec_ind.pdf')
    fig.savefig(plot_file)
    subprocess.run(["xdg-open", plot_file])
    return plot_file


def speed_test_matrix_vector(**kw):
    events_file = speed_test_matrix_vector_exp(**kw)
    return speed_test_matrix_vector_vis(events_file)


if __name__ == '__main__':
    #run_pendulum_control_trival()
    #run_pendulum_control_cbf_clf()
    # learn_dynamics()
    #run_pendulum_control_online_learning()
    learn_dynamics_matrix_vector()
    speed_test_matrix_vector()

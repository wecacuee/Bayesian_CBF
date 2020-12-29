"""

Move to specified pose

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai(@Atsushi_twi)

P. I. Corke, "Robotics, Vision & Control", Springer 2017, ISBN 978-3-319-54413-7

"""
from datetime import datetime
from random import random
from functools import partial
from collections import namedtuple, OrderedDict
import inspect
import sys
import subprocess
import os.path as osp
import math
import logging
import json
import glob
import os
import timeit

from scipy.special import erfinv
from kwplus.variations import kwvariations, expand_variations
from kwplus.functools import recpartial
from kwplus import default_kw
logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

CONFIG_FILE_BASENAME = 'config.json'

import numpy as np
NUMPY_AS_TORCH = False
if NUMPY_AS_TORCH:
    import bayes_cbf.numpy2torch as torch
    import numpy.testing as testing
else:
    import torch
    def torch_to(arr, **kw):
        return arr.to(**kw)
    import torch.testing as testing
    torch.set_default_dtype(torch.float64)



from torch.utils.tensorboard import SummaryWriter

import bayes_cbf
from bayes_cbf.gp_algebra import DeterministicGP, GaussianProcess
from bayes_cbf.control_affine_model import (ControlAffineRegressor,
                                            ControlAffineRegressorRankOne,
                                            ControlAffineRegressorExact,
                                            ControlAffineRegressorExactRankOne,
                                            ControlAffineRegressorVector,
                                            ControlAffineRegMatrixDiag,
                                            ControlAffineRegVectorDiag)
from bayes_cbf.misc import (to_numpy, normalize_radians, ZeroDynamicsModel,
                            make_tensor_summary, add_tensors, gitdescribe,
                            stream_tensorboard_scalars, load_tensorboard_scalars,
                            mkdir_savefig, TBLogger, torch_kron)
from bayes_cbf.sampling import sample_generator_trajectory, VisualizerZ
from bayes_cbf.planner import PiecewiseLinearPlanner, SplinePlanner
from bayes_cbf.cbc2 import cbc2_quadratic_terms, cbc2_gp, cbc2_safety_factor
from bayes_cbf.plotting import (draw_ellipse, var_to_scale_theta,
                                speed_test_matrix_vector_plot)

TBLOG = None

PolarState = namedtuple('PolarState', 'rho alpha beta'.split())
CartesianState = namedtuple('CartesianState', 'x y theta'.split())
CartesianStateWithGoal = namedtuple('CartesianStateWithGoal',
                                    'state state_goal'.split())

def polar2cartesian(x: PolarState, state_goal : CartesianState) -> CartesianState:
    """
    rho is the distance between the robot and the goal position
    : \sqrt((x*-x)^2 + (y*-y)^2)

    alpha is the heading of the robot relative the angle to the goal
    : theta - atan2((y*-y),(x*-x))

    beta is the goal position relative to the angle to the goal
    : theta* - atan2((y*-y),(x*-x))

    >>> polar = (torch.rand(3) * torch.tensor([1, 2*math.pi, 2*math.pi]) -
    ...         torch.tensor([0, math.pi, math.pi]))
    >>> state_goal = (torch.rand(3) * torch.tensor([2, 2, 2*math.pi]) -
    ...              torch.tensor([1, 1, math.pi]))
    >>> state = polar2cartesian(polar, state_goal)
    >>> polarp = cartesian2polar(state, state_goal)
    >>> testing.assert_allclose(polar, polarp)
    """
    rho, alpha, beta = x
    x_goal, y_goal, theta_goal = state_goal
    phi = angdiff(theta_goal, beta)
    theta = normalize_radians(phi + alpha)
    x_diff = rho * torch.cos(phi)
    y_diff = rho * torch.sin(phi)
    return torch.tensor([x_goal - x_diff,
                     y_goal - y_diff,
                     theta])


def cartesian2polar(state: CartesianState, state_goal : CartesianState) -> PolarState:
    """
    rho is the distance between the robot and the goal position
    : \sqrt((x*-x)^2 + (y*-y)^2)
    alpha is the heading of the robot relative the angle to the goal
    : theta - atan2((y*-y),(x*-x))
    beta is the goal position relative to the angle to the goal
    : theta* - atan2((y*-y),(x*-x))

    >>> state = torch.rand(3)* torch.tensor([2, 2, 2*math.pi]) - torch.tensor([1, 1, math.pi])
    >>> state_goal = torch.rand(3)* torch.tensor([2, 2, 2*math.pi]) - torch.tensor([1, 1, math.pi])
    >>> polar = cartesian2polar(state, state_goal)
    >>> statep = polar2cartesian(polar, state_goal)
    >>> testing.assert_allclose(state, statep)
    """
    x, y, theta = state
    x_goal, y_goal, theta_goal = state_goal

    x_diff = x_goal - x
    y_diff = y_goal - y

    # reparameterization
    rho = torch.sqrt(x_diff**2 + y_diff**2)
    assert rho.abs() > 1e-6, "Invalid conversion"
    phi = torch.atan2(y_diff, x_diff)
    alpha = angdiff(theta, phi)
    beta = angdiff(theta_goal , phi)
    return torch.tensor((rho, alpha, beta))



class PolarDynamics:
    state_size = 3
    ctrl_size = 2
    def __init__(self):
        self.current_state = None

    def set_init_state(self, x0):
        self.current_state = x0

    def f_func(self, x : PolarState):
        return torch.zeros_like(x)

    def g_func(self, x : PolarState):
        rho, alpha, beta = x
        assert rho > 1e-6
        return torch.tensor([[-torch.cos(alpha), 0],
                             [-torch.sin(alpha)/rho, 1],
                             [-torch.sin(alpha)/rho, 0]])

    def step(self, u_torch, dt):
        x = self.current_state
        xdot = self.f_func(x) + self.g_func(x) @ u_torch
        self.current_state = x + xdot * dt
        return dict(xdot = xdot,
                    x = self.current_state)


class CartesianDynamics(PolarDynamics):
    state_size = 3
    ctrl_size = 2
    def f_func(self, x : CartesianState):
        return torch.zeros_like(x)

    def g_func(self, state_in: CartesianState):
        assert state_in.shape[-1] == self.state_size
        state = state_in.unsqueeze(0) if state_in.dim() <= 1 else state_in

        x, y, theta = state[..., 0:1], state[..., 1:2], state[..., 2:3]
        theta_cos = theta.cos().unsqueeze(-1)
        theta_sin = theta.sin().unsqueeze(-1)
        zeros_ = torch.zeros_like(theta_cos)
        ones_ = torch.ones_like(theta_cos)
        gX = torch.cat([torch.cat([theta_cos, zeros_], dim=-1),
                          torch.cat([theta_sin, zeros_], dim=-1),
                          torch.cat([zeros_,    ones_], dim=-1)], dim=-2)
        return gX.squeeze(0) if state_in.dim() <= 1 else gX

    def fu_func_gp(self, u):
        f, g = self.f_func, self.g_func
        n = self.state_size
        return GaussianProcess(
            mean = lambda x: f(x) + g(x) @ u,
            knl = lambda x, xp: (u @ u + 1) * torch.eye(n),
            shape=(n,),
            name="CartesianDynamics")


class AckermanDrive:
    """
    ẋ = v cos(θ)
    ẏ = v sin(θ)
    θ̇ = (v/L) tan(ϕ)

    L is the distance between front and back wheels
    state = [x, y, θ]
    input = [v, v tan(ϕ)]
    """
    state_size = 3
    ctrl_size = 2
    def __init__(self, L = 0.2, kernel_diag_A=[1.0, 1.0, 1.0]):
        self.L = L
        self.current_state = None
        self.kernel_diag_A = torch.tensor(kernel_diag_A)

    def set_init_state(self, x):
        self.current_state = x

    def f_func(self, x):
        """
        ṡ = f(s) + G(s) u

        s = [x, y, θ]
        u = [v, v tan(ϕ)]

               [ 0 ]
        f(s) = [ 0 ]
               [ 0 ]
        """
        return torch.zeros_like(x)

    def g_func(self, state_in):
        """
        ṡ = f(s) + G(s) u

        s = [x, y, θ]
        u = [v, v tan(ϕ)]

                [ cos(θ) , 0  ]
        G(s)u = [ sin(θ) , 0  ] [ v ]
                [ 0        1/L] [ v tan(ø) ]
        """
        assert state_in.shape[-1] == self.state_size
        state = state_in.unsqueeze(0) if state_in.dim() <= 1 else state_in

        x, y, theta = state[..., 0:1], state[..., 1:2], state[..., 2:3]
        theta_cos = theta.cos().unsqueeze(-1)
        theta_sin = theta.sin().unsqueeze(-1)
        zeros_ = torch.zeros_like(theta_cos)
        inv_L = torch.ones_like(theta_cos) / self.L
        gX = torch.cat([torch.cat([theta_cos, zeros_], dim=-1),
                        torch.cat([theta_sin, zeros_], dim=-1),
                        torch.cat([zeros_,    inv_L], dim=-1)], dim=-2)
        return gX.squeeze(0) if state_in.dim() <= 1 else gX

    def F_func(self, X):
        return torch.cat([self.f_func(X).unsqueeze(-1), self.g_func(X)], dim=-1)

    def fu_func_gp(self, u):
        f, g = self.f_func, self.g_func
        n = self.state_size
        self.kernel_diag_A = torch_to(self.kernel_diag_A,
                                     device=u.device,
                                     dtype=u.dtype)
        A = torch.diag(self.kernel_diag_A)
        u_hom = torch.cat([torch.tensor([1.]), u])
        B = torch.eye(self.ctrl_size + 1)
        return GaussianProcess(
            mean = lambda x: f(x) + g(x) @ u,
            knl = lambda x, xp:  (u_hom @ B @ u_hom) * A,
            shape=(n,),
            name="AckermanDrive")

    def step(self, u_torch, dt):
        x = self.current_state
        xdot = self.f_func(x) + self.g_func(x) @ u_torch
        self.current_state = x + xdot * dt
        return dict(xdot = xdot,
                    x = self.current_state)

    def custom_predict_fullmat(self, X):
        A = torch.diag(self.kernel_diag_A)
        B = torch.eye(self.ctrl_size + 1)
        b = X.shape[0] if X.ndim > 1 else 1
        bI = torch.eye(b)
        return (self.F_func(X).transpose(-2, -1).reshape(-1),
                torch_kron(bI,
                           torch_kron(B, A, batch_dims=0),
                           batch_dims=0))


class LearnedShiftInvariantDynamics:
    state_size = 3
    ctrl_size = 2
    def __init__(self,
                 dt = None,
                 learned_dynamics = None,
                 learned_dynamics_class = ControlAffineRegressorExactRankOne,
                 mean_dynamics = AckermanDrive(),
                 max_train = 200,
                 training_iter = 100,
                 shift_invariant = True,
                 train_every_n_steps = 20,
                 enable_learning = True):
        self.max_train = max_train
        self.training_iter = training_iter
        self.dt = dt
        self.mean_dynamics = mean_dynamics
        self.learned_dynamics = (learned_dynamics
                                 if learned_dynamics is not None else
                                 learned_dynamics_class(
                                     self.state_size,
                                     self.ctrl_size))
        self.shift_invariant = shift_invariant
        self._trans_invariant_wrapper = (self._make_trans_invariant
                                         if shift_invariant else
                                         lambda x: x)
        self.train_every_n_steps = train_every_n_steps
        self.enable_learning = enable_learning
        self.Xtrain = []
        self.Utrain = []

    def _make_trans_invariant(self, X):
        assert X.shape[-1] == self.state_size
        return torch.cat(
            [torch.zeros((*X.shape[:-1], self.state_size-1)),
             X[..., self.state_size-1:]], dim=-1)

    def f_func(self, X):
        x_orig = self._trans_invariant_wrapper(X)
        return self.mean_dynamics.f_func(x_orig) + self.learned_dynamics.f_func(x_orig)

    def g_func(self, X):
        x_orig = self._trans_invariant_wrapper(X)
        return self.mean_dynamics.g_func(x_orig) + self.learned_dynamics.g_func(x_orig)

    def train(self, xi, uopt):
        if (len(self.Xtrain) > 0
            and len(self.Xtrain) % int(self.train_every_n_steps) == 0
            and self.enable_learning
        ):
            # train every n steps
            Xtrain = torch.cat(self.Xtrain).reshape(-1, self.Xtrain[0].shape[-1])
            Utrain = torch.cat(self.Utrain).reshape(-1, self.Utrain[0].shape[-1])
            XdotTrain = (Xtrain[1:, :] - Xtrain[:-1, :]) / self.dt
            self.fit(Xtrain[:-1, :], Utrain[:-1, :], XdotTrain)

        # record the xi, ui pair
        self.Xtrain.append(xi.detach())
        self.Utrain.append(uopt.detach())
        assert len(self.Xtrain) == len(self.Utrain)

    def fit(self, Xtrain, Utrain, XdotTrain, training_iter=100):
        if not len(Xtrain):
            LOG.info("Nothing to fit")
            return
        LOG.info("Training GP with dataset size {}".format(len(Xtrain)))
        assert len(Xtrain) == len(Utrain), \
            "Call train when Xtrain and Utrain are balanced"

        Xtrain = self._trans_invariant_wrapper(Xtrain)

        XdotMean = self.mean_dynamics.f_func(Xtrain) + (
            self.mean_dynamics.g_func(Xtrain).bmm(Utrain.unsqueeze(-1)).squeeze(-1))
        XdotError = XdotTrain - XdotMean

        LOG.info("Training model with datasize {}".format(XdotTrain.shape[0]))
        if XdotTrain.shape[0] > self.max_train:
            indices = np.arange(XdotTrain.shape[0])
            shuffled_indices = torch.from_numpy(np.random.shuffle(indices))
            #indices = torch.randint(XdotTrain.shape[0], (self.max_train,))
            train_indices = (shuffled_indices[:self.max_train]
                             if XdotTrain.shape[0] > self.max_train
                             else shuffled_indices)
            train_data = Xtrain[train_indices, :], Utrain[train_indices, :], XdotError[train_indices, :],
        else:
            train_data = Xtrain, Utrain, XdotError

        self.learned_dynamics.fit(*train_data, training_iter=training_iter)

    def fu_func_gp(self, U):
        if self.enable_learning:
            md = self.mean_dynamics
            n = self.state_size
            return (DeterministicGP(lambda x: md.f_func(x) + md.g_func(x) @ U,
                                    shape=(n,))
                    + self.learned_dynamics.fu_func_gp(U))
        else:
            gp = self.mean_dynamics.fu_func_gp(U)
            return gp

    def step(self, u_torch, dt):
        x = self.current_state
        xdot = self.f_func(x) + self.g_func(x) @ u_torch
        self.current_state = x + xdot * dt
        return dict(xdot = xdot,
                    x = self.current_state)

    def custom_predict_fullmat(self, Xtest_in, **kw):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else  Xtest_in)
        if not self.enable_learning:
            meanFX, varFX = self.mean_dynamics.custom_predict_fullmat(Xtest)
            return  meanFX, varFX
        Xtest_orig = self._trans_invariant_wrapper(Xtest)
        with torch.no_grad():
            # (b(1+m)n)
            diffFX, diffVarFX = self.learned_dynamics.custom_predict_fullmat(Xtest_orig, **kw)
        # (bn(1+m))
        meanFX = torch_to(self.mean_dynamics.F_func(Xtest_orig),
                          dtype=diffFX.dtype,
                          device=diffFX.device)
        b = Xtest.shape[0]
        n = self.state_size
        m = self.ctrl_size
        return (meanFX.transpose(-2, -1).reshape(-1) + diffFX, # b(1+m)n
                diffVarFX) # (b(1+m)n, b(1+m)n)

    def clear_cache(self):
        self.learned_dynamics.clear_cache()


def angdiff(thetap, theta):
    return normalize_radians(thetap - theta)


def cosdist(thetap, theta):
    return 1 - torch.cos(thetap - theta)


def angdist(thetap, theta):
    return angdiff(thetap, theta)**2

class CLFPolar:
    def __init__(self,
                 Kp = torch.tensor([6., 15., 40., 0.])/10.):
        self.Kp = Kp

    def clf_terms(self, polar, state_goal):
        return self._clf_terms(polar, state_goal)

    def _clf_terms(self, polar, state_goal):
        rho, alpha, beta = polar
        return torch.tensor((0.5 * self.Kp[0] * rho ** 2,
                         self.Kp[1] * cosdist(alpha, 0),
                         self.Kp[2] * cosdist(beta, 0),
                         self.Kp[3] * (1-torch.cos(beta - alpha))
        ))

    def grad_clf(self, polar, state_goal):
        """
        >>> self = CLFPolar()
        >>> x0 = torch.rand(3)
        >>> state_goal = torch.rand(3)
        >>> ajac = self.grad_clf(x0, state_goal)
        >>> njac = numerical_jac(lambda x: self._clf_terms(x, state_goal).sum(), x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        return self._grad_clf_terms(polar, state_goal).sum(axis=-1)

    def grad_clf_wrt_goal(self, polar, state_goal):
        return torch.zeros_like(state_goal)

    def _grad_clf_terms(self, polar, state_goal):
        """
        >>> self = CLFPolar()
        >>> x0 = torch.rand(3)
        >>> x0_goal = torch.rand(3)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 0]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x,x0_goal)[0], x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 1]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x,x0_goal)[1], x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 2]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x,x0_goal)[2], x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 3]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x,x0_goal)[3], x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        rho, alpha, beta = polar
        return torch.tensor([[self.Kp[0] * rho,  0, 0, 0],
                         [0, self.Kp[1] * torch.sin(alpha), 0, - self.Kp[3] * torch.sin(beta - alpha)],
                         [0,  0, self.Kp[2] * torch.sin(beta), self.Kp[3] * torch.sin(beta - alpha)]])

    def isconverged(self, x, state_goal):
        rho, alpha, beta = cartesian2polar(x, state_goal)
        return rho < 1e-3


def numerical_jac(func, x0_in, eps, dtype=torch.float64):
    """
    >>> def func(x): return torch.tensor([torch.cos(x[0]), torch.sin(x[1])])
    >>> def jacfunc(x): return torch.tensor([[-torch.sin(x[0]), 0], [0, torch.cos(x[1])]])
    >>> x0 = torch.rand(2)
    >>> njac = numerical_jac(func, x0, 1e-6)
    >>> ajac = jacfunc(x0)
    >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
    """
    x0 = torch_to(x0_in, dtype=dtype)
    f0 = func(x0)
    m = f0.shape[-1] if len(f0.shape) else 1
    jac = torch.empty((m, x0.shape[-1]), dtype=x0.dtype)
    Dx = eps * torch.eye(x0.shape[-1], dtype=x0.dtype)
    XpDx = x0 + Dx
    for c in range(x0.shape[-1]):
        jac[:, c:c+1] = (func(XpDx[c, :]).reshape(-1, 1) - f0.reshape(-1, 1)) / eps

    return torch_to(jac, dtype=x0_in.dtype,
                    device=getattr(x0_in, 'device', None))


class CLFCartesian:
    def __init__(self,
                 Kp = torch.tensor([9., 15., 40.])/10.):
        self.Kp = Kp

    def clf_terms(self, state, state_goal):
        rho, alpha, beta = cartesian2polar(state, state_goal)
        x,y, theta = state
        x_goal, y_goal, theta_goal = state_goal
        return torch.tensor((0.5 * self.Kp[0] * rho ** 2,
                         self.Kp[1] * (1-torch.cos(alpha)),
                         self.Kp[2] * (1-torch.cos(beta))
        ))

    def _grad_clf_terms_wrt_goal(self, state, state_goal):
        """
        >>> self = CLFCartesian()
        >>> x0 = torch.rand(3)
        >>> x0_goal = torch.rand(3)
        >>> ajac = self._grad_clf_terms_wrt_goal(x0, x0_goal)[:, 0]
        >>> njac = numerical_jac(lambda xg: self.clf_terms(x0, xg)[0], x0_goal, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        >>> ajac = self._grad_clf_terms_wrt_goal(x0, x0_goal)[:, 1]
        >>> njac = numerical_jac(lambda xg: self.clf_terms(x0, xg)[1], x0_goal, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        >>> ajac = self._grad_clf_terms_wrt_goal(x0, x0_goal)[:, 2]
        >>> njac = numerical_jac(lambda xg: self.clf_terms(x0, xg)[2], x0_goal, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        x_diff, y_diff, theta_diff = state_goal - state
        rho, alpha, beta = cartesian2polar(state, state_goal)
        return torch.tensor([[self.Kp[0] * x_diff,
                          self.Kp[1] * torch.sin(alpha) * y_diff / (rho**2),
                          self.Kp[2] * torch.sin(beta) * y_diff / (rho**2)
                          ],
                         [self.Kp[0] * y_diff,
                          - self.Kp[1] * torch.sin(alpha) * x_diff / (rho**2),
                          - self.Kp[2] * torch.sin(beta) * x_diff / (rho**2)],
                         [0, 0,
                          self.Kp[2] * torch.sin(beta)]
                         ])

    def _grad_clf_terms(self, state, state_goal):
        """
        >>> self = CLFCartesian()
        >>> x0 = torch.rand(3)
        >>> x0_goal = torch.rand(3)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 0]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x,x0_goal)[0], x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 1]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x,x0_goal)[1], x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 2]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x,x0_goal)[2], x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        x_diff, y_diff, theta_diff = state_goal - state
        rho, alpha, beta = cartesian2polar(state, state_goal)
        return torch.tensor([[- self.Kp[0] * x_diff,
                          - self.Kp[1] * torch.sin(alpha) * y_diff / (rho**2),
                          - self.Kp[2] * torch.sin(beta) * y_diff / (rho**2)
                          ],
                         [- self.Kp[0] * y_diff,
                          self.Kp[1] * torch.sin(alpha) * x_diff / (rho**2),
                          self.Kp[2] * torch.sin(beta) * x_diff / (rho**2)],
                         [0, self.Kp[1] * torch.sin(alpha), 0]
                         ])
    def grad_clf(self, state, state_goal):
        """
        >>> self = CLFCartesian()
        >>> x0 = torch.rand(3)
        >>> x0_goal = torch.rand(3)
        >>> ajac = self.grad_clf(x0, x0_goal)
        >>> njac = numerical_jac(lambda x: self.clf_terms(x, x0_goal).sum(), x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        return self._grad_clf_terms(state, state_goal).sum(axis=-1)

    def grad_clf_wrt_goal(self, state, state_goal):
        """
        >>> self = CLFCartesian()
        >>> x0 = torch.rand(3)
        >>> x0_goal = torch.rand(3)
        >>> ajac = self.grad_clf_wrt_goal(x0, x0_goal)
        >>> njac = numerical_jac(lambda xg: self.clf_terms(x0, xg).sum(), x0_goal, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        return self._grad_clf_terms_wrt_goal(state, state_goal).sum(axis=-1)


    def isconverged(self, x, state_goal):
        rho, alpha, beta = cartesian2polar(x, state_goal)
        return rho < 1e-3


class ObstacleCBF:
    def __init__(self, center, radius, term_weights=[0.5, 0.5]):
        self.center = center
        self.radius = radius
        self.term_weights = term_weights

    def _cbf_radial(self, state):
        return ((state[:2] - self.center)**2).sum() - self.radius**2

    def _cbf_heading(self, state):
        good_heading = state[:2] - self.center
        good_heading_norm = good_heading / torch.norm(good_heading)
        return torch.cos(state[2]) * good_heading_norm[0] + torch.sin(state[2]) * good_heading_norm[1]

    def _cbf_terms(self, state):
        return [self._cbf_radial(state), self._cbf_heading(state)]

    def cbf(self, state):
        self.center, self.radius = map(partial(torch_to,
                                               device=state.device,
                                               dtype=state.dtype),
                                       (self.center, self.radius))
        return sum(w * t for w, t in zip(self.term_weights, self._cbf_terms(state)))

    def _grad_cbf_radial(self, state):
        """
        >>> self = ObstacleCBF(torch.rand(2), torch.rand(1))
        >>> x0 = torch.rand(3)
        >>> ajac = self._grad_cbf_radial(x0)
        >>> njac = numerical_jac(lambda x: self._cbf_radial(x), x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        gcbf = torch.zeros_like(state)
        gcbf[:2] = 2*(state[:2]-self.center)
        return gcbf

    def _grad_cbf_heading(self, state):
        """
        α = atan2(y, x)

        [∂ cos(α-θ),     ∂ cos(α - θ)  ∂ cos(θ - α) ]
        [------------,   ------------, -----------  ]
        [∂ x             ∂ y           ∂ θ          ]
        >>> self = ObstacleCBF(torch.rand(2), torch.rand(1))
        >>> x0 = torch.rand(3)
        >>> ajac = self._grad_cbf_heading(x0)
        >>> njac = numerical_jac(lambda x: self._cbf_heading(x), x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        gcbf = torch.zeros_like(state)
        θ = state[2]
        good_heading = state[:2] - self.center
        ρ = torch.norm(good_heading)
        α = torch.atan2(good_heading[1], good_heading[0])
        # ∂ / ∂ x cos(α-θ) = sin(α - θ) y / ρ²
        gcbf[0] = (α - θ).sin() * good_heading[1] / ρ**2
        # ∂ / ∂ y cos(α-θ) = - sin(α - θ) x / ρ²
        gcbf[1] = - (α - θ).sin() * good_heading[0] / ρ**2
        # ∂ / ∂ θ cos(θ-α) = - sin(θ - α)
        gcbf[2] = - (θ-α).sin()
        return gcbf

    def _grad_cbf_terms(self, state):
        return [self._grad_cbf_radial(state), self._grad_cbf_heading(state)]

    def grad_cbf(self, state):
        """
        >>> self = ObstacleCBF(torch.rand(2), torch.rand(1))
        >>> x0 = torch.rand(3)
        >>> ajac = self.grad_cbf(x0)
        >>> njac = numerical_jac(lambda x: self.cbf(x), x0, 1e-6)[0]
        >>> testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        self.center, self.radius = map(partial(torch_to,
                                               device=state.device,
                                               dtype=state.dtype),
                                       (self.center, self.radius))
        return sum(w * t for w, t in zip(self.term_weights,
                                         self._grad_cbf_terms(state)))


class ControllerCLF:
    """
    Aicardi, M., Casalino, G., Bicchi, A., & Balestrino, A. (1995). Closed loop steering of unicycle like vehicles via Lyapunov techniques. IEEE Robotics & Automation Magazine, 2(1), 27-35.
    """
    def __init__(self, # simulation parameters
                 planner,
                 u_dim = 2,
                 coordinate_converter = None, # cartesian2polar/ lambda x, xg: x
                 dynamics = None, # PolarDynamics()/CartesianDynamics()
                 clf = None, # CLFPolar()/CLFCartesian()
                 clf_gamma = 10,
                 clf_relax_weight = 10,
                 cbfs = [],
                 cbf_gammas = [],
                 max_risk=1e-2,
                 visualizer = None):
        self.planner = planner
        self.u_dim = 2
        self.coordinate_converter = coordinate_converter
        self.dynamics = dynamics
        self.clf = clf
        self.clf_gamma = 10
        self.clf_relax_weight = 10
        self.cbfs = cbfs
        self.cbf_gammas = cbf_gammas
        self.max_risk = max_risk
        self.visualizer = visualizer

    @property
    def model(self):
        return self.dynamics

    def _clc(self, x, state_goal, t):
        polar = self.coordinate_converter(x, state_goal)
        fx = self.dynamics.f_func(polar)
        gx = self.dynamics.g_func(polar)
        gclf = self.clf.grad_clf(polar, state_goal)
        gclf_goal = self.clf.grad_clf_wrt_goal(polar, state_goal)
        TBLOG.add_scalar("x_0", x[0], t)
        bfa = to_numpy(gclf @ gx)
        b = to_numpy(gclf @ fx
                     + gclf_goal @ self.planner.dot_plan(t)
                     + self.clf_gamma * self.clf.clf_terms(polar, state_goal).sum())
        return bfa, b

    def _cbcs(self, x_in, state_goal, t):
        x = self.coordinate_converter(x_in, state_goal)
        fx = self.dynamics.f_func(x)
        gx = self.dynamics.g_func(x)
        for cbf, cbf_gamma in zip(self.cbfs, self.cbf_gammas):
            gcbf = cbf.grad_cbf(x)
            cbfx = cbf.cbf(x)
            TBLOG.add_scalar("cbf", cbfx, t)
            yield to_numpy(gcbf @ gx), to_numpy(gcbf @ fx + cbf_gamma * cbfx)

    def _ctrl_ref(self, x, u):
        return np.zeros_like(x)

    def control(self, x_torch, t):
        state_goal = self.planner.plan(t)
        import cvxpy as cp # pip install cvxpy
        x = x_torch
        uvar = cp.Variable(self.u_dim)
        uvar.value = np.zeros(self.u_dim)
        relax = cp.Variable(1)
        obj = cp.Minimize(cp.sum_squares(uvar) + self.clf_relax_weight * relax)
        clc_bfa, clc_b = self._clc(x, state_goal, t)
        constraints = [
            uvar >= np.array([-10., -np.pi*5]),
            uvar <= np.array([10., np.pi*5]),
            clc_bfa @ uvar + clc_b - relax <= 0]
        for cbc_bfa, cbc_b in self._cbcs(x, state_goal, t):
            constraints.append(cbc_bfa @ uvar + cbc_b >= 0)

        problem = cp.Problem(obj, constraints)
        problem.solve(solver='GUROBI')
        if problem.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            # print("Optimal value: %s" % problem.value)
            pass
        else:
            raise ValueError(problem.status)
        # for variable in problem.variables():
        #     print("Variable %s: value %s" % (variable.name(), variable.value))
        uopt =  torch_to(torch.from_numpy(uvar.value),
                        device=getattr(x_torch, 'device', None),
                        dtype=x_torch.dtype)
        if hasattr(self.dynamics, 'train'):
            self.dynamics.train(x_torch, uopt)
        return uopt

    def isconverged(self, state, state_goal):
        return self.clf.isconverged(state, state_goal)


class ZeroDynamicsBayesian(ZeroDynamicsModel):
    def fu_func_gp(self, U):
        return GaussianProcess( shape = (self.state_size,),
                                mean = lambda x: self.f_func(x) + self.g_func(x) @ U,
                                knl = lambda x, xp: (U @ U + 1) * torch.eye(self.state_size) )


class ControllerCLFBayesian:
    def __init__(self, # simulation parameters
                 planner,
                 u_dim = 2,
                 coordinate_converter = None, # cartesian2polar/ lambda x, xg: x
                 dynamics = None, # PolarDynamics()/CartesianDynamics()
                 clf = None, # CLFPolar()/CLFCartesian()
                 clf_gamma = 10.,
                 cost_weights = [0.33, 0.33, 0.33],
                 cbfs = [],
                 cbf_gammas = [],
                 ctrl_min = [-10., -np.pi*5],
                 ctrl_max = [10., np.pi*5],
                 ctrl_ref = [0., 0.],
                 max_risk = 1e-2,
                 visualizer = None):
        self.u_dim = 2
        self.planner = planner
        self.coordinate_converter = coordinate_converter
        self.dynamics = dynamics
        self.clf = clf
        self.clf_gamma = clf_gamma
        self.cost_weights = cost_weights
        self.cbfs = cbfs
        self.cbf_gammas = cbf_gammas
        self.ctrl_min = np.array(ctrl_min)
        self.ctrl_max = np.array(ctrl_max)
        self.ctrl_ref = np.array(ctrl_ref)
        self.max_risk = max_risk
        self.visualizer = visualizer

    @property
    def model(self):
        return self.dynamics

    @staticmethod
    def convert_cbc_terms_to_socp_terms(bfe, e, V, bfv, v, extravars,
                                        testing=True):
        assert (V.eig()[0][:, 0].abs() > 0).all()
        assert v.abs() > 0
        m = bfe.shape[-1]
        bfc = bfe.new_zeros((m+extravars))
        if testing:
            u0 = torch.zeros((m,))
            u0_hom = torch.cat((torch.tensor([1.]), u0))
        with torch.no_grad():
            # [1, u] Asq [1; u]
            Asq = torch.cat(
                (
                    torch.cat((torch.tensor([[v]]),         (bfv / 2).reshape(1, -1)), dim=-1),
                    torch.cat(((bfv / 2).reshape(-1, 1),    V), dim=-1)
                ),
                dim=-2)
            if testing:
                np.testing.assert_allclose(u0_hom @ Asq @ u0_hom,
                                        u0 @ V @ u0 + bfv @ u0 + v)

            # [1, u] Asq [1; u] = |L[1; u]|_2 = |[0, A] [y_1; y_2; u] + b|_2
            n_constraints = m+1
            L = torch.cholesky(Asq) # (m+1) x (m+1)

        if testing:
            np.testing.assert_allclose(L @ L.T, Asq, rtol=1e-2, atol=1e-3)

        A = torch.zeros((n_constraints, m + extravars))
        A[:, extravars:] = L.T[:, 1:]
        bfb = L.T[:, 0] # (m+1)
        if testing:
            y_u0 = torch.cat((torch.zeros(extravars), u0))
            np.testing.assert_allclose(A @ y_u0 + bfb, L.T @ u0_hom)
            np.testing.assert_allclose(u0_hom @ Asq @ u0_hom, u0_hom @ L @ L.T @ u0_hom, rtol=1e-2, atol=1e-3)
            np.testing.assert_allclose(u0 @ V @ u0 + bfv @ u0 + v, (A @ y_u0 + bfb) @ (A @ y_u0 + bfb), rtol=1e-2, atol=1e-3)
        if extravars >= 1: # "I assumed atleast δ "
            bfc[extravars-1] = 1 # For delta the relaxation factor
        bfc[extravars:] = bfe # only affine terms need to be negative?
        d = e
        return A, bfb, bfc, d

    def _clc(self, state, state_goal, u, t):
        n = state.shape[-1]
        clfgp = DeterministicGP(lambda x: self.clf_gamma * self.clf.clf_terms(x, state_goal).sum(), shape=(1,))
        gclfgp = DeterministicGP(lambda x: self.clf.grad_clf(x, state_goal), shape=(n,))
        gclf_goal = DeterministicGP(lambda x: self.clf.grad_clf_wrt_goal(x, state_goal),
                                    shape=(n,))
        fu_func_gp = self.dynamics.fu_func_gp(u)
        dot_plan = DeterministicGP(lambda x: self.planner.dot_plan(t), shape=(n,))
        return gclfgp.t() @ fu_func_gp + gclf_goal.t() @ dot_plan + clfgp

    def _clc_terms(self, state, state_goal, t):
        m = self.u_dim
        (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
            lambda u: self._clc(state, state_goal, u, t) * -1.0,
            state, torch.rand(m))
        assert (V.eig()[0][:, 0].abs() > 0).all()
        assert v.abs() > 0
        A, bfb, bfc, d = self.convert_cbc_terms_to_socp_terms(
            bfe, e, V, bfv, v, 0)
        return map(to_numpy, (A, bfb, bfc, d))

    def _cbc(self, cbf, cbf_gamma, state, u, t):
        n = state.shape[-1]
        cbfx = DeterministicGP(lambda x: cbf_gamma * cbf.cbf(x), shape=(1,))
        gcbfx = DeterministicGP(lambda x: cbf.grad_cbf(x), shape=(n,))
        fu_func_gp = self.dynamics.fu_func_gp(u)
        return gcbfx.t() @ fu_func_gp + cbfx


    def _cbc_terms(self, cbf, cbf_gamma, state, t):
        m = self.u_dim
        (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
            lambda u: self._cbc(cbf, cbf_gamma, state, u, t),
            state, torch.rand(m))
        A, bfb, bfc, d = self.convert_cbc_terms_to_socp_terms(
            bfe, e, V, bfv, v, 0)
        return map(to_numpy, (A, bfb, bfc, d))

    def _cbcs(self, state, t):
        for cbf, cbf_gamma in zip(self.cbfs, self.cbf_gammas):
            yield self._cbc_terms(cbf, cbf_gamma, state, t)

    def _factor(self):
        assert self.max_risk <= 0.5 and self.max_risk >= 0
        return math.sqrt(2)*erfinv(1-2*self.max_risk)

    def control(self, x_torch, t):
        import cvxpy as cp # pip install cvxpy
        state_goal = self.planner.plan(t)
        x = x_torch
        uvar = cp.Variable(self.u_dim)
        uvar.value = np.zeros(self.u_dim)
        relax = cp.Variable(1)
        u_ref = np.array(self.ctrl_ref)
        cost_terms = [cp.sum_squares(uvar[0] - u_ref[0]),
                      cp.sum_squares(uvar[1] - u_ref[1]),
                      relax**2]

        assert len(self.cost_weights) == len(cost_terms), 'assumed same cost terms'
        obj = cp.Minimize(sum([w * t for w, t in zip(self.cost_weights, cost_terms)]))
        clc_A, clc_bfb, clc_bfc, clc_d = self._clc_terms(x, state_goal, t)
        rho = self._factor()
        constraints = [
            clc_bfc @ uvar + clc_d + relax >= rho * cp.norm(clc_A @ uvar + clc_bfb)]

        for cbc_A, cbc_bfb, cbc_bfc, cbc_d in self._cbcs(x, t):
            constraints.append(
                cbc_bfc @ uvar + cbc_d >= rho * cp.norm(cbc_A @ uvar + cbc_bfb)
            )

        # print("det(V(x,x)): ", np.linalg.det(cbc_A.T @ cbc_A))

        problem = cp.Problem(obj, constraints)
        problem.solve(solver='GUROBI', verbose=False)
        if problem.status not in ["infeasible", "unbounded", "infeasible_inaccurate"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            TBLOG.add_scalar("opt/rho", rho, t)
            TBLOG.add_scalar("opt/value", problem.value, t)
            TBLOG.add_scalar("opt/cost_vel", self.cost_weights[0] *
                             (uvar.value[0] - u_ref[0])**2, t)
            TBLOG.add_scalar("opt/cost_relax", (self.cost_weights[2] * relax.value**2), t)
            TBLOG.add_scalar("opt/cbc_norm", np.linalg.norm(cbc_A @ uvar.value + cbc_bfb), t)
        else:
            raise ValueError(problem.status)
        # for variable in problem.variables():
        #     print("Variable %s: value %s" % (variable.name(), variable.value))
        uopt = torch_to(torch.from_numpy(uvar.value),
                        device=getattr(x_torch, 'device', None),
                        dtype=x_torch.dtype)
        if self.visualizer is not None:
            fu_gp = self.dynamics.fu_func_gp(uopt)
            self.visualizer.add_info(t, 'xtp1',
                                     x_torch + fu_gp.mean(x_torch) * self.planner.dt)
            self.visualizer.add_info(t, 'Fxu_var', fu_gp.knl(x_torch, x_torch))
            # meanFX.shape = (b(1+m)n), varFX.shape = (b(1+m)n, b(1+m)n)
            meanFX, varFX = self.dynamics.custom_predict_fullmat(x_torch)
            self.visualizer.add_info(t, 'Fx_var', varFX) # B otimes A
            cbc_sigma = cbc_A @ uvar.value + cbc_bfb
            cbc_var = torch_to(torch.from_numpy(
                cbc_sigma.reshape(-1, 1) @ cbc_sigma.reshape(1, -1)),
                                device=x_torch.device,
                                dtype=x_torch.dtype)
            self.visualizer.add_info(t, 'cbcs', self._cbcs)
            self.visualizer.add_info(t, 'rho', rho)
            self.visualizer.add_info(t, 'cbc_var', rho**2 * cbc_var)
            TBLOG.add_scalar("opt/cbc_var", torch.det(cbc_var), t)
        if hasattr(self.dynamics, 'train'):
            self.dynamics.train(x_torch, uopt)
        return uopt

    def isconverged(self, state, state_goal):
        return self.clf.isconverged(state, state_goal)


class ControllerPID:
    def __init__(self,
                 planner,
                 # simulation parameters
                 Kp_rho = 9,
                 Kp_alpha = -15,
                 Kp_beta = -3):
        self.planner = planner
        self.Kp_rho = Kp_rho
        self.Kp_alpha = Kp_alpha
        self.Kp_beta = Kp_beta

    def control(self, x, t):
        state_goal = self.planner.plan(t)
        rho, alpha, beta = cartesian2polar(x, state_goal)
        Kp_rho   = self.Kp_rho
        Kp_alpha = self.Kp_alpha
        Kp_beta  = self.Kp_beta
        v = Kp_rho * rho
        w = Kp_alpha * alpha + Kp_beta * beta
        if alpha > math.pi / 2 or alpha < -math.pi / 2:
            v = -v
        return torch.cat([v.unsqueeze(-1), w.unsqueeze(-1)])

    def isconverged(self, x, state_goal):
        rho, alpha, beta = cartesian2polar(x, state_goal)
        return rho < 1e-3


def add_scalars(tag, var_dict, t):
    for k, v in var_dict.items():
        TBLOG.add_scalar("/".join((tag, k)), v, t)


class VisualizerScalarPlotCtrl:
    def __init__(self):
        self.u_traj = []

    def setStateCtrl(self, ax, info, state, uopt, t=None, **kw):
        self.u_traj.append(to_numpy(uopt))
        self._plot_ctrl(ax, [abs(float(uopt[0])) for uopt in self.u_traj])

    def _plot_ctrl(self, ax, u_traj):
        #ax.set_yscale('log')
        ax.plot(u_traj)
        ax.set_ylabel(r'$|v|$')

class VisualizerScalarPlotCBC:
    def __init__(self):
        self.cbc_traj = []

    def setStateCtrl(self, ax, info, state, uopt, t=None, **kw):
        cbcs = info[t]['cbcs']
        rho = info[t]['rho']
        uopt_np = to_numpy(uopt)
        self.cbc_traj.append(
            min([c @ uopt_np + d - rho * np.linalg.norm(A @ uopt_np  + b)
                        for A, b, c, d in cbcs(state, t)]))
        self._plot_cbc(ax, self.cbc_traj)

    def _plot_cbc(self, ax, cbc_traj):
        ax.plot(cbc_traj)
        ax.set_ylabel(r'SCBC')
        ax.set_xlabel(r'timesteps')


class VisualizerScalarPlotTraceKnlNoCtrl:
    def __init__(self):
        self.trace_knl_traj = []

    def setStateCtrl(self, ax, info, state, uopt, t=None, **kw):
        Fx_var = info[t]['Fx_var']
        Fx_var_np = (to_numpy(Fx_var)
                      if isinstance(Fx_var, torch.Tensor)
                      else Fx_var)
        self.trace_knl_traj.append(np.trace(Fx_var_np))
        self._plot_det_knl(ax, self.trace_knl_traj)

    def _plot_det_knl(self, ax, trace_knl_traj):
        ax.plot(trace_knl_traj)
        ax.set_ylabel(r'$\mathbf{tr}(\mathbf{B}_k(\mathbf{x},\mathbf{x}) \otimes \mathbf{A})$')
        #ax.set_xlabel(r'timesteps')
        ax.set_yscale('log')


class Visualizer:
    SCALAR_PLOTS_GEN=dict(Ctrl=VisualizerScalarPlotCtrl,
                          TraceKnl=VisualizerScalarPlotTraceKnlNoCtrl,
                          DetKnl=VisualizerScalarPlotTraceKnlNoCtrl,
                          CBC=VisualizerScalarPlotCBC)
    def __init__(self, planner, dt, cbfs=[], compute_contour_every_n_steps=20,
                 scalar_plots = ['Ctrl', 'CBC'],
                 suptitle=None,
                 subplots_adjust_kw=dict(top=0.95),
                 figsize=(4, 8),
                 animationframepathpatt='data/animation/frame%04d.png',
                 plots_ylims=dict()):
        self.planner = planner
        self.dt = dt
        self.cbfs = cbfs
        self.compute_contour_every_n_steps = compute_contour_every_n_steps
        self._scalar_plots = scalar_plots
        self.suptitle = suptitle
        self.subplots_adjust_kw = subplots_adjust_kw
        self.animationframepathpatt = animationframepathpatt
        self.scalar_plots_ylims = plots_ylims


        self.scalar_plots = [self.SCALAR_PLOTS_GEN[s]() for s in scalar_plots]
        self.state_start = None
        self.x_traj, self.y_traj = [], []
        self.state_traj = []
        self.fig = plt.figure(figsize=figsize)
        self.info = dict()
        self.axes = self._subplots()

    def _subplots(self):
        h = 0.4 / len(self.scalar_plots)
        axs = self.fig.subplots(1+len(self.scalar_plots),1, gridspec_kw=dict(
            height_ratios=[0.6] + [h]*len(self.scalar_plots)))
        if self.suptitle:
            self.fig.suptitle(self.suptitle)
        self.fig.subplots_adjust(**self.subplots_adjust_kw)
        for i in reversed(range(len(self.scalar_plots)-1)):
            axs[i+1].sharex(axs[i+2])
        return axs

    def add_info(self, t, key, value):
        self.info.setdefault(t, dict())[key] = value

    @staticmethod
    def _plot_static(ax, state_goal, state_start, cbfs):
        scale = (state_goal[:2] - state_start[:2]).norm() / 10.
        # x_start, y_start, theta_start = state_start
        # ax.arrow(x_start, y_start, torch.cos(theta_start) * scale,
        #             torch.sin(theta_start)*scale, color='c', width=0.1*scale,
        #          label='Start')
        x_goal, y_goal, theta_goal = state_goal
        ax.plot(x_goal, y_goal, 'g+', linewidth=0.4, label='Goal')
        ax.add_patch(Circle(np.array([x_goal, y_goal]),
                            radius=scale/5, fill=False, color='g'))

        labelled_once = False
        for cbf in cbfs:
            circle = Circle(to_numpy(cbf.center),
                            radius=to_numpy(cbf.radius),
                            fill=True, color='r',
                            label=(None if labelled_once else 'Obstacles'))
            ax.add_patch(circle)
            labelled_once = True

    @staticmethod
    def _compute_contour_grid(grid, rho, cbcs, state, uopt, t, npts):
        cbc_value_grid = np.zeros(grid.shape[1:])
        for ridx in range(grid.shape[1]):
            for cidx in range(grid.shape[2]):
                x, y = grid[:, ridx, cidx]

                state_copy = state.clone()
                state_copy[0] = x
                state_copy[1] = y
                cbc_value_grid[ridx, cidx] = min(
                    [c @ uopt + d - rho * np.linalg.norm(A @ uopt  + b)
                      for A, b, c, d in cbcs(state_copy, t)])
        return cbc_value_grid

    def _plot_cbf_contour(self, rho, cbcs, state, state_start,
                          state_goal, uopt, t, npts=15):
        ax = self.axes[0]
        xmin, xmax, ymin, ymax = self._get_bbox(state_start, state_goal)
        xstep = (xmax - xmin) / (npts-1)
        ystep = (ymax - ymin) / (npts-1)
        grid = np.mgrid[ymin:(ymax+ystep):ystep, xmin:(xmax+xstep):xstep]
        ccens = self.compute_contour_every_n_steps
        t_cache = t - (t % ccens)
        if t % ccens == 0 and 'cbc_value_grid' not in self.info.get(t_cache, {}):
            self.info[t]['cbc_value_grid'] = self._compute_contour_grid(
                grid, rho, cbcs, state, uopt, t, npts)

        CS = ax.contour(grid[0], grid[1], self.info[t_cache]['cbc_value_grid'])
        ax.clabel(CS, CS.levels, inline=True, fmt='%r', fontsize=8)
        # cbar = ax.figure.colorbar(CS)
        # cbar.ax.set_ylabel('CBC value')

    @staticmethod
    def _get_bbox(state_start, state_goal):
        state_min = np.minimum(state_start[:2], state_goal[:2]).min()
        state_max = np.maximum(state_start[:2], state_goal[:2]).max()
        state_dist = state_max - state_min
        ymin = xmin = state_min - 0.1 * state_dist
        ymax = xmax = state_max + 0.1 * state_dist
        return [xmin, xmax, ymin, ymax]

    def setStateCtrl(self, state, uopt, t=None, **kw):
        if t == 0:
            self.state_start = state.clone()
        self.fig.clf()
        self.axes = self._subplots()
        ax = self.axes[0]
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        xmin, xmax, ymin, ymax = self._get_bbox(to_numpy(self.state_start),
                                                to_numpy(self.planner.x_goal))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        state_goal = self.planner.x_goal

        # Plot CBF contour
        if 'cbcs' in self.info.get(t, {}):
            cbcs = self.info[t]['cbcs']
            rho = self.info[t]['rho']
            uopt_np = to_numpy(uopt)
            self._plot_cbf_contour(rho, cbcs, state, to_numpy(self.state_start),
                                   to_numpy(state_goal), uopt_np, t)
        self._plot_static(ax, self.planner.x_goal, self.state_start, self.cbfs)
        xy_plan_traj = [self.planner.plan(ti)[:2] for ti in range(0, t, 10)]
        if len(xy_plan_traj):
            x_plan_traj, y_plan_traj = zip(*xy_plan_traj)
            ax.plot(x_plan_traj, y_plan_traj, 'g+', linewidth=0.01)
        x, y, theta = state
        self.x_traj.append(x)
        self.y_traj.append(y)
        self.state_traj.append(state)
        for s in self.state_traj:
            if any(cbf.cbf(s) < 0 for cbf in self.cbfs):
                ax.text(s[0], s[1], "Crash", bbox=dict(facecolor='red', fill=True, alpha=0.1))
                break
        plot_vehicle(ax, x, y, theta, uopt, self.x_traj, self.y_traj,
                     to_numpy(self.state_start), to_numpy(self.planner.x_goal))

        # Plot uncertainty ellipse
        if False and 'xtp1' in self.info.get(t, {}) and 'xtp1_var' in self.info.get(t, {}):
            xtp1 = self.info[t]['xtp1']
            xtp1_var = self.info[t]['xtp1_var']
            pos = to_numpy(xtp1[:2])
            pos_var = to_numpy(xtp1_var[:2, :2])
            scale, theta = var_to_scale_theta(pos_var)
            if (scale > 1e-6).all():
                draw_ellipse(ax, scale * 1e4, theta, pos)

        for ax, spname, sp in zip(self.axes[1:], self._scalar_plots,
                                  self.scalar_plots):
            sp.setStateCtrl(ax, self.info, state, uopt, t=t, **kw)
            if spname in self.scalar_plots_ylims:
                print("setting ylim", spname, self.scalar_plots_ylims[spname])
                ax.set_ylim(*self.scalar_plots_ylims[spname])
        mkdir_savefig(self.fig, self.animationframepathpatt % t)
        plt.pause(self.dt)

class Logger:
    def __init__(self, planner, dt, cbfs=[], compute_contour_every_n_steps=20,
                 scalar_plots=['Ctrl', 'CBC'],
                 suptitle='',
                 subplots_adjust_kw=dict(top=0.95)):
        self.planner = planner
        self.dt = dt
        self.cbfs = cbfs
        self.suptitle = suptitle
        self.subplots_adjust_kw = subplots_adjust_kw

        self.state_start = None
        self.info = dict()
        self.compute_contour_every_n_steps = compute_contour_every_n_steps

    def _log_cbf_contour(self, rho, cbcs, state, state_start, state_goal, uopt, t, npts=15):
        xmin, xmax, ymin, ymax = map(float, Visualizer._get_bbox(state_start, state_goal))
        xstep = (xmax - xmin) / (npts-1)
        ystep = (ymax - ymin) / (npts-1)
        grid = np.mgrid[ymin:(ymax+ystep):ystep, xmin:(xmax+xstep):xstep]
        assert grid.shape[1] == npts and grid.shape[2] == npts
        if t % self.compute_contour_every_n_steps == 0:
            cbc_value_grid = Visualizer._compute_contour_grid(
                grid, rho, cbcs, state, uopt, t, npts)
            add_tensors(TBLOG, "vis", dict(grid=grid,
                                    cbc_value_grid=cbc_value_grid),
                        t)

    def add_info(self, t, key, value):
        self.info.setdefault(t, dict())[key] = value

    def setStateCtrl(self, state, uopt, t=None, **kw):
        if t == 0:
            self.state_start = state.clone()
        plan_x = self.planner.plan(t)

        add_tensors(TBLOG, "vis", dict(state=to_numpy(state), uopt=to_numpy(uopt),
                                plan_x=to_numpy(plan_x)),
                    t)
        if 'cbcs' in self.info.get(t, {}):
            cbcs = self.info[t]['cbcs']
            rho = self.info[t]['rho']
            uopt_np = to_numpy(uopt)
            self._log_cbf_contour(rho, cbcs, state, self.state_start,
                                  self.planner.x_goal, uopt_np, t)
            add_tensors(TBLOG, "vis",
                        dict(cbc_value=min(
                            [c @ uopt_np + d - rho * np.linalg.norm(A @ uopt_np  + b)
                             for A, b, c, d in cbcs(state, t)])),
                        t)
            add_scalars("vis", dict(rho=rho), t)
        info_np = { k : (to_numpy(v) if isinstance(v, torch.Tensor) else v)
                    for k, v in self.info[t].items()
                    if isinstance(v, (torch.Tensor, float, int, np.ndarray))}
        add_tensors(TBLOG, "vis", info_np, t)

    @classmethod
    def _reconstruct_cbcs(cls, state, uopt, cache):
        def cbcs(state, t):
            m = uopt.shape[0]
            A = np.zeros((0, m))
            b = np.zeros((0,))
            c = np.zeros((m,))
            try:
                d = cache['vis/cbc_value']
            except KeyError:
                print(cache)
                raise
            return [(A, b, c, d)]
        return cbcs

    @classmethod
    def _make_info(cls, state, uopt, cache):
        info = dict()
        info['rho'] = cache['vis/rho']
        if 'vis/cbc_value_grid' in cache:
            info['cbc_value_grid'] = cache['vis/cbc_value_grid']
        info['cbcs'] = cls._reconstruct_cbcs(state, uopt, cache)
        for k, v in cache.items():
            if k.startswith("vis/"):
                info[k[len("vis/"):]] = v
        return info


    def load_visualizer(self, event_file):
        cache = dict()
        running_t = 0
        for t, tag, value in stream_tensorboard_scalars(event_file):
            if t > running_t:
                state = torch.tensor(cache[running_t].pop('vis/state'))
                uopt = torch.tensor(cache[running_t].pop('vis/uopt'))
                info = self._make_info(state, uopt, cache[running_t])
                del cache[running_t]
                yield running_t, state, uopt, info
            cache.setdefault(t, dict())[tag] = value
            running_t = t


def visualize_tensorboard_logs(events_dir, ax = None, traj_marker='b--', label=None):
    events_file = glob.glob(osp.join(events_dir, "*.tfevents*"))[0]
    config = json.load(open(osp.join(events_dir, CONFIG_FILE_BASENAME)))
    grouped_by_tag = load_tensorboard_scalars(events_file)
    x_traj = np.array(list(zip(*grouped_by_tag['vis/x0']))[1])
    y_traj = np.array(list(zip(*grouped_by_tag['vis/x1']))[1])
    theta_traj = np.array(list(zip(*grouped_by_tag['vis/x2']))[1])
    u0_traj = np.array(list(zip(*grouped_by_tag['vis/u0']))[1])
    u1_traj = np.array(list(zip(*grouped_by_tag['vis/u1']))[1])
    x = x_traj[-1]
    y = y_traj[-1]
    theta = theta_traj[-1]
    state_start = np.array(config['state_start'])
    x_goal = np.array(config['state_goal'])
    if ax is None:
        fig, ax = plt.subplots(1,1)

        cbfs = obstacles_at_mid_from_start_and_goal(torch.from_numpy(state_start), torch.from_numpy(x_goal))
        Visualizer._plot_static(ax, torch.from_numpy(x_goal),
                                torch.from_numpy(state_start), cbfs)
    uopt = torch.tensor([u0_traj[-1], u1_traj[-1]])
    plot_vehicle(ax, x, y, theta, uopt, x_traj, y_traj,
                 state_start, x_goal, traj_marker=traj_marker, label=label)
    return ax

def filter_log_files(
        runs_dir="data/runs",
        test_config=lambda c: ('state_goal' in c
                               and 'Logger' == c['visualizer_class']['__callable_name__']),
        topk=2):
    valid_files = []
    for f in sorted(glob.glob(osp.join(runs_dir, "*", CONFIG_FILE_BASENAME)),
                    key=osp.getmtime, reverse=True):
        try: config = json.load(open(f))
        except json.JSONDecodeError:
            continue
        if test_config(config):
            valid_files.append(f)
        if len(valid_files) >= topk:
            break

    return valid_files

def visualize_last_n_files(runs_dir="data/runs",
                           last_n=2,
                           file_filter=filter_log_files,
                           traj_markers=['b--', 'r--', 'k--', 'c--']):
    ax = None
    for config_file, traj_marker in zip(
            file_filter(runs_dir=runs_dir, topk=last_n),
            traj_markers):
        config = json.load(open(config_file))
        start_goal = tuple(config['state_start'] + config['state_goal'])
        run_dir = osp.dirname(config_file)
        ax = visualize_tensorboard_logs(
            run_dir, ax = ax,
            traj_marker=traj_marker,
            label='{}; {}'.format(('Learning' if config['enable_learning'] else 'No Learning'),
                                  ('Bayes CBF' if config['controller_class']['__callable_name__'] == ControllerCLFBayesian.__name__ else 'Mean CBF')))
        ax.set_title('true L = %.02f' % config['true_dynamics_gen']['L']
                    + '; prior L = %.02f' % config['mean_dynamics_gen']['L'])
        ax.legend()
        mkdir_savefig(ax.figure, osp.join(run_dir, 'vis.pdf'))
    plt.close(ax.figure.number)


def playback_logfile(events_dir, visualizer_kw=dict()):
    config_file = osp.join(events_dir, CONFIG_FILE_BASENAME)
    config = json.load(open(config_file))
    run_dir = osp.dirname(config_file)
    state_start = torch.tensor(config['state_start'])
    state_goal = torch.tensor(config['state_goal'])
    numSteps = config['numSteps']
    dt = config['dt']
    planner = PiecewiseLinearPlanner(state_start, state_goal, numSteps, dt)
    cbfs_gen = globals()[config['cbfs'].pop('__callable_name__')]
    cbfs = cbfs_gen(state_start, state_goal, **config['cbfs'])
    logger = Logger(planner, dt, cbfs)
    _ = config['visualizer_class'].pop('__callable_name__')
    visualizer_kw_from_log = { k: v
                              for k, v in config['visualizer_class'].items()
                              if k in ("suptitle", "subplots_adjust_kw", "scalar_plots")}
    visualizer_kw_from_log.update(visualizer_kw)
    visualizer = Visualizer(planner, dt, cbfs,
                            **visualizer_kw_from_log)
    events_file = max(glob.glob(osp.join(events_dir, '*tfevents*')),
                      key=lambda f: os.stat(f).st_mtime)
    for t, state, uopt, info in logger.load_visualizer(events_file):
        for k, v in info.items():
            visualizer.add_info(t, k, v)
        visualizer.setStateCtrl(state, uopt, t)
    mkdir_savefig(visualizer.fig, osp.join(events_dir, 'final_visualization.pdf'))
    cmd = "ffmpeg -y -i {pngframes} -frames {frames} -pix_fmt yuv420p -c:v libx264 {outvideopath}".format(
        frames=numSteps-1,
        pngframes=visualizer.animationframepathpatt,
        outvideopath=osp.join(events_dir, 'animation.mp4')).split()
    print("Running ", " ".join(cmd))
    subprocess.run(cmd)



def move_to_pose(state_start, state_goal,
                 dt = 0.01,
                 show_animation = True,
                 controller=None,
                 dynamics=CartesianDynamics(),
                 visualizer=None
):
    """
    rho is the distance between the robot and the goal position
    alpha is the angle to the goal relative to the heading of the robot
    beta is the angle between the robot's position and the goal position plus the goal angle

    Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards the goal
    Kp_beta*beta rotates the line so that it is parallel to the goal angle
    """
    visualizer = (Visualizer(NoPlanner(state_goal), dt)
                  if visualizer is None else
                  visualizer)
    state = state_start.clone()
    count = 0
    dynamics.set_init_state(state)
    while not controller.isconverged(state, state_goal):
        x, y, theta = state

        # control
        ctrl = controller.control(state, t=count)

        # simulation
        state = dynamics.step(ctrl, dt)['x']

        if show_animation:  # pragma: no cover
            visualizer.setStateCtrl(state, ctrl, t=count)
        count = count + 1


def plot_vehicle(ax, x, y, theta, uopt, x_traj, y_traj, state_start, state_goal,
                 traj_marker='b--', **plot_kw): # pragma: no cover
    # Corners of triangular vehicle when pointing to the right (0 radians)
    scale = np.linalg.norm(state_goal[:2] - state_start[:2]) / 10.
    triangle = np.array([[0.0, 0],
                         [-1.0, 0.25],
                         [-1.0, -0.25]])

    tri = (rot_matrix(theta) @ (scale * triangle.T) + np.array([x, y]).reshape(-1, 1)).T

    ax.add_patch(Polygon(tri, fill=False, edgecolor='k'))

    ax.plot(x_traj, y_traj, traj_marker, **plot_kw)
    ax.arrow(x, y,
             torch.cos(uopt[1] + theta) * uopt[0] * scale,
             torch.sin(uopt[1] + theta) * uopt[0] * scale,
             color='c',
             width=0.1*scale, label='Control')

    # for stopping simulation with the esc key.
    # self.fig.canvas.mpl_connect('key_release_event',
    #         lambda event: [sys.exit(0) if event.key == 'escape' else None])


def rot_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


class NoPlanner:
    def __init__(self, x_goal):
        self.x_goal = x_goal

    def plan(self, t):
        return self.x_goal

    def dot_plan(self, t):
        return torch.zeros_like(self.x_goal)


def R90():
    return torch.tensor([[0., -1.],
                         [1, 0.]])


def getfuncname(func):
    # handle partials
    while not hasattr(func, "__name__") and hasattr(func, "func"):
        func = func.func
    return getattr(func, "__name__", "")

def extract_keywords(func):
    keywords = default_kw(func)
    for k, v in keywords.items():
        if callable(v):
            keywords[k] = extract_keywords(v)
            funcname = getfuncname(v)
            if funcname:
                keywords[k]["__callable_name__"] = funcname
    return keywords


def applyall(fs, *a, **kw):
    return  [f(*a, **kw) for f in fs]

####################################################################
# configured methods
####################################################################

def obstacles_at_mid_from_start_and_goal(x, x_g, term_weights=[0.5, 0.5]):
    return [ObstacleCBF((x[:2] + x_g[:2])/2
                        + R90() @ (x[:2] - x_g[:2])/3,
                        (x[:2] - x_g[:2]).norm()/4,
                        term_weights=term_weights),
            ObstacleCBF((x[:2] + x_g[:2])/2
                        - R90() @ (x[:2] - x_g[:2])/3,
                        (x[:2] - x_g[:2]).norm()/4,
                        term_weights=term_weights)]


def single_obstacle_at_mid_from_start_and_goal(x, x_g, term_weights=[0.5, 0.5]):
    return [ObstacleCBF((x[:2] + x_g[:2])/2 + R90() @ (x[:2] - x_g[:2]) / 10,
                        (x[:2] - x_g[:2]).norm()/8,
                        term_weights=term_weights)]


def move_to_pose_clf_polar(x, x_g, dt = None, **kw):
    return move_to_pose(
            x , x_g,
            dynamics=CartesianDynamics(),
            controller=ControllerCLF(
                NoPlanner(x_g),
                coordinate_converter = cartesian2polar,
                dynamics=PolarDynamics(),
                clf = CLFPolar()
            ),
            visualizer=Visualizer(NoPlanner(x_g), dt),
            dt = dt,
            **kw)

def move_to_pose_clf_cartesian(x, x_g, dt = None, **kw):
    return move_to_pose(x, x_g,
                        dynamics=CartesianDynamics(),
                        controller=ControllerCLF(
                            NoPlanner(x_g),
                            coordinate_converter = lambda x, x_g: (x),
                            dynamics=CartesianDynamics(),
                            clf = CLFCartesian()
                        ),
                        visualizer=Visualizer(NoPlanner(x_g), dt),
                        dt = dt,
                        **kw
    )


def move_to_pose_pid(x, x_g, dt = None, **kw):
    return move_to_pose(x, x_g,
                        dynamics=CartesianDynamics(),
                        controller=ControllerPID(NoPlanner(x_g)),
                        visualizer=Visualizer(NoPlanner(x_g), dt),
                        dt = dt,
                        **kw)

def move_to_pose_sample_clf_cartesian(x, x_g, dt = None, **kw):
    return sample_generator_trajectory(
        dynamics_model=CartesianDynamics(),
        D=200,
        controller=ControllerCLF(
            NoPlanner(x_g),
            coordinate_converter = lambda x, x_g: x,
            dynamics = CartesianDynamics(),
            clf = CLFCartesian()
        ).control,
        visualizer=Visualizer(NoPlanner(x_g), dt),
        x0=x,
        dt=dt)

def track_trajectory_clf_cartesian(x, x_g, dt = None,
                                   cbfs = None,
                                   cbf_gammas = None,
                                   numSteps = None,
                                   Kp = [0.9, 1.5, 0.],
                                   **kw):
    return sample_generator_trajectory(
        dynamics_model=CartesianDynamics(),
        D=numSteps,
        controller=ControllerCLF(
            PiecewiseLinearPlanner(x, x_g, numSteps, dt),
            coordinate_converter = lambda x, x_g: x,
            dynamics = CartesianDynamics(),
            clf = CLFCartesian(
                Kp = torch.tensor(Kp)
            ),
            cbfs = cbfs(x , x_g),
            cbf_gammas = cbf_gammas
        ).control,
        visualizer=Visualizer(
            PiecewiseLinearPlanner(x, x_g, numSteps, dt),
            dt,
            cbfs = cbfs(x, x_g)
        ),
        x0=x,
        dt=dt,
        **kw)


def track_trajectory_clf_bayesian(x, x_g, dt = None,
                                  cbfs = None,
                                  cbf_gammas = None,
                                  numSteps = None,
                                  Kp = [0.9, 1.5, 0.],
                                  **kw):
    return sample_generator_trajectory(
        dynamics_model=CartesianDynamics(),
        D=numSteps,
        controller=ControllerCLFBayesian(
            PiecewiseLinearPlanner(x, x_g, numSteps, dt),
            coordinate_converter = lambda x, x_g: x,
            dynamics = LearnedShiftInvariantDynamics(dt = dt),
            # dynamics = CartesianDynamics(),
            clf = CLFCartesian(
                Kp = torch.tensor(Kp)
            ),
            cbfs = cbfs(x , x_g),
            cbf_gammas = torch.tensor(cbf_gammas)
        ).control,
        visualizer=Visualizer(
            PiecewiseLinearPlanner(x, x_g, numSteps, dt),
            dt,
            cbfs = cbfs(x, x_g)
        ),
        x0=x,
        dt=dt,
        **kw)


def track_trajectory_ackerman_clf_bayesian(x, x_g, dt = None,
                                           cbfs = None,
                                           cbf_gammas = None,
                                           numSteps = None,
                                           enable_learning = True,
                                           mean_dynamics_gen=partial(AckermanDrive,
                                                                     L = 10.0),
                                           true_dynamics_gen=partial(AckermanDrive,
                                                                     L = 1.0),
                                           visualizer_class=Visualizer,
                                           controller_class=ControllerCLFBayesian,
                                           train_every_n_steps = 20,
                                           **kw):
    """
    mean_dynamics is either ZeroDynamicsModel(m = 2, n = 3) or AckermanDrive(L = 10.0)
    """
    visualizer = visualizer_class(
        PiecewiseLinearPlanner(x, x_g, numSteps, dt),
        dt,
        cbfs = cbfs(x, x_g)
    )
    return sample_generator_trajectory(
        dynamics_model=true_dynamics_gen(),
        D=numSteps,
        controller=controller_class(
            PiecewiseLinearPlanner(x, x_g, numSteps, dt,
                                   frac_time_to_reach_goal=0.95),
            coordinate_converter = lambda x, x_g: x,
            dynamics = LearnedShiftInvariantDynamics(
                dt = dt,
                mean_dynamics = mean_dynamics_gen(),
                enable_learning = enable_learning,
                train_every_n_steps = train_every_n_steps
            ),
            # dynamics = ZeroDynamicsBayesian(m = 2, n = 3),
            clf = CLFCartesian(
                Kp = torch.tensor([0.9, 1.5, 0.])
            ),
            cbfs = cbfs(x , x_g),
            cbf_gammas = cbf_gammas,
            visualizer = visualizer
        ).control,
        visualizer = visualizer,
        x0=x,
        dt=dt,
        **kw)

####################################################################
# entry points: Possible main methods
####################################################################

def unicycle_demo(simulator = move_to_pose, exp_tags = []):
    global TBLOG
    state_start, state_goal = [-3, -1, -math.pi/4], [0, 0, math.pi/4]
    directory_name = ('data/runs/unicycle_move_to_pose_fixed_'
                          + '_'.join(exp_tags + [bayes_cbf.__version__]))
    TBLOG = SummaryWriter(directory_name)
    json.dump(dict(extract_keywords(simulator),
                    state_start=state_start,
                    state_goal=state_goal),
               open(osp.join(directory_name , CONFIG_FILE_BASENAME), 'w'),
               skipkeys=True, indent=True)
    simulator(torch.tensor(state_start), torch.tensor(state_goal))
    for i in range(0):
        if TBLOG is not None:
            TBLOG.close()
        x_start = 20 * random()
        y_start = 20 * random()
        theta_start = 2 * math.pi * random() - math.pi
        x_goal = 20 * random()
        y_goal = 20 * random()
        theta_goal = 2 * math.pi * random() - math.pi
        print("Initial x: %.2f m\nInitial y: %.2f m\nInitial theta: %.2f rad\n" %
              (x_start, y_start, theta_start))
        print("Goal x: %.2f m\nGoal y: %.2f m\nGoal theta: %.2f rad\n" %
              (x_goal, y_goal, theta_goal))
        state_goal = [x_goal, y_goal, theta_goal]
        state_start =[x_start, y_start, theta_start]
        directory_name = (('data/runs/unicycle_move_to_pose_%d_' % i)
                          + '_'.join(exp_tags + 
                                     [bayes_cbf.__version__]))
        TBLOG = SummaryWriter(directory_name)
        json.dump(dict(extract_keywords(simulator),
                        state_start=state_start,
                        state_goal=state_goal,
                        gitdescribe=gitdescribe(__file__)),
                open(osp.join(directory_name , CONFIG_FILE_BASENAME), 'w'),
                skipkeys=True, indent=True)
        simulator(torch.tensor(state_start), torch.tensor(state_goal))
    return directory_name


def unicycle_demo_clf_polar(dt = 0.01):
    return unicycle_demo(simulator=partial(
        move_to_pose_clf_polar, dt = dt))


def unicycle_demo_clf_cartesian(dt = 0.01):
    return unicycle_demo(simulator=partial(move_to_pose_clf_cartesian, dt = dt))


def unicycle_demo_pid(dt = 0.01):
    return unicycle_demo(simulator=partial(
        move_to_pose_pid, dt = dt))


def unicycle_demo_sim_cartesian_clf(dt = 0.01):
    return unicycle_demo(simulator=partial(move_to_pose_sample_clf_cartesian, dt = dt))


def unicycle_demo_sim_cartesian_clf_traj(
        dt = 0.01,
        numSteps = 400,
        cbfs = lambda x, x_g: [
            ObstacleCBF((x[:2] + x_g[:2])/2
                        + R90() @ (x[:2] - x_g[:2])/15,
                        (x[:2] - x_g[:2]).norm()/20),
            ObstacleCBF((x[:2] + x_g[:2])/2
                        - R90() @ (x[:2] - x_g[:2])/15,
                        (x[:2] - x_g[:2]).norm()/20),
        ],
        cbf_gammas = [torch.tensor(10.), torch.tensor(10.)]
        # cbfs = lambda x, x_g: []
        # cbf_gammas = []
):
    return unicycle_demo(simulator=partial(track_trajectory_clf_cartesian,
                                           dt = dt, cbfs = cbfs,
                                           cbf_gammas = cbf_gammas,
                                           numSteps = numSteps))

def unicycle_demo_track_trajectory_clf_bayesian(
        dt = 0.01,
        numSteps = 400,
        cbfs = obstacles_at_mid_from_start_and_goal,
        cbf_gammas = [10., 10.]
        # cbfs = lambda x, x_g: [],
        # cbf_gammas = []
):
    return unicycle_demo(simulator=partial(track_trajectory_clf_bayesian,
                                           dt = dt, cbfs = cbfs,
                                           cbf_gammas = cbf_gammas,
                                           numSteps = numSteps))

unicycle_demo_track_trajectory_ackerman_clf_bayesian = partial(
    unicycle_demo,
    simulator=partial(track_trajectory_ackerman_clf_bayesian,
                      dt = 0.01,
                      numSteps = 400,
                      cbfs = obstacles_at_mid_from_start_and_goal,
                      cbf_gammas = [5., 5.],
                      # cbfs = lambda x, x_g: [],
                      # cbf_gammas = [],
                      controller_class=ControllerCLFBayesian, # or ControllerCLF
                      visualizer_class=Logger, # or Visualizer
                      true_dynamics_gen=partial(AckermanDrive, L = 1.0),
                      mean_dynamics_gen=partial(AckermanDrive, L = 4.0), # or ZeroDynamicsBayesian
                      enable_learning = True),
    exp_tags = ['ackerman']
)

# Nov 16th
# Run four experiments
unicycle_demo_track_trajectory_ackerman_clf_bayesian_mult = partial(
    applyall,
    map(partial(recpartial,unicycle_demo_track_trajectory_ackerman_clf_bayesian),
        expand_variations(
                {'simulator.enable_learning' : kwvariations([True, False]),
                 'simulator.controller_class': kwvariations([ControllerCLFBayesian, ControllerCLF])})))

# Nov 18th
# Force the unicycle around the obstacle by controlling the uncertainity
unicycle_force_around_obstacle = partial(
    unicycle_demo,
    simulator=partial(track_trajectory_ackerman_clf_bayesian,
                      dt = 0.01,
                      numSteps = 400,
                      cbfs = partial(
                          single_obstacle_at_mid_from_start_and_goal,
                          term_weights=[0.5, 0.5]),
                      cbf_gammas = [5., 5.],
                      controller_class=ControllerCLFBayesian, # or ControllerCLF
                      visualizer_class=Visualizer, # or Logger
                      true_dynamics_gen=partial(AckermanDrive, L = 1.0),
                      mean_dynamics_gen=partial(AckermanDrive, L = 1.0,
                                                kernel_diag_A=[1e-2, 1e-2, 1e-2]),
                      enable_learning = False),
    exp_tags = ['around_obstacle'])

# Nov 19th
# Compare 
unicycle_force_around_obstacle_mult = partial(
    applyall,
    map(partial(recpartial, unicycle_force_around_obstacle),
        expand_variations(
            {'simulator.mean_dynamics_gen.kernel_diag_A':
             kwvariations([[1e-2, 1e-2, 1e-2],
                           [5e-2, 5e-2, 5e-2]])})))

# Dec 9th
# Recipe for Mean CBF visualization that collides
unicycle_mean_cbf_collides_obstacle_exp = partial(
    unicycle_demo,
    simulator=partial(track_trajectory_ackerman_clf_bayesian,
                      dt = 0.05,
                      numSteps = 200,
                      cbfs = partial(
                          obstacles_at_mid_from_start_and_goal,
                          term_weights=[0.7, 0.3]),
                      cbf_gammas = [5., 5.],
                      controller_class=partial(ControllerCLFBayesian,
                                               max_risk=0.5), # or ControllerCLFBayesian
                      visualizer_class=partial(Logger), # or Logger
                      true_dynamics_gen=partial(AckermanDrive, L = 12.0),
                      mean_dynamics_gen=partial(AckermanDrive, L = 1.0,
                                                kernel_diag_A=[1e-2, 1e-2, 1e-2]),
                      enable_learning = False),
    exp_tags = ['mean_cbf_collides'])

# Works on Dec 27th
# Try events_dir = 
# 'saved-runs/unicycle_move_to_pose_fixed_mean_cbf_collides_v1.2.3'
unicycle_mean_cbf_collides_obstacle_vis = partial(playback_logfile,
        visualizer_kw=dict(
            suptitle='Mean CBF',
            figsize=(3, 6),
            subplots_adjust_kw=dict(top=0.98, left=0.18, hspace=0.13,
                                    bottom=0.08 ))
)

def unicycle_mean_cbf_collides_obstacle(**kw):
    events_dir = unicycle_mean_cbf_collides_obstacle_exp(**kw)
    return unicycle_mean_cbf_collides_obstacle_vis(events_dir)


# Dec 9th
# Recipe for Bayes CBF visualization
unicycle_bayes_cbf_safe_obstacle_exp = recpartial(
    unicycle_mean_cbf_collides_obstacle_exp,
    {'exp_tags' : ['bayes_cbf_safe_obstacle'],
     'simulator.controller_class.max_risk' : 0.01 })
# Works on Dec 27th
# Try events_dir = 
# 'saved-runs/unicycle_move_to_pose_fixed_mean_cbf_collides_1209-1257'
unicycle_bayes_cbf_safe_obstacle_vis = partial(playback_logfile,
        visualizer_kw=dict(
            suptitle='Bayes CBF',
            figsize=(3, 6),
            subplots_adjust_kw=dict(top=0.98, left=0.18, hspace=0.13,
                                    bottom=0.08 ))
)


def unicycle_bayes_cbf_safe_obstacle(**kw):
    events_dir = unicycle_bayes_cbf_safe_obstacle_exp(**kw)
    return unicycle_bayes_cbf_safe_obstacle_vis(events_dir)


# Dec 8th
# Learning makes it pass, otherwise it gets stuck
unicycle_learning_helps_avoid_getting_stuck_exp = partial(
    unicycle_demo,
    simulator=partial(track_trajectory_ackerman_clf_bayesian,
                      dt = 0.01,
                      numSteps = 200,
                      cbfs = partial(
                          obstacles_at_mid_from_start_and_goal,
                          term_weights=[0.7, 0.3]),
                      cbf_gammas = [5., 5.],
                      controller_class=partial(ControllerCLFBayesian,
                                               max_risk=0.01), # or ControllerCLFBayesian
                      visualizer_class=partial(Logger, # or Logger
                                               suptitle='With Learning',
                                               subplots_adjust_kw=dict(left=0.25,
                                                                       top=0.95),
                                               scalar_plots=['TraceKnl', 'CBC']),
                      true_dynamics_gen=partial(AckermanDrive, L = 1.0),
                      mean_dynamics_gen=partial(AckermanDrive, L = 12.0,
                                                kernel_diag_A=[1.0, 1.0, 1.0]),
                      train_every_n_steps = 40, # Change this
                      enable_learning = True), # Do not change this
    exp_tags = ['learning_helps_avoid_getting_stuck'])


# Dec 27th
# unicycle_learning_helps_avoid_getting_stuck_vis('saved-runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.2.3')
unicycle_learning_helps_avoid_getting_stuck_vis = partial(
    playback_logfile,
    visualizer_kw=dict(
        figsize=(3, 6),
        subplots_adjust_kw=dict(top=0.98, left=0.22, hspace=0.13,
                                bottom=0.08 ),
        plots_ylims=dict(DetKnl=(1e-7, 100),
                         CBC=(-0.1,11))
        )
    )

def unicycle_learning_helps_avoid_getting_stuck(**kw):
    events_dir = unicycle_learning_helps_avoid_getting_stuck_exp(**kw)
    unicycle_learning_helps_avoid_getting_stuck_vis(events_dir)


# Dec 8th
unicycle_no_learning_gets_stuck_exp = recpartial(
    unicycle_learning_helps_avoid_getting_stuck_exp,
    {'exp_tags' : ['no_learning_gets_stuck'],
     'simulator.train_every_n_steps': 200,
     'simulator.visualizer_class.suptitle' : 'No Learning'})

# Dec 27th
# unicycle_no_learning_gets_stuck_vis('saved-runs/unicycle_move_to_pose_fixed_no_learning_gets_stuck_v1.2.3/')
unicycle_no_learning_gets_stuck_vis = partial(
    playback_logfile,
    visualizer_kw=dict(
        figsize=(3, 6),
        subplots_adjust_kw=dict(top=0.98, left=0.22, hspace=0.13,
                                bottom=0.08 ),
        plots_ylims=dict(DetKnl=(1e-7, 100),
                         CBC=(-0.1,11))
        )
    )

def unicycle_no_learning_gets_stuck(**kw):
    events_dir = unicycle_no_learning_gets_stuck_exp(**kw)
    unicycle_no_learning_gets_stuck_vis(events_dir)


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
    n = np.prod(FX_true.shape[:-1])
    return np.sqrt(to_numpy(sq_sum) / n)


def unicycle_speed_test_matrix_vector_exp(
        # max_train_variations=[64, 512], # testing GPU
        max_train_variations=[64, 64+16, 64+32, 128], # final GPU
        # max_train_variations=[10, 25, 50, 80, 125], # CPU
        # ntimes = 20, # How many times the inference should be repeated
        ntimes = 10, # How many times the inference should be repeated
        repeat = 50, # How many times to repeat at take minima
        errorbartries = 20, # How many repeats needed to plot accuracy error bars
        state_start = [-3, -1, -math.pi/4],
        state_goal = [0, 0, math.pi/4],
        numSteps = 512,
        dt = 0.01,
        true_dynamics_gen=partial(AckermanDrive,
                                  L = 1.0),
        mean_dynamics_gen=partial(AckermanDrive,
                                  L = 12.0),
        logger_class=partial(TBLogger,
                             exp_tags=['unicycle_speed_test_matrix_vector'],
                             runs_dir='data/runs'),
        exps=dict(
            matrix=dict(
                regressor_class=partial(
                    LearnedShiftInvariantDynamics,
                    learned_dynamics_class=ControlAffineRegressorExact)),
            vector=dict(
                regressor_class=partial(
                    LearnedShiftInvariantDynamics,
                    learned_dynamics_class=ControlAffineRegressorVector)),
            vectordiag=dict(
                regressor_class=partial(
                    LearnedShiftInvariantDynamics,
                    learned_dynamics_class=ControlAffineRegVectorDiag)),
            matrixdiag=dict(
                regressor_class=partial(
                    LearnedShiftInvariantDynamics,
                    learned_dynamics_class=ControlAffineRegMatrixDiag))),
):
    logger = logger_class()
    global TBLOG
    TBLOG = logger.summary_writer
    true_dynamics_model=true_dynamics_gen()
    Xdot, X, U = sample_generator_trajectory(
        dynamics_model=true_dynamics_model,
        D=numSteps,
        controller=ControllerCLF(
            NoPlanner(torch.tensor(state_goal)),
            coordinate_converter = lambda x, x_g: x,
            dynamics = CartesianDynamics(),
            clf = CLFCartesian()
        ).control,
        visualizer=VisualizerZ(),
        x0=state_start,
        dt=dt)
    for t,  (dx, x, u) in enumerate(zip(Xdot, X, U)):
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
        XdotTrain = Xdot[train_indices, :]

        slice_range = []
        for i, num in enumerate([1, 1, 20]):
            x_range = slice(*list(map(float,
                                    (Xtrain[:, i].min(),
                                    Xtrain[:, i].max(),
                                     (Xtrain[:, i].max() - Xtrain[:, i].min()) / num))))
            slice_range.append(x_range)
        xtest_grid = np.mgrid[tuple(slice_range)]
        Xtest = torch.from_numpy(
            xtest_grid.reshape(-1, Xtrain.shape[-1])).to(
                dtype=Xtrain.dtype,
                device=Xtrain.device)
        # b, n, 1+m
        FX_true = true_dynamics_model.F_func(Xtest).transpose(-2, -1)

        for name, kw in exps.items():
            model = kw['regressor_class'](dt = dt,
                                          mean_dynamics=mean_dynamics_gen(),
                                          max_train=max_train)
            model.fit(Xtrain, Utrain, XdotTrain, training_iter=50)
            elapsed = min(timeit.repeat(
                stmt='model.custom_predict_fullmat(Xtest);model.clear_cache()',
                repeat=repeat,
                number=ntimes,
                globals=dict(model=model,
                             Xtest=Xtest)))
            errors = compute_errors(kw['regressor_class'],
                                    partial(sample_generator_trajectory,
                                            dynamics_model=true_dynamics_model,
                                            D=numSteps,
                                            controller=ControllerCLF(
                                                NoPlanner(torch.tensor(state_goal)),
                                                coordinate_converter = lambda x, x_g: x,
                                                dynamics = CartesianDynamics(),
                                                clf = CLFCartesian()
                                            ).control,
                                            visualizer=VisualizerZ(),
                                            x0=state_start,
                                            dt=dt),
                                    true_dynamics_model,
                                    mean_dynamics_gen = mean_dynamics_gen,
                                    dt = dt,
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


def get_grid_from_Xtrain(Xtrain):
    theta_range = slice(Xtrain[:, 0].min(), Xtrain[:, 0].max(),
                        (Xtrain[:, 0].max() - Xtrain[:, 0].min()) / 20)
    omega_range = slice(Xtrain[:, 1].min(), Xtrain[:, 1].max(),
                        (Xtrain[:, 1].max() - Xtrain[:, 1].min()) / 20)

    theta_omega_grid = np.mgrid[theta_range, omega_range]
    return theta_omega_grid


def compute_errors(regressor_class, sampling_callable, true_dynamics,
                   ntries=5, max_train=200, test_on_grid=False, ntest=400,
                   dt = 0.01, mean_dynamics_gen = None):
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
        FX_true = true_dynamics.F_func(Xtest).transpose(-2, -1) # (b, n, 1+m) -> (b, 1+m, n)

        dgp = regressor_class(dt = dt,
                              mean_dynamics=mean_dynamics_gen(),
                              max_train=max_train)


        # b(1+m)n
        FX_learned, var_FX = dgp.custom_predict_fullmat(
            Xtest.reshape(-1, Xtest.shape[-1]))
        b = Xtest.shape[0]
        n = true_dynamics.state_size
        m = true_dynamics.ctrl_size

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


def unicycle_speed_test_matrix_vector_vis(
        events_file='saved-runs/unicycle_speed_test_matrix_vector_v1.6.1-1-gd845a15/events.out.tfevents.1608669492.dwarf.656.0',
        exp_names=['matrix', 'vector', 'matrixdiag', 'vectordiag'],
):
    logdata = load_tensorboard_scalars(events_file)
    events_dir = osp.dirname(events_file)
    exp_data = dict()
    for gp in exp_names:
        training_samples, elapsed = zip(*logdata[gp + '/elapsed'])
        training_samples, errors = zip(*logdata[gp + '/errors'])
        exp_data[gp] = dict(elapsed=elapsed, errors=errors)
    fig, axes = plt.subplots(1,2, figsize=(8, 4.7))
    fig.subplots_adjust(bottom=0.1, wspace=0.20, top=0.90, right=0.95, left=0.1)
    fig.suptitle('Ackerman Drive')
    speed_test_matrix_vector_plot(axes,
                                              training_samples,
                                              exp_data)
    plot_file = osp.join(events_dir, 'speed_test_mat_vec_ind.pdf')
    fig.savefig(plot_file)
    # subprocess.run(["xdg-open", plot_file])
    return plot_file


def unicycle_speed_test_matrix_vector(**kw):
    events_file = unicycle_speed_test_matrix_vector_exp(**kw)
    return unicycle_speed_test_matrix_vector_vis(events_file)




if __name__ == '__main__':
    import doctest
    doctest.testmod() # always run unittests first
    # Entrypoints:
    # unicycle_demo_pid()
    # unicycle_demo_clf_polar()
    # unicycle_demo_clf_cartesian()
    # unicycle_demo_sim_cartesian_clf()
    # unicycle_demo_sim_cartesian_clf_traj()
    # unicycle_demo_track_trajectory_clf_bayesian()
    # unicycle_demo_track_trajectory_ackerman_clf_bayesian()
    # unicycle_demo_track_trajectory_ackerman_clf_bayesian_mult()
    # unicycle_force_around_obstacle_mult()

    # unicycle_mean_cbf_collides_obstacle()
    # unicycle_bayes_cbf_safe_obstacle()

    # unicycle_speed_test_matrix_vector()

    # Dec 27th
    # unicycle_mean_cbf_collides_obstacle_vis('saved-runs/unicycle_move_to_pose_fixed_mean_cbf_collides_v1.2.3')
    # unicycle_bayes_cbf_safe_obstacle_vis('saved-runs/unicycle_move_to_pose_fixed_mean_cbf_collides_1209-1257')
    # unicycle_learning_helps_avoid_getting_stuck_vis('saved-runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.2.3')
    # unicycle_no_learning_gets_stuck_vis('saved-runs/unicycle_move_to_pose_fixed_no_learning_gets_stuck_v1.2.3/')

    # Dec 28th
    # Regenerate uncertainty measure replaced with \ubfu
    # unicycle_no_learning_gets_stuck()
    # unicycle_learning_helps_avoid_getting_stuck()

    unicycle_no_learning_gets_stuck_vis('saved-runs/unicycle_move_to_pose_fixed_no_learning_gets_stuck_v1.6.2/')
    unicycle_learning_helps_avoid_getting_stuck_vis('saved-runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.6.2/')

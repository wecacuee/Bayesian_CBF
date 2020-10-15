"""

Move to specified pose

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai(@Atsushi_twi)

P. I. Corke, "Robotics, Vision & Control", Springer 2017, ISBN 978-3-319-54413-7

"""
from datetime import datetime
from random import random
from functools import partial
from collections import namedtuple
import sys
import math
import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

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


from torch.utils.tensorboard import SummaryWriter

from bayes_cbf.gp_algebra import DeterministicGP
from bayes_cbf.control_affine_model import ControlAffineRegressor
from bayes_cbf.misc import to_numpy, normalize_radians
from bayes_cbf.sampling import sample_generator_trajectory
from bayes_cbf.planner import PiecewiseLinearPlanner, SplinePlanner
from bayes_cbf.cbc2 import cbc2_quadratic_terms, cbc2_gp, cbc2_safety_factor

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


class LearnedShiftInvariantDynamics:
    state_size = 3
    ctrl_size = 2
    def __init__(self,
                 dt = None,
                 learned_dynamics = None,
                 mean_dynamics = CartesianDynamics(),
                 max_train = 200,
                 training_iter = 100,
                 shift_invariant = True,
                 train_every_n_steps = 10,
                 enable_learning = True):
        self.max_train = max_train
        self.training_iter = training_iter
        self.dt = dt
        self.mean_dynamics = mean_dynamics
        self.learned_dynamics = (learned_dynamics
                                 if learned_dynamics is not None else
                                 ControlAffineRegressor(
                                     self.state_size,
                                     self.ctrl_size))
        self.shift_invariant = shift_invariant
        self._shift_invariant_wrapper = (self._shift_invariant_input
                                         if shift_invariant else
                                         lambda x: x)
        self.train_every_n_steps = train_every_n_steps
        self.enable_learning = enable_learning
        self.Xtrain = []
        self.Utrain = []

    def _shift_invariant_input(self, X):
        assert X.shape[-1] == self.state_size
        return torch.cat(
            [torch.zeros((*X.shape[:-1], self.state_size-1)),
             X[..., self.state_size-1:]], dim=-1)

    def f_func(self, X):
        x_orig = self._shift_invariant_wrapper(X)
        return self.mean_dynamics.f_func(x_orig) + self.learned_dynamics.f_func(x_orig)

    def g_func(self, X):
        x_orig = self._shift_invariant_wrapper(X)
        return self.mean_dynamics.g_func(x_orig) + self.learned_dynamics.g_func(x_orig)

    def train(self, xi, uopt):
        if (len(self.Xtrain) > 0
            and len(self.Xtrain) % int(self.train_every_n_steps) == 0
            and self.enable_learning
        ):
            # train every n steps
            self._train()

        # record the xi, ui pair
        self.Xtrain.append(xi.detach())
        self.Utrain.append(uopt.detach())
        assert len(self.Xtrain) == len(self.Utrain)

    def _train(self):
        LOG.info("Training GP with dataset size {}".format(len(self.Xtrain)))
        if not len(self.Xtrain):
            return
        assert len(self.Xtrain) == len(self.Utrain), \
            "Call train when Xtrain and Utrain are balanced"

        Xtrain = torch.cat(self.Xtrain).reshape(-1, self.Xtrain[0].shape[-1])

        Utrain = torch.cat(self.Utrain).reshape(-1, self.Utrain[0].shape[-1])
        XdotTrain = (Xtrain[1:, :] - Xtrain[:-1, :]) / self.dt

        if self.shift_invariant:
            Xtrain[:, :2] = 0.

        XdotMean = self.mean_dynamics.f_func(Xtrain) + (
            self.mean_dynamics.g_func(Xtrain).bmm(Utrain.unsqueeze(-1)).squeeze(-1))
        XdotError = XdotTrain - XdotMean[:-1, :]

        LOG.info("Training model with datasize {}".format(XdotTrain.shape[0]))
        if XdotTrain.shape[0] > self.max_train:
            indices = torch.randint(XdotTrain.shape[0], (self.max_train,))
            train_data = Xtrain[indices, :], Utrain[indices, :], XdotError[indices, :],
        else:
            train_data = Xtrain[:-1, :], Utrain[:-1, :], XdotError

        self.learned_dynamics.fit(*train_data, training_iter=100)

    def fu_func_gp(self, U):
        md = self.mean_dynamics
        return (DeterministicGP(lambda x: md.f_func(x) + md.g_func(x) @ U,
                                shape=(self.state_size,))
                + self.learned_dynamics.fu_func_gp(U))

    def step(self, u_torch, dt):
        x = self.current_state
        xdot = self.f_func(x) + self.g_func(x) @ u_torch
        self.current_state = x + xdot * dt
        return dict(xdot = xdot,
                    x = self.current_state)


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
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def cbf(self, state):
        self.center, self.radius = map(partial(torch_to,
                                               device=state.device,
                                               dtype=state.dtype),
                                       (self.center, self.radius))
        return ((state[:2] - self.center)**2).sum() - self.radius**2

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
        gcbf = torch.zeros_like(state)
        gcbf[:2] = 2*(state[:2]-self.center)
        return gcbf


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
                 cbf_gammas = []):
        self.planner = planner
        self.u_dim = 2
        self.coordinate_converter = coordinate_converter
        self.dynamics = dynamics
        self.clf = clf
        self.clf_gamma = 10
        self.clf_relax_weight = 10
        self.cbfs = cbfs
        self.cbf_gammas = cbf_gammas

    def _clc(self, x, state_goal, t):
        polar = self.coordinate_converter(x, state_goal)
        fx = self.dynamics.f_func(polar)
        gx = self.dynamics.g_func(polar)
        gclf = self.clf.grad_clf(polar, state_goal)
        gclf_goal = self.clf.grad_clf_wrt_goal(polar, state_goal)
        TBLOG.add_scalar("x_0", x[0], t)
        bfa = to_numpy(gclf @ gx)
        b = to_numpy(gclf @ fx
                     - gclf_goal @ self.planner.dot_plan(t)
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


class ControllerCLFBayesian:
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
                 ctrl_min = np.array([-10., -np.pi*5]),
                 ctrl_max = np.array([10., np.pi*5])):
        self.planner = planner
        self.u_dim = 2
        self.coordinate_converter = coordinate_converter
        self.dynamics = dynamics
        self.clf = clf
        self.clf_gamma = 10
        self.clf_relax_weight = 10
        self.cbfs = cbfs
        self.cbf_gammas = cbf_gammas
        self.ctrl_min = ctrl_min
        self.ctrl_max = ctrl_max

    @staticmethod
    def convert_cbc_terms_to_socp_terms(bfe, e, V, bfv, v, extravars,
                                        testing=True):
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
            try:
                L = torch.cholesky(Asq) # (m+1) x (m+1)
            except RuntimeError as err:
                if "cholesky" in str(err) and "singular" in str(err):
                    L = torch.cholesky(Asq + torch.diag(torch.ones(m+1)*1e-3))
                else:
                    raise
            # try:
            #     n_constraints = m+1
            #     L = torch.cholesky(Asq) # (m+1) x (m+1)
            # except RuntimeError as err:
            #     if "cholesky" in str(err) and "singular" in str(err):
            #         if torch.allclose(v, torch.zeros((1,))) and torch.allclose(bfv, torch.zeros(bfv.shape[0])):
            #             L = torch.zeros((m+1, m+1))
            #             L[1:, 1:] = torch.cholesky(V)
            #         else:
            #             diag_e, U = torch.symeig(Asq, eigenvectors=True)
            #             n_constraints = torch.nonzero(diag_e, as_tuple=True)[0].shape[-1]
            #             L = torch.diag(diag_e).sqrt() @ U[:, -n_constraints:]
            #     else:
            #         raise

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
        clf = DeterministicGP(lambda x: self.clf_gamma * self.clf.clf_terms(x, state_goal).sum(), shape=(1,))
        gclf = DeterministicGP(lambda x: self.clf.grad_clf(x, state_goal), shape=(n,))
        gclf_goal = DeterministicGP(lambda x: self.clf.grad_clf_wrt_goal(x, state_goal),
                                    shape=(n,))
        fu_func_gp = self.dynamics.fu_func_gp(u)
        dot_plan = DeterministicGP(lambda x: self.planner.dot_plan(t), shape=(n,))
        return gclf.t() @ fu_func_gp + gclf_goal.t() @ dot_plan + clf

    def _clc_terms(self, state, state_goal, t):
        m = self.u_dim
        (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
            lambda u: self._clc(state, state_goal, u, t),
            state, torch.rand(m))
        A, bfb, bfc, d = self.convert_cbc_terms_to_socp_terms(
            bfe, e, V, bfv, v, 0)
        return map(to_numpy, (A, bfb, bfc, d))

    def _cbc(self, cbf, cbf_gamma, state, state_goal, u, t):
        n = state.shape[-1]
        cbfx = DeterministicGP(lambda x: cbf_gamma * cbf.cbf(x), shape=(1,))
        gcbfx = DeterministicGP(lambda x: cbf.grad_cbf(x), shape=(n,))
        fu_func_gp = self.dynamics.fu_func_gp(u)
        return gcbfx.t() @ fu_func_gp + cbfx


    def _cbc_terms(self, cbf, cbf_gamma, state, state_goal, t):
        m = self.u_dim
        (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
            lambda u: self._cbc(cbf, cbf_gamma, state, state_goal, u, t),
            state, torch.rand(m))
        A, bfb, bfc, d = self.convert_cbc_terms_to_socp_terms(
            bfe, e, V, bfv, v, 0)
        return map(to_numpy, (A, bfb, bfc, d))

    def _cbcs(self, state, state_goal, t):
        for cbf, cbf_gamma in zip(self.cbfs, self.cbf_gammas):
            yield self._cbc_terms(cbf, cbf_gamma, state, state_goal, t)

    def control(self, x_torch, t):
        import cvxpy as cp # pip install cvxpy
        state_goal = self.planner.plan(t)
        x = x_torch
        uvar = cp.Variable(self.u_dim)
        uvar.value = np.zeros(self.u_dim)
        relax = cp.Variable(1)
        obj = cp.Minimize(
            cp.sum_squares(uvar) + self.clf_relax_weight * relax)
        clc_A, clc_bfb, clc_bfc, clc_d = self._clc_terms(x, state_goal, t)
        constraints = [
            clc_bfc @ uvar + clc_d - relax >= cp.norm(clc_A @ uvar + clc_bfb)]

        for cbc_A, cbc_bfb, cbc_bfc, cbc_d in self._cbcs(x, state_goal, t):
            constraints.append(
                cbc_bfc @ uvar + cbc_d >= cp.norm(cbc_A @ uvar + cbc_bfb)
            )

        problem = cp.Problem(obj, constraints)
        problem.solve(solver='GUROBI', verbose=True)
        if problem.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            # print("Optimal value: %s" % problem.value)
            pass
        else:
            raise ValueError(problem.status)
        # for variable in problem.variables():
        #     print("Variable %s: value %s" % (variable.name(), variable.value))
        uopt = torch_to(torch.from_numpy(uvar.value),
                        device=getattr(x_torch, 'device', None),
                        dtype=x_torch.dtype)
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

class Visualizer:
    def __init__(self, planner, dt, cbfs=[]):
        self.planner = planner
        self.dt = dt
        self.state_start = None
        self.cbfs = cbfs
        self.x_traj, self.y_traj = [], []


    def setStateCtrl(self, state, u, t=None, **kw):
        if t == 0:
            self.state_start = state.clone()
        plt.cla()
        scale = (self.planner.x_goal[:2] - self.state_start[:2]).norm() / 10.
        x_start, y_start, theta_start = self.state_start
        plt.arrow(x_start, y_start, torch.cos(theta_start) * scale,
                    torch.sin(theta_start)*scale, color='r', width=0.1*scale)
        x_plan, y_plan, theta_plan = self.planner.plan(t)
        plt.plot(x_plan, y_plan, 'g+', linewidth=0.01)
        x_goal, y_goal, theta_goal = self.planner.x_goal
        plt.plot(x_goal, y_goal, 'g+', linewidth=0.4)
        plt.gca().add_patch(Circle(np.array([x_goal, y_goal]),
                                    radius=scale/5, fill=False, color='g'))
        for cbf in self.cbfs:
            circle = Circle(to_numpy(cbf.center),
                            radius=to_numpy(cbf.radius),
                            fill=False, color='r')
            plt.gca().add_patch(circle)
        x, y, theta = state
        self.x_traj.append(x)
        self.y_traj.append(y)
        plot_vehicle(x, y, theta, self.x_traj, self.y_traj, self.dt,
                     self.state_start, self.planner.x_goal)

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


def plot_vehicle(x, y, theta, x_traj, y_traj, dt, state_start, state_goal):  # pragma: no cover
    # Corners of triangular vehicle when pointing to the right (0 radians)
    scale = (state_goal[:2] - state_start[:2]).norm() / 10.
    triangle = torch.tensor([[0.5, 0],
                             [-0.5, 0.25],
                             [-0.5, -0.25]])

    tri = (rot_matrix(theta) @ (scale * triangle.T) + torch.tensor([x, y]).reshape(-1, 1)).T

    plt.gca().add_patch(Polygon(to_numpy(tri), fill=False, edgecolor='k'))

    plt.plot(x_traj, y_traj, 'b--')

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [sys.exit(0) if event.key == 'escape' else None])

    plt.gca().set_aspect('equal')
    state_min = torch.min(state_start[:2], state_goal[:2]).min()
    state_max = torch.max(state_start[:2], state_goal[:2]).max()
    plt.xlim(-0.5 + state_min, state_max + 0.5)
    plt.ylim(-0.5 + state_min, state_max + 0.5)

    plt.pause(dt)


def rot_matrix(theta):
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta),  torch.cos(theta)]
    ])


class NoPlanner:
    def __init__(self, x_goal):
        self.x_goal = x_goal

    def plan(self, t):
        return self.x_goal

    def dot_plan(self, t):
        return torch.zeros_like(self.x_goal)

def rotmat(theta):
    return torch.tensor([[theta.cos(), -theta.sin()],
                         [theta.sin(), theta.cos()]])

def R90():
    return torch.tensor([[0., -1.],
                         [1, 0.]])

####################################################################
# configured methods
####################################################################

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
                                   **kw):
    return sample_generator_trajectory(
        dynamics_model=CartesianDynamics(),
        D=numSteps,
        controller=ControllerCLF(
            PiecewiseLinearPlanner(x, x_g, numSteps, dt),
            coordinate_converter = lambda x, x_g: x,
            dynamics = CartesianDynamics(),
            clf = CLFCartesian(
                Kp = torch.tensor([0.9, 1.5, 0.])
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
                                  **kw):
    return sample_generator_trajectory(
        dynamics_model=CartesianDynamics(),
        D=numSteps,
        controller=ControllerCLF(
            PiecewiseLinearPlanner(x, x_g, numSteps, dt),
            coordinate_converter = lambda x, x_g: x,
            dynamics = LearnedShiftInvariantDynamics(
                dt = dt,
                learned_dynamics = ControlAffineRegressor(
                    x_dim = x.shape[-1],
                    u_dim = 2
                )),
            clf = CLFCartesian(
                Kp = torch.tensor([0.9, 1.5, 0.])
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


####################################################################
# entry points: Possible main methods
####################################################################

def unicycle_demo(simulator = move_to_pose):
    global TBLOG
    TBLOG = SummaryWriter('data/runs/unicycle_move_to_pose_fixed_'
                            + datetime.now().strftime("%m%d-%H%M"))
    simulator(torch.tensor([-3, -1, -math.pi/4]), torch.tensor([0, 0, math.pi/4]))
    for i in range(5):
        TBLOG = SummaryWriter(('data/runs/unicycle_move_to_pose_%d_' % i)
                              + datetime.now().strftime("%m%d-%H%M"))
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
        state_goal = torch.tensor([x_goal, y_goal, theta_goal])
        state_start = torch.tensor([x_start, y_start, theta_start])
        simulator(state_start, state_goal)


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
    return unicycle_demo(simulator=partial(track_trajectory_clf_bayesian,
                                           dt = dt, cbfs = cbfs,
                                           cbf_gammas = cbf_gammas,
                                           numSteps = numSteps))


if __name__ == '__main__':
    import doctest
    doctest.testmod() # always run unittests first
    # Run any one of these
    # unicycle_demo_pid()
    # unicycle_demo_clf_polar()
    # unicycle_demo_clf_cartesian()
    # unicycle_demo_sim_cartesian_clf()
    # unicycle_demo_sim_cartesian_clf_traj()
    unicycle_demo_track_trajectory_clf_bayesian()

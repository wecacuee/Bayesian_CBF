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

from bayes_cbf.misc import to_numpy, normalize_radians
from bayes_cbf.sampling import sample_generator_trajectory
from bayes_cbf.planner import PiecewiseLinearPlanner, SplinePlanner

LOG = SummaryWriter('data/runs/unicycle_move_to_pose' + datetime.now().strftime("%m%d-%H%M"))

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

    def g_func(self, state: CartesianState):
        x, y, theta = state
        return torch.tensor([[math.cos(theta), 0],
                         [math.sin(theta), 0],
                         [0, 1]])


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
        LOG.add_scalar("x_0", x[0], t)
        # print("x :", x)
        # print("clf terms :", self.clf.clf_terms(polar, state_goal))
        # print("clf:", self.clf.clf_terms(polar, state_goal).sum())
        # print("grad_x clf:", gclf)
        # print("g(x): ", g(polar))
        # print("grad_u clf:", gclf @ g(polar))
        bfa = to_numpy(gclf @ gx)
        b = to_numpy(gclf @ fx
                     - gclf_goal @ self.planner.dot_plan(t) + self.clf_gamma *
                     self.clf.clf_terms(polar, state_goal).sum())
        return bfa, b

    def _cbcs(self, x, t):
        fx = self.dynamics.f_func(x)
        gx = self.dynamics.g_func(x)
        for cbf, cbf_gamma in zip(self.cbfs, self.cbf_gammas):
            gcbf = cbf.grad_cbf(x)
            cbfx = cbf.cbf(x)
            LOG.add_scalar("cbf", cbfx, t)
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
        for cbc_bfa, cbc_b in self._cbcs(x, t):
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
        return torch_to(torch.from_numpy(uvar.value),
                        device=getattr(x_torch, 'device', None),
                        dtype=x_torch.dtype)


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
                 dynamics=CartesianDynamics()):
    """
    rho is the distance between the robot and the goal position
    alpha is the angle to the goal relative to the heading of the robot
    beta is the angle between the robot's position and the goal position plus the goal angle

    Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards the goal
    Kp_beta*beta rotates the line so that it is parallel to the goal angle
    """

    state = state_start.clone()
    count = 0
    visualizer = Visualizer(state_goal, dt)
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


class LinearPlanner:
    def __init__(self, x_goal):
        self.x_goal = x_goal

    def plan(self, t):
        pass

    def dot_plan(self, t):
        pass

def rotmat(theta):
    return torch.tensor([[theta.cos(), -theta.sin()],
                         [theta.sin(), theta.cos()]])

def R90():
    return torch.tensor([[0., -1.],
                         [1, 0.]])

class Configs:
    @property
    def clf_polar(self):
        return dict(simulator=partial(
            lambda x, x_g, **kw: move_to_pose(
                x , x_g,
                dynamics=CartesianDynamics(),
                controller=ControllerCLF(
                    NoPlanner(x_g),
                    coordinate_converter = cartesian2polar,
                    dynamics=PolarDynamics(),
                    clf = CLFPolar()
                ),
                **kw
            )))

    @property
    def clf_cartesian(self):
        return dict(simulator=partial(
            lambda x, x_g, **kw: move_to_pose(x, x_g,
                                              dynamics=CartesianDynamics(),
                                              controller=ControllerCLF(
                                                  NoPlanner(x_g),
                                                  coordinate_converter = lambda x, x_g: (x),
                                                  dynamics=CartesianDynamics(),
                                                  clf = CLFCartesian()
                                              ))))

    @property
    def pid(self):
        return dict(simulator=partial(
            lambda x, x_g, **kw: move_to_pose(x, x_g,
                                      dynamics=CartesianDynamics(),
                                      controller=ControllerPID(NoPlanner(x_g)))))

    @property
    def sim_cartesian_clf(self):
        dt = 0.01
        return dict(simulator=partial(
            lambda x, x_g, **kw: sample_generator_trajectory(
                                      dynamics_model=CartesianDynamics(),
                                      D=200,
                                      controller=ControllerCLF(
                                          NoPlanner(x_g),
                                          coordinate_converter = lambda x, x_g: x,
                                          dynamics = CartesianDynamics(),
                                          clf = CLFCartesian()
                                      ).control,
                                      visualizer=Visualizer(x_g, dt),
                                      x0=x,
                                      dt=dt)))

    @property
    def sim_cartesian_clf_traj(self):
        dt = 0.01
        numSteps = 400
        cbfs = lambda x, x_g: [
            ObstacleCBF((x[:2] + x_g[:2])/2
                        + R90() @ (x[:2] - x_g[:2])/15,
                        (x[:2] - x_g[:2]).norm()/20),
            ObstacleCBF((x[:2] + x_g[:2])/2
                        - R90() @ (x[:2] - x_g[:2])/15,
                        (x[:2] - x_g[:2]).norm()/20),
        ]
        cbf_gammas = [torch.tensor(10.), torch.tensor(10.)]
        # cbfs = lambda x, x_g: []
        # cbf_gammas = []
        return dict(simulator=partial(
            lambda x, x_g, **kw: sample_generator_trajectory(
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
                                      dt=dt)))

def main(simulator = move_to_pose):
    simulator(torch.tensor([-3, -1, -math.pi/4]), torch.tensor([0, 0, math.pi/4]))
    for i in range(5):
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

if __name__ == '__main__':
    import doctest
    doctest.testmod() # always run unittests first
    main(**getattr(Configs(), 'sim_cartesian_clf_traj')) # 'pid', 'clf_polar' or 'clf_cartesian' 'sim_cartesian_clf', 'sim_cartesian_clf_traj'

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

from bayes_cbf.misc import to_numpy

LOG = SummaryWriter('data/runs/' + datetime.now().strftime("%m%d-%H%M"))

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
    theta = normalize_angle(phi + alpha)
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
        return (torch.tensor([[-torch.cos(alpha), 0],
                          [-torch.sin(alpha)/rho, 1],
                          [-torch.sin(alpha)/rho, 0]])
                if (rho > 1e-6) else
                torch.array([[-1, 0],
                          [-1, 1],
                          [-1, 0]]))

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


def normalize_angle(theta):
    # Restrict alpha and beta (angle differences) to the range
    # [-pi, pi] to prevent unstable behavior e.g. difference going
    # from 0 rad to 2*pi rad with slight turn
    return (theta + math.pi) % (2 * math.pi) - math.pi


def angdiff(thetap, theta):
    return normalize_angle(thetap - theta)


def cosdist(thetap, theta):
    return 1 - torch.cos(thetap - theta)


def angdist(thetap, theta):
    return angdiff(thetap, theta)**2

class CLFPolar:
    def __init__(self,
                 Kp = torch.tensor([5, 15, 40, 0])/10.):
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
                 Kp = torch.tensor([9, 15, 40])/10.):
        self.Kp = Kp

    def clf_terms(self, state, state_goal):
        rho, alpha, beta = cartesian2polar(state, state_goal)
        x,y, theta = state
        x_goal, y_goal, theta_goal = state_goal
        return torch.tensor((0.5 * self.Kp[0] * rho ** 2,
                         self.Kp[1] * (1-torch.cos(alpha)),
                         self.Kp[2] * (1-torch.cos(beta))
        ))

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

    def isconverged(self, x, state_goal):
        rho, alpha, beta = cartesian2polar(x, state_goal)
        return rho < 1e-3


class ControllerCLF:
    """
    Aicardi, M., Casalino, G., Bicchi, A., & Balestrino, A. (1995). Closed loop steering of unicycle like vehicles via Lyapunov techniques. IEEE Robotics & Automation Magazine, 2(1), 27-35.
    """
    def __init__(self, # simulation parameters
                 state_goal,
                 u_dim = 2,
                 coordinate_converter = cartesian2polar,
                 dynamics = PolarDynamics(),
                 clf = CLFPolar()):
        self.state_goal = state_goal
        self.u_dim = 2
        self.coordinate_converter = coordinate_converter
        self.dynamics = dynamics
        self.clf = clf

    def _clf(self, polar, state_goal):
        return self.clf.clf_terms(polar, state_goal).sum()

    def _grad_clf(self, polar, state_goal):
        return self.clf.grad_clf(polar, state_goal)

    def _clc(self, x, state_goal, u, t):
        polar = self.coordinate_converter(x, state_goal)
        f = self.dynamics.f_func
        g = self.dynamics.g_func
        gclf = self._grad_clf(polar, state_goal)
        LOG.add_scalar("x_0", x[0], t)
        # print("x :", x)
        # print("clf terms :", self.clf.clf_terms(polar, state_goal))
        # print("clf:", self.clf.clf_terms(polar, state_goal).sum())
        # print("grad_x clf:", gclf)
        # print("g(x): ", g(polar))
        # print("grad_u clf:", gclf @ g(polar))
        bfa = to_numpy(gclf @ g(polar))
        b = to_numpy(gclf @ f(polar) + 10 * self._clf(polar, state_goal))
        return bfa @ u + b

    def _cost(self, x, u):
        import cvxpy as cp # pip install cvxpy
        return cp.sum_squares(u)

    def control(self, x_torch, t):
        state_goal = self.state_goal
        import cvxpy as cp # pip install cvxpy
        x = x_torch
        uvar = cp.Variable(self.u_dim)
        uvar.value = np.zeros(self.u_dim)
        relax = cp.Variable(1)
        obj = cp.Minimize(self._cost(x, uvar) + 10*self._clc(x, state_goal, uvar, t))
        #constr = (self._clc(x, uvar) + relax <= 0)
        problem = cp.Problem(obj)#, [constr])
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
                 state_goal,
                 # simulation parameters
                 Kp_rho = 9,
                 Kp_alpha = -15,
                 Kp_beta = -3):
        self.state_goal = state_goal
        self.Kp_rho = Kp_rho
        self.Kp_alpha = Kp_alpha
        self.Kp_beta = Kp_beta

    def control(self, x, t):
        rho, alpha, beta = cartesian2polar(x, self.state_goal)
        Kp_rho   = self.Kp_rho
        Kp_alpha = self.Kp_alpha
        Kp_beta  = self.Kp_beta
        v = Kp_rho * rho
        w = Kp_alpha * alpha + Kp_beta * beta
        if alpha > math.pi / 2 or alpha < -math.pi / 2:
            v = -v
        return [v, w]

    def isconverged(self, x, state_goal):
        rho, alpha, beta = cartesian2polar(x, state_goal)
        return rho < 1e-3

class Visualizer:
    def __init__(self, state_goal, dt):
        self.state_goal = state_goal
        self.dt = dt
        self.state_start = None
        self.x_traj, self.y_traj = [], []


    def setStateCtrl(self, state, u, t=None, **kw):
        if t == 0:
            self.state_start = state
        plt.cla()
        x_start, y_start, theta_start = self.state_start
        plt.arrow(x_start, y_start, torch.cos(theta_start),
                    torch.sin(theta_start), color='r', width=0.1)
        x_goal, y_goal, theta_goal = self.state_goal
        plt.arrow(x_goal, y_goal, torch.cos(theta_goal),
                    torch.sin(theta_goal), color='g', width=0.1)
        x, y, theta = state
        self.x_traj.append(x)
        self.y_traj.append(y)
        plot_vehicle(x, y, theta, self.x_traj, self.y_traj, self.dt)

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

    state = state_start.copy()
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


def plot_vehicle(x, y, theta, x_traj, y_traj, dt):  # pragma: no cover
    # Corners of triangular vehicle when pointing to the right (0 radians)
    p1_i = torch.tensor([0.5, 0, 1]).T
    p2_i = torch.tensor([-0.5, 0.25, 1]).T
    p3_i = torch.tensor([-0.5, -0.25, 1]).T

    T = transformation_matrix(x, y, theta)
    p1 = torch.matmul(T, p1_i)
    p2 = torch.matmul(T, p2_i)
    p3 = torch.matmul(T, p3_i)

    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k-')
    plt.plot([p3[0], p1[0]], [p3[1], p1[1]], 'k-')

    plt.plot(x_traj, y_traj, 'b--')

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [sys.exit(0) if event.key == 'escape' else None])

    plt.xlim(-3.5, 22)
    plt.ylim(-3.5, 22)

    plt.pause(dt)


def transformation_matrix(x, y, theta):
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), x],
        [torch.sin(theta), torch.cos(theta), y],
        [0, 0, 1]
    ])


class Configs:
    @property
    def clf_polar(self):
        return dict(simulator=partial(
            lambda x, x_g, **kw: move_to_pose(
                x , x_g,
                dynamics=CartesianDynamics(),
                controller=ControllerCLF(
                    x_g,
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
                                                  x_g,
                                                  coordinate_converter = lambda x, x_g: (x),
                                                  dynamics=CartesianDynamics(),
                                                  clf = CLFCartesian()
                                              ))))

    @property
    def pid(self):
        return dict(simulator=partial(
            lambda x, x_g, **kw: move_to_pose(x, x_g,
                                      dynamics=CartesianDynamics(),
                                      controller=ControllerPID(x_g))))

    @property
    def sim_cartesian_clf(self):
        dt = 0.01
        return dict(simulator=partial(
            lambda x, x_g, **kw: sample_generator_trajectory(
                                      dynamics_model=CartesianDynamics(),
                                      D=200,
                                      controller=ControllerCLF(
                                          x_g,
                                          coordinate_converter = lambda x, x_g: x,
                                          dynamics = CartesianDynamics(),
                                          clf = CLFCartesian()
                                      ).control,
                                      visualizer=Visualizer(x_g, dt),
                                      x0=x,
                                      dt=dt)))




def sample_generator_trajectory(dynamics_model, D, dt=0.01, x0=None,
                                true_model=None,
                                controller=None,
                                visualizer=None):
    m = dynamics_model.ctrl_size
    n = dynamics_model.state_size
    U = torch.empty((D, m))
    X = torch.zeros((D+1, n))
    X[0, :] = torch.rand(n) if x0 is None else x0
    Xdot = torch.zeros((D, n))
    # Single trajectory
    dynamics_model.set_init_state(X[0, :])
    for t in range(D):
        U[t, :] = controller(X[t, :], t=t)
        visualizer.setStateCtrl(X[t, :], U[t, :], t=t)
        obs = dynamics_model.step(U[t, :], dt)
        Xdot[t, :] = obs['xdot'] # f(X[t, :]) + g(X[t, :]) @ U[t, :]
        X[t+1, :] = obs['x'] # normalize_state(X[t, :] + Xdot[t, :] * dt)
    return Xdot, X, U


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
    main(**getattr(Configs(), 'sim_cartesian_clf')) # 'pid', 'clf_polar' or 'clf_cartesian'

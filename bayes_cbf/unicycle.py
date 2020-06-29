import random
import math
from functools import partial
from collections import namedtuple

from scipy.special import erfinv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from bayes_cbf.misc import (t_hstack, store_args, DynamicsModel,
                            ZeroDynamicsModel, epsilon, to_numpy,
                            get_affine_terms, get_quadratic_terms)
from bayes_cbf.gp_algebra import (DeterministicGP,)
from bayes_cbf.cbc1 import RelDeg1Safety
from bayes_cbf.car.vis import CarWithObstacles
from bayes_cbf.sampling import sample_generator_trajectory, Visualizer
from bayes_cbf.plotting import plot_learned_2D_func, plot_results
from bayes_cbf.control_affine_model import ControlAffineRegressor
from bayes_cbf.controllers import (ControlCBFLearned, NamedAffineFunc,
                                   ConstraintPlotter)


class UnicycleDynamicsModel(DynamicsModel):
    """
    Ẋ     =     f(X)     +   g(X)     u

    [ ẋ  ] = [ 0 ] + [ cos(θ), 0 ] [ v ]
    [ ẏ  ]   [ 0 ]   [ sin(θ), 0 ] [ ω ]
    [ θ̇  ]   [ 0 ]   [ 0,      1 ]
    """
    def __init__(self):
        self.m = 2 # [v, ω]
        self.n = 3 # [x, y, θ]

    @property
    def ctrl_size(self):
        return self.m

    @property
    def state_size(self):
        return self.n

    def f_func(self, X_in):
        """
                [ 0   ]
         f(x) = [ 0   ]
                [ 0   ]
        """
        return X_in.new_zeros(X_in.shape)

    def g_func(self, X_in):
        """
                [ cos(θ), 0 ]
         g(x) = [ sin(θ), 0 ]
                [ 0,      1 ]
        """
        X = X_in.unsqueeze(0) if X_in.dim() <= 1 else X_in
        gX = torch.zeros((*X.shape, self.m))
        θ = X[..., 2]
        gX[..., 0, 0] = θ.cos()
        gX[..., 1, 0] = θ.cos()
        gX[..., 2, 1] = 1
        return gX.squeeze(0) if X_in.dim() <= 1 else gX

    def normalize_state(self, X_in):
        X_in[..., 2] = X_in[..., 2] % math.pi
        return X_in


class ObstacleCBF(RelDeg1Safety, NamedAffineFunc):
    """
    ∇h(x)ᵀf(x) + ∇h(x)ᵀg(x)u + γ_c h(x) > 0
    """
    @partial(store_args, skip=["model"])
    def __init__(self, model, center, radius, γ_c=1e-3, name="obstacle_cbf",
                 dtype=torch.get_default_dtype(),
                 max_unsafe_prob=0.01):
        self._model = model
        self.center = torch.tensor(center, dtype=dtype)
        self.radius = torch.tensor(radius, dtype=dtype)

    @property
    def model(self):
        return self._model

    @model.setter
    def set_model(self, model):
        self._model = model

    @property
    def gamma(self):
        return self.γ_c

    def cbf(self, x):
        return (((x[:2] - self.center)**2).sum(-1) - self.radius**2)

    value = cbf

    def grad_cbf(self, x):
        return torch.cat([2 * x[..., :2],
                          x.new_zeros(*x.shape[:-1], 1)], dim=-1)

    def __call__ (self, x, u):
        """
        A(x) u - b(x) < 0
        """
        return self.A(x) @ u - self.b(x)

    def A(self, x):
        return -self.grad_cbf(x) @ self.model.g_func(x)

    def b(self, x):
        return self.grad_cbf(x) @ self.model.f_func(x) + self.gamma * self.cbf(x)


class ControllerUnicycle(ControlCBFLearned):
    ground_truth = False,
    @store_args
    def __init__(self,
                 x_goal=[1, 1, math.pi/4],
                 quad_goal_cost=[[1.0, 0, 0],
                                 [0, 1.0, 0],
                                 [0, 0.0, 1]],
                 egreedy_scheme=[1, 0.01],
                 iterations=100,
                 max_train=200,
                 #gamma_length_scale_prior=[1/deg2rad(0.1), 1],
                 gamma_length_scale_prior=None,
                 true_model=UnicycleDynamicsModel(),
                 plotfile='plots/ctrl_cbf_learned_{suffix}.pdf',
                 dtype=torch.get_default_dtype(),
                 use_ground_truth_model=False,
                 x_dim=3,
                 u_dim=2,
                 train_every_n_steps=10,
                 mean_dynamics_model_class=ZeroDynamicsModel,
                 dt=0.001,
                 constraint_plotter_class=ConstraintPlotter,
                 cbc_class=ObstacleCBF,
                 obstacle_centers=[(0, 0)],
                 obstacle_radii=[0.5],
                 numSteps=1000,
                 ctrl_range=[[-10, math.pi],
                             [10, math.pi]]
    ):
        super().__init__(x_dim=x_dim,
                         u_dim=u_dim,
                         train_every_n_steps=train_every_n_steps,
                         mean_dynamics_model_class=mean_dynamics_model_class,
                         dt=dt,
                         constraint_plotter_class=constraint_plotter_class,
                         plotfile=plotfile,
                         ctrl_range=ctrl_range)
        if self.use_ground_truth_model:
            self.model = self.true_model
        else:
            self.model = ControlAffineRegressor(
                x_dim, u_dim,
                gamma_length_scale_prior=gamma_length_scale_prior)
        self.cbf2 = cbc_class(self.model, obstacle_centers[0],
                              obstacle_radii[0], dtype=dtype)
        self.ground_truth_cbf2 = cbc_class(self.true_model,
                                           obstacle_centers[0],
                                           obstacle_radii[0], dtype=dtype)
        self.x_goal = torch.tensor(x_goal)
        self.x_quad_goal_cost = torch.tensor(quad_goal_cost)

    def fit(self, Xtrain, Utrain, XdotError, training_iter=100):
        Xtrain[..., :2] = 0
        super().fit(Xtrain, Utrain, XdotError, training_iter=training_iter)

    def f_func_mean(self, Xtest_in):
        Xtest_in[..., :2] = 0
        return super().f_func_mean(Xtest_in)

    def f_func_knl(self, Xtest_in, Xtestp_in, grad_check=False):
        Xtest_in[..., :2] = 0
        Xtestp_in[..., :2] = 0
        return super().f_func_knl(Xtest_in, Xtestp_in)

    def fu_func_mean(self, Utest_in, Xtest_in):
        Xtest_in[..., :2] = 0
        return super().fu_func_mean(Utest_in, Xtest_in)

    def fu_func_knl(self, Utest_in, Xtest_in, Xtestp_in):
        Xtest_in[..., :2] = 0
        Xtestp_in[..., :2] = 0
        return super().fu_func_knl(Utest_in, Xtest_in, Xtestp_in)

    def covar_fu_f(self, Utest_in, Xtest_in, Xtestp_in):
        Xtest_in[..., :2] = 0
        Xtestp_in[..., :2] = 0
        return super().covar_fu_f(Utest_in, Xtest_in, Xtestp_in)


    def f_func(self, Xtest_in):
        Xtestp_in[..., :2] = 0
        return super().f_func(Xtest_in)

    def g_func(self, Xtest_in):
        Xtestp_in[..., :2] = 0
        return super().g_func(Xtest_in)


class UnicycleVisualizer(Visualizer):
    def __init__(self, centers, radii, x_goal):
        super().__init__()
        self.carworld = CarWithObstacles()
        for c, r in zip(centers, radii):
            self.carworld.addObstacle(c[0], c[1], r)

    def setStateCtrl(self, x, u, t=0):
        x_ = x[0]
        y_ = x[1]
        theta_ = x[2]
        self.carworld.setCarPose(x_, y_, theta_)
        self.carworld.show()


BBox = namedtuple('BBox', 'XMIN YMIN XMAX YMAX'.split())


class UnicycleVisualizerMatplotlib(Visualizer):
    @store_args
    def __init__(self, robotsize, obstacle_centers, obstacle_radii, x_goal):
        self.fig, self.axes = plt.subplots(2,2)
        self._bbox = BBox(-2.0, -2.0, 2.0, 2.0)
        self._latest_robot = None
        self._latest_history = None
        self._history_state = []
        self._history_ctrl = []
        self._init_drawing(self.axes[0,0])

    def _init_drawing(self, ax):
        self._add_obstacles(ax, self.obstacle_centers, self.obstacle_radii)
        obs_bbox = BBox(
            min((c[0]-r) for c, r in zip(self.obstacle_centers, self.obstacle_radii)),
            min((c[1]-r) for c, r in zip(self.obstacle_centers, self.obstacle_radii)),
            max((c[0]+r) for c, r in zip(self.obstacle_centers, self.obstacle_radii)),
            max((c[1]+r) for c, r in zip(self.obstacle_centers, self.obstacle_radii)))
        self._bbox = BBox(min(obs_bbox.XMIN, self._bbox.XMIN),
                          min(obs_bbox.YMIN, self._bbox.YMIN),
                          max(obs_bbox.XMAX, self._bbox.XMAX),
                          max(obs_bbox.YMAX, self._bbox.YMAX))
        ax.set_xlim(self._bbox.XMIN, self._bbox.XMAX)
        ax.set_ylim(self._bbox.YMIN, self._bbox.YMAX)
        ax.set_aspect('equal')
        self._add_goal(ax, self.x_goal, markersize=1)


    def _add_obstacles(self, ax, centers, radii):
        for c, r in zip(centers, radii):
            circle = Circle(c, radius=r, fill=True, color='r')
            ax.add_patch(circle)

    def _add_goal(self, ax, pos, markersize, color='g'):
        ax.plot(pos[0], pos[1], '*', markersize=1, color=color)

    def _add_robot(self, ax, pos, theta, robotsize):
        if self._latest_robot is not None:
            self._latest_robot.remove()
        dx = pos[0] + math.cos(theta)* robotsize
        dy = pos[1] + math.sin(theta)* robotsize

        arrow = FancyArrowPatch(pos, (dx, dy), mutation_scale=10)
        self._latest_robot = ax.add_patch(arrow)
        self._bbox = BBox(min(pos[0], self._bbox.XMIN),
                          min(pos[1], self._bbox.YMIN),
                          max(pos[0], self._bbox.XMAX),
                          max(pos[1], self._bbox.YMAX))
        ax.set_xlim(self._bbox.XMIN, self._bbox.XMAX)
        ax.set_ylim(self._bbox.YMIN, self._bbox.YMAX)

    def _add_history_state(self, ax):
        if self._latest_history is not None:
            for line in self._latest_history:
                line.remove()

        if len(self._history_state):
            hpos = np.asarray(self._history_state)
            self._latest_history = ax.plot(hpos[:, 0], hpos[:, 1], '-')

    def _plot_state_ctrl_history(self, axs):
        hctrl = np.array(self._history_ctrl)
        for i in range(hctrl.shape[-1]):
            ctrlax = axs[i]
            ctrlax.clear()
            ctrlax.set_title("u[{i}]".format(i=i))
            ctrlax.plot(hctrl[:, i])

    def setStateCtrl(self, x, u, t=0):
        self._add_robot(self.axes[0,0], x[:2], x[2], self.robotsize)
        self._add_history_state(self.axes[0,0])
        self._plot_state_ctrl_history(self.axes.flatten()[1:])
        self._history_state.append(to_numpy(x))
        self._history_ctrl.append(to_numpy(u))
        plt.draw()
        plt.pause(0.01)


class UnsafeControllerUnicycle(ControllerUnicycle):
    def control(self, xi, t=None):
        ui = self.unsafe_control(xi)
        print("unsafe control on {xi} is {ui}".format(xi=xi, ui=ui))
        return ui

def run_unicycle_control_learned(
        robotsize=0.2,
        obstacle_centers=[(0., 0.)],
        obstacle_radii=[0.5],
        x0=[-10.5, -10.5, math.pi/4],
        x_goal=[1., 1., math.pi/4],
        D=1000,
        controller_class=partial(ControllerUnicycle,
                                 mean_dynamics_model_class=partial(
                                     ZeroDynamicsModel, m=2, n=3)),
        visualizer_class=UnicycleVisualizerMatplotlib):
    """
    Run safe unicycle control with learned model
    """
    controller = controller_class(
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        x_goal=x_goal)
    return sample_generator_trajectory(
        dynamics_model=UnicycleDynamicsModel(),
        D=D,
        controller=controller.control,
        visualizer=visualizer_class(robotsize, obstacle_centers,
                                    obstacle_radii, x_goal),
        x0=x0)

def run_unicycle_control_unsafe():
    run_unicycle_control_learned(
        controller_class=partial(
            UnsafeControllerUnicycle,
            mean_dynamics_model_class=UnicycleDynamicsModel))


if __name__ == '__main__':
    #run_unicycle_control_unsafe()
    run_unicycle_control_learned()

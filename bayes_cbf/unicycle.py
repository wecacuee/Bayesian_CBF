import random
import math
from functools import partial

from scipy.special import erfinv
import torch
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
    def __init__(self, model, center, radius, γ_c=1.0, name="obstacle_cbf",
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
                 ctrl_range=torch.tensor([[-1, math.pi/10],
                                          [1, math.pi/10]])
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

    def unsafe_control(self, x):
        with torch.no_grad():
            x_g = self.x_goal
            P = self.x_quad_goal_cost
            R = torch.eye(self.u_dim)
            λ = 0.5
            fx = (self.dt * self.model.f_func(x)
                + self.dt * self.mean_dynamics_model.f_func(x))
            Gx = (self.dt * self.model.g_func(x.unsqueeze(0)).squeeze(0)
                + self.dt * self.mean_dynamics_model.g_func(x))
            # xp = x + fx + Gx @ u
            # (1-λ) (x + fx + Gx @ u - x_g)ᵀ P (x_g - (x + fx + Gx @ u)) + λ uᵀ R u
            # Quadratic term: uᵀ (λ R + (1-λ)GₓᵀPGₓ) u
            # Linear term   : - (2(1-λ)GₓP(x_g - x - fx)  )ᵀ u
            # Constant term : + (1-λ)(x_g-fx)ᵀP(x_g- x - fx)
            # Minima at u* = ((λ R + (1-λ)GₓᵀPGₓ))⁻¹ ((1-λ)GₓP(x_g - x - fx)  )


            # Quadratic term λ R + (1-λ)Gₓᵀ P Gₓ
            Q = λ * R + (1-λ) * Gx.T @ P @ Gx
            # Linear term - ((1-λ)Gₓ P(x_g - x - fx)  )ᵀ u
            c = (1-λ) * Gx.T @ P @ (x_g - x - fx)
            ugreedy = torch.solve(c.unsqueeze(0), Q).solution.reshape(-1)
            assert ugreedy.shape[-1] == self.u_dim
            return ugreedy

    def epsilon_greedy_unsafe_control(self, i, x, min_=-5., max_=5.):
        eps = epsilon(i, interpolate={0: self.egreedy_scheme[0],
                                      self.numSteps: self.egreedy_scheme[1]})
        randomact = torch.rand(self.u_dim) * (max_ - min_) + min_
        uegreedy = (randomact
                if (random.random() < eps)
                else self.unsafe_control(x))
        return torch.max(torch.min(uegreedy, max_), min_)


class UnicycleVisualizer(Visualizer):
    def __init__(self, centers, radii):
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

class UnicycleVisualizerMatplotlib(Visualizer):
    XMIN, YMIN, XMAX, YMAX = range(4)
    @store_args
    def __init__(self, robotsize, obstacle_centers, obstacle_radii):
        self.fig, self.axes = plt.subplots(1,1)
        self._range = [-1, -1, 1, 1]
        self._init_drawing()

    def _init_drawing(self):
        self._add_obstacles(self.obstacle_centers, self.obstacle_radii)
        self._range = [
            min((c[0]-r) for c, r in zip(self.obstacle_centers, self.obstacle_radii)),
            min((c[1]-r) for c, r in zip(self.obstacle_centers, self.obstacle_radii)),
            max((c[0]+r) for c, r in zip(self.obstacle_centers, self.obstacle_radii)),
            max((c[1]+r) for c, r in zip(self.obstacle_centers, self.obstacle_radii))]
        self.axes.set_xlim(self._range[self.XMIN], self._range[self.XMAX])
        self.axes.set_ylim(self._range[self.YMIN], self._range[self.YMAX])
        self.axes.set_aspect('equal')


    def _add_obstacles(self, centers, radii):
        for c, r in zip(centers, radii):
            circle = Circle(c, radius=r, fill=True, color='r')
            self.axes.add_patch(circle)

    def _add_robot(self, pos, theta, robotsize):
        dx = pos[0] + math.cos(theta)* robotsize
        dy = pos[1] + math.sin(theta)* robotsize

        arrow = FancyArrowPatch(pos, (dx, dy), mutation_scale=10)
        self.axes.add_patch(arrow)
        self._range[self.XMIN] = min(pos[0], self._range[self.XMIN])
        self._range[self.XMAX] = max(pos[0], self._range[self.XMAX])
        self._range[self.YMIN] = min(pos[1], self._range[self.YMIN])
        self._range[self.YMAX] = max(pos[1], self._range[self.YMAX])
        self.axes.set_xlim(self._range[self.XMIN], self._range[self.XMAX])
        self.axes.set_ylim(self._range[self.YMIN], self._range[self.YMAX])

    def setStateCtrl(self, x, u, t=0):
        self._add_robot(x[:2], x[2], self.robotsize)
        plt.draw()
        plt.pause(0.001)


def run_unicycle_control_learned(
        robotsize=2.0,
        obstacle_centers=[(0., 0.)],
        obstacle_radii=[2.0],
        x0=[-10., -10., math.pi/4],
        D=1000):
    """
    Run safe unicycle control with learned model
    """

    controller = ControllerUnicycle(mean_dynamics_model_class=ZeroDynamicsModel,
                                    obstacle_centers=obstacle_centers,
                                    obstacle_radii=obstacle_radii)
    return sample_generator_trajectory(
        dynamics_model=UnicycleDynamicsModel(),
        D=D,
        controller=controller.control,
        visualizer=UnicycleVisualizerMatplotlib(robotsize, obstacle_centers, obstacle_radii),
        x0=x0)

if __name__ == '__main__':
    run_unicycle_control_learned()

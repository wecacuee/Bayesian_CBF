import random
import math
from functools import partial

from scipy.special import erfinv
import torch

from bayes_cbf.misc import (t_hstack, store_args, DynamicsModel,
                            ZeroDynamicsModel, epsilon, to_numpy)
from bayes_cbf.gp_algebra import (DeterministicGP,)
from bayes_cbf.cbc2 import (get_affine_terms, get_quadratic_terms)
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


class ObstacleCBF(NamedAffineFunc):
    """
    ∇h(x)ᵀf(x) + ∇h(x)ᵀg(x)u + γ_c h(x) > 0
    """
    @store_args
    def __init__(self, model, center, radius, γ_c=1.0, name="obstacle_cbf",
                 dtype=torch.get_default_dtype(),
                 max_unsafe_prob=0.01):
        self.model = model
        self.center = torch.tensor(center, dtype=dtype)
        self.radius = torch.tensor(radius, dtype=dtype)

    def cbf(self, x):
        return (((x[:2] - self.center)**2).sum(-1) - self.radius**2)

    value = cbf

    def __call__ (self, x, u):
        """
        A(x) u - b(x) < 0
        """
        return self.A(x) @ u - self.b(x)

    def grad_cbf(self, x):
        return torch.cat([2 * x[..., :2],
                          x.new_zeros(*x.shape[:-1], 1)], dim=-1)

    def A(self, x):
        return -self._grad_h_col(x) @ self.model.g_func(x)

    def b(self, x):
        return self._grad_h_col(x) @ self.f_func(x) + γ_c * self.h_col(x)

    def cbc(self, u0):
        h_gp = DeterministicGP(lambda x: self.γ_c * self.cbf(x),
                               shape=(1,), name="h(x)")
        grad_h_gp = DeterministicGP(self.grad_cbf,
                                    shape=(self.model.state_size,),
                                    name="∇ h(x)")
        fu_gp = self.model.fu_func_gp(u0)
        cbc = grad_h_gp.t() @ fu_gp + h_gp
        return cbc

    def as_socp(self, i, xi, u0, convert_out=to_numpy):
        δ = self.max_unsafe_prob
        assert δ < 0.5 # Ask for at least more than 50% safety
        ratio = math.sqrt(2)*erfinv(1 - 2*δ)
        assert ratio > 0
        bfe, e = get_affine_terms(lambda u: self.cbc(u).mean(xi), u0)
        V, bfv, v = get_quadratic_terms(lambda u: self.cbc(u).knl(xi, xi), u0)
        m = u0.shape[-1]
        with torch.no_grad():
            # [1, u] Asq [1; u]
            Asq = torch.cat(
                (
                    torch.cat((torch.tensor([[v]]),         (bfv / 2).reshape(1, -1)), dim=-1),
                    torch.cat(((bfv / 2).reshape(-1, 1),    V), dim=-1)
                ),
                dim=-2)

            # [1, u] Asq [1; u] = |L[1; u]|_2 = |A [y; u] + b|_2
            A = torch.zeros((m + 1, m + 1))
            try:
                L = torch.cholesky(Asq) # (m+1) x (m+1)
            except RuntimeError as err:
                if "cholesky" in str(err) and "singular" in str(err):
                    diag_e, V = torch.symeig(Asq, eigenvectors=True)
                    L = torch.max(torch.diag(diag_e),
                                  torch.tensor(0.)).sqrt() @ V.t()
                else:
                    raise
            A[:, 1:] = L[:, 1:]
            b = L[:, 0] # (m+1)
            c = torch.zeros((m+1,))
            c[1:] = bfe
            # # We want to return in format?
            # (name, (A, b, c, d))
            # s.t. ||A[y, u] + b||_2 <= c'x + d
            return list(map(convert_out, (ratio * A, ratio * b, c, e)))



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
                 true_model=UnicycleDynamicsModel,
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
            self.model = self.true_model()
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
            # Linear term - (2λRu₀ + 2(1-λ)Gₓ P(x_g - fx)  )ᵀ u
            c = (λ * R + (1-λ) * Gx.T @ P @ (x_g - x - fx))
            return torch.solve(c, Q).solution.reshape(-1)

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


def run_unicycle_control_learned(
        obstacle_centers=[(0,0)],
        obstacle_radii=[0.5],
        x0=[-3, -3, math.pi/4],
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
        visualizer=UnicycleVisualizer(obstacle_centers, obstacle_radii),
        x0=x0)

if __name__ == '__main__':
    run_unicycle_control_learned()

import time
import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
from abc import ABC, abstractmethod, abstractproperty
from functools import partial
import math
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from bayes_cbf.misc import (store_args, ZeroDynamicsModel, epsilon, t_jac,
                            variable_required_grad, SumDynamicModels)
from bayes_cbf.plotting import plot_results, plot_learned_2D_func, plt_savefig_with_data
from bayes_cbf.cbc2 import cbc2_quadratic_terms, cbc2_gp, cbc2_safety_factor


class NamedFunc:
    def __init__(self, func, name):
        self.__name__ = name
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def identity(x):
    return x

def to_numpy(x):
    return x.detach().double().cpu().numpy() if isinstance(x, torch.Tensor) else x

def controller_socp_cvxopt(u0, linear_objective, socp_constraints):
    """
    Solve the optimization problem

    min_u   A u + b
       s.t. h₀ - (G u)₀ ≻ |h₁ - (Gu)₁ |₂

    u0: reference control signal
    linear
    """
    from cvxopt import matrix
    from cvxopt.solvers import socp
    m = u0.shape[0]
    c = matrix(linear_objective, (m, 1))
    Gq = []
    hq = []
    for name, (A, bfb, bfc, d) in socp_constraints:
        # (name, (A, b, c, d))
        # |Au + bfb| < bfc' u + d
        # s.t. ||Ax + b||_2 <= c'x + d
        # But || h₁ - G₁x ||₂ ≺ h₀ - g₀' x
        # G = [g₀'; G₁] = [-c'; -A]
        # h = [h₀; h₁] = [d; b]
        Gqi = matrix(0.0, (A.shape[0]+1, m))
        Gqi[0, :] = -bfc
        Gqi[1:, :] = -A
        Gq.append(Gqi)
        hqi = matrix(0.0, (A.shape[0]+1, 1))
        hqi[0, 0] = d.reshape(1,1)
        hqi[1:, 0] = bfb
        hq.append(hqi)

    sol = socp(c, Gq = Gq, hq = hq)
    if sol['status'] != 'optimal':
        print("{c}.T [y, u]\n".format(c=c)
              + "s.t. sq = {hq} - {Gq} [y, u]\n".format(hq=hq, Gq=Gq))
        raise ValueError("Infeasible problem: %s" % sol['status'])

    return np.asarray(sol['x']).astype(u0.dtype).reshape(-1)

def add_diag_const(Q, const=1.0):
    return torch.cat((torch.cat([Q,     torch.zeros(Q.shape[0], 1)], dim=0),
                      torch.cat([torch.zeros(1, Q.shape[1]), torch.tensor([[const]])], dim=0)),
                     dim=1)


class Controller(ABC):
    """
    Controller interface
    """
    needs_ground_truth = False
    @abstractmethod
    def control(self, xi, t=None):
        pass


class LQRController(Controller):
    def __init__(self, x_goal, model, Q, R, numSteps, dt):
        self.x_goal = x_goal
        self.model = model
        self.Q = to_numpy(Q)
        self.R = to_numpy(R)
        self.numSteps = numSteps
        self.dt = dt

    def control(self, x, t=None):
        from bdlqr.lqr import LinearSystem
        with variable_required_grad(x) as xp:
            A = t_jac(self.model.f_func(xp), xp)
        B = self.model.g_func(x)
        Q = self.Q
        R = self.R
        sys = LinearSystem(A=(np.eye(A.shape[-1]) + to_numpy(A) * self.dt),
                           B=to_numpy(B) * self.dt,
                           Q=Q,
                           s=np.zeros(Q.shape[-1]),
                           R=R,
                           z=np.zeros(R.shape[-1]),
                           Q_T=self.numSteps*Q,
                           s_T=np.zeros(Q.shape[-1]),
                           T=self.numSteps
        )
        xs, us = sys.solve(x, 1)

        with variable_required_grad(x) as xp:
            A = t_jac(self.model.f_func(xp)
                      + self.model.g_func(xp) @ torch.tensor(us[0], dtype=xp.dtype),
                      xp)
        sys = LinearSystem(A=(np.eye(A.shape[-1]) + to_numpy(A) * self.dt),
                           B=to_numpy(B) * self.dt,
                           Q=Q,
                           s=np.zeros(Q.shape[-1]),
                           R=R,
                           z=np.zeros(R.shape[-1]),
                           Q_T=self.numSteps*Q,
                           s_T=np.zeros(Q.shape[-1]),
                           T=self.numSteps
        )
        xs, us = sys.solve(x, 1)

        return torch.tensor(us[0], dtype=xp.dtype)


class GreedyController(Controller):
    def __init__(self, x_goal, model, Q, R, numSteps, dt):
        self.x_goal = x_goal
        self.model = model
        self.Q = Q
        self.R = R
        self.numSteps = numSteps
        self.dt = dt

    def control(self, x, t=None):
        u_dim = self.R.shape[-1]
        with torch.no_grad():
            x_g = self.x_goal
            P = self.Q
            R = self.R * self.dt
            λ = 0.5
            fx = (self.dt * self.model.f_func(x))
            Gx = (self.dt * self.model.g_func(x.unsqueeze(0)).squeeze(0))
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
            ugreedy = torch.solve(c.unsqueeze(-1), Q).solution.reshape(-1)
            assert ugreedy.shape[-1] == u_dim
            return ugreedy

class ConstraintPlotter:
    @store_args
    def __init__(self,
                 axes=None,
                 constraint_hists=[],
                 plotfile='plots/constraint_hists_{i}.pdf'):
        pass

    def plot_constraints(self, funcs, x, u, t=None):
        axs = self.axes
        nfuncs = len(funcs)
        if axs is None:
            nplots = nfuncs
            nrows = math.ceil(math.sqrt(nplots))
            ncols = math.ceil(nplots / nrows)
            shape = (nrows, ncols)
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


class ControlCBFLearned(Controller):
    needs_ground_truth = False
    @store_args
    def __init__(self,
                 x_dim=2,
                 u_dim=1,
                 model=None,
                 train_every_n_steps=10,
                 mean_dynamics_model_class=ZeroDynamicsModel,
                 dt=0.001,
                 constraint_plotter_class=ConstraintPlotter,
                 plotfile='plots/ctrl_cbf_learned_{suffix}.pdf',
                 ctrl_range=[-5., 5.],
                 x_goal=None,
                 x_quad_goal_cost=None,
                 u_quad_cost=None,
                 numSteps=1000,
                 unsafe_controller_class=GreedyController,
    ):
        self.Xtrain = []
        self.Utrain = []
        self.mean_dynamics_model = mean_dynamics_model_class()
        self.axes = [None, None]
        self.constraint_plotter = constraint_plotter_class(
            plotfile=plotfile.format(suffix='_constraints_{i}'))
        self._has_been_trained_once = False
        self.ctrl_range = torch.tensor(ctrl_range)
        self.x_goal = torch.tensor(x_goal)
        self.x_quad_goal_cost = torch.tensor(x_quad_goal_cost)
        self.u_quad_cost = torch.tensor(u_quad_cost)
        self.net_model = SumDynamicModels(self.model, self.mean_dynamics_model)
        self.unsafe_controller = unsafe_controller_class(
            self.x_goal, self.net_model, self.x_quad_goal_cost,
            self.u_quad_cost, numSteps, dt
        )


    def train(self):
        if not len(self.Xtrain):
            return
        assert len(self.Xtrain) == len(self.Utrain), "Call train when Xtrain and Utrain are balanced"
        Xtrain = torch.cat(self.Xtrain).reshape(-1, self.x_dim)
        Utrain = torch.cat(self.Utrain).reshape(-1, self.u_dim)
        XdotTrain = (Xtrain[1:, :] - Xtrain[:-1, :]) / self.dt
        XdotMean = self.mean_dynamics_model.f_func(Xtrain) + (
            self.mean_dynamics_model.g_func(Xtrain).bmm(Utrain.unsqueeze(-1)).squeeze(-1))
        XdotError = XdotTrain - XdotMean[:-1, :]
        LOG.info("Training model with datasize {}".format(XdotTrain.shape[0]))
        if XdotTrain.shape[0] > self.max_train:
            indices = torch.randint(XdotTrain.shape[0], (self.max_train,))
            train_data = Xtrain[indices, :], Utrain[indices, :], XdotError[indices, :],
        else:
            train_data = Xtrain[:-1, :], Utrain[:-1, :], XdotError

        self.model.fit(*train_data, training_iter=100)
        self._has_been_trained_once = True

        self.axes[0] = plot_learned_2D_func(Xtrain.detach().cpu().numpy(),
                                self.model.f_func,
                                self.true_model.f_func,
                                axtitle="f(x)[{i}]",
                                axs=self.axes[0])
        plt_savefig_with_data(
            self.axes[0].flatten()[0].figure,
            'plots/online_f_learned_vs_f_true_%d.pdf' % Xtrain.shape[0])
        self.axes[1] = plot_learned_2D_func(Xtrain.detach().cpu().numpy(),
                                lambda x: self.model.g_func(x)[..., 0],
                                lambda x: self.true_model.g_func(x)[..., 0],
                                axtitle="g(x)[{i}]",
                                axs=self.axes[1])
        plt_savefig_with_data(
            self.axes[1].flatten()[0].figure,
            'plots/online_g_learned_vs_g_true_%d.pdf' % Xtrain.shape[0])


    def _socp_objective(self, i, x, u0, yidx=0, convert_out=to_numpy):
        # s.t. ||[0, Q][y_1; y_2; u] - Q u_0||_2 <= [1, 0, 0] [y_1; y_2; u] + 0
        # s.t. ||R[y_1; y_2; u] + h||_2 <= a' [y_1; y_2; u] + b
        Q = torch.eye(self.u_dim)
        R = torch.zeros(self.u_dim, self.u_dim + 2)
        h = torch.zeros(self.u_dim)
        with torch.no_grad():
            R[:, :2] = 0
            R[:, 2:] = Q
            h = - Q @ u0
        a = torch.zeros((self.u_dim + 2,))
        a[yidx] = 1
        b = torch.tensor(0.)
        # s.t. ||R[y_1; y_2; u] + h||_2 <= a' [y_1; y_2; u] + b
        return list(map(convert_out, (R, h, a, b)))

    def _socp_safety(self, cbc2, x, u0,
                     safety_factor=None,
                     convert_out=to_numpy):
        """
        Var(CBC2) = Au² + b' u + c
        E(CBC2) = e' u + e
        """
        factor = safety_factor
        m = u0.shape[-1]

        (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(cbc2, x, u0)
        with torch.no_grad():
            # [1, u] Asq [1; u]
            Asq = torch.cat(
                (
                    torch.cat((torch.tensor([[v]]),         (bfv / 2).reshape(1, -1)), dim=-1),
                    torch.cat(((bfv / 2).reshape(-1, 1),    V), dim=-1)
                ),
                dim=-2)

            # [1, u] Asq [1; u] = |L[1; u]|_2 = |[0, A] [y_1; y_2; u] + b|_2
            A = torch.zeros((m + 1, m + 2))
            try:
                L = torch.cholesky(Asq) # (m+1) x (m+1)
            except RuntimeError as err:
                if "cholesky" in str(err) and "singular" in str(err):
                    diag_e, V = torch.symeig(Asq, eigenvectors=True)
                    L = torch.max(torch.diag(diag_e),
                                    torch.tensor(0.)).sqrt() @ V.t()
                else:
                    raise
            A[:, 2:] = L[:, 1:]
            b = L[:, 0] # (m+1)
            c = torch.zeros((m+2,))
            c[2:] = bfe
            # # We want to return in format?
            # (name, (A, b, c, d))
            # s.t. factor * ||A[y_1; y_2; u] + b||_2 <= c'x + d
            return list(map(convert_out, (factor * A, factor * b, c, e)))

    def _socp_constraints(self, i, x, u_ref, u_rand, convert_out=to_numpy):
        return [
            (r"$y_1 - \|R (u - u_{ref})\|>0$",
             self._socp_objective(i, x, u_ref, yidx=0, convert_out=convert_out)),
            (r"$y_2 - \|R (u - u_{rand})\|>0$",
             self._socp_objective(i, x, u_rand, yidx=1, convert_out=convert_out)),
            (r"$\mbox{CBC} > 0$", self._socp_safety(
                self.cbf2.cbc,
                x, u_ref,
                safety_factor=self.cbf2.safety_factor(),
                convert_out=convert_out))
        ]

    def _all_constraints(self, i, x, u0, u_rand, convert_out=to_numpy):
        if self.model.ground_truth:
            A = self.cbf2.A(x)
            b = self.cbf2.b(x)
            return [("CBC2det", (torch.tensor([[0.]]), A, -b))]
        else:
            return self._socp_constraints(i, x, u0, u_rand,
                                          convert_out=convert_out)

    def _plottables(self, i, x, u_ref, u_rand, y_uopt):
        def true_h(xp, y_uopt):
            val = - self.ground_truth_cbf2.cbf(xp)
            return val

        def true_cbc2(xp, y_uopt):
            val = ( self.ground_truth_cbf2.A(xp) @ y_uopt[2:]
                    - self.ground_truth_cbf2.b(xp))
            return val

        def scaled_variance(xp, y_uopt):
            up = y_uopt[2:]
            A, bfb, bfc, d = self._socp_safety(self.cbf2.cbc, xp, u_ref,
                                               safety_factor=self.cbf2.safety_factor(),
                                               convert_out=identity)
            return (A @ y_uopt + bfb).norm(p=2,dim=-1)

        def mean(xp, y_uopt):
            A, bfb, bfc, d = self._socp_safety(self.cbf2.cbc, xp, u_ref,
                                               safety_factor=self.cbf2.safety_factor(),
                                               convert_out=identity)
            return (bfc @ y_uopt + d)

        def ref_diff(xp, y_uopt):
            R, h, a, b = self._socp_objective(0, xp, u_ref,
                                              convert_out=identity)
            return (R @ y_uopt + h).norm(p=2, dim=-1)

        def true_dot_h(xp, y_uopt):
            return self.cbf2.grad_cbf(xp) @ (
                self.true_model.f_func(xp) +  self.true_model.g_func(xp) @ y_uopt[2:])

        return [
            NamedFunc(
                lambda _, y_u: (bfc @ y_u + d
                                - (A @ y_u + bfb).norm(p=2,dim=-1)) , name)
            for name, (A, bfb, bfc, d) in self._all_constraints(
                    i, x, u_ref, u_rand, convert_out=identity)
        ] + [
            NamedFunc(true_cbc2,
                      r"$ \mbox{CBC}_{true} < 0$"),
            NamedFunc(true_h, r"$-h_{true}(x) < 0$"),
            NamedFunc(true_dot_h, r"$\dot{h}_{true}(x)$"),
            NamedFunc(scaled_variance, r"$c(\tilde{p}_k)\|V(x,x)[1;u]\|$"),
            NamedFunc(mean, r"$e(x)^\top[1;u]>0$"),
            NamedFunc(ref_diff, r"$\|Q (u - u0)\|$")
        ]

    def control(self, xi, t=None):
        if (len(self.Xtrain) % int(self.train_every_n_steps) == 0):
            # train every n steps
            LOG.info("Training GP with dataset size {}".format(len(self.Xtrain)))
            self.train()

        tic = time.time()
        u_ref = self.epsilon_greedy_unsafe_control(t, xi,
                                                min_=self.ctrl_range[0],
                                                max_=self.ctrl_range[1])
        u_rand = self.random_unsafe_control(t, xi,
                                            min_=self.ctrl_range[0],
                                            max_=self.ctrl_range[1])
        if self._has_been_trained_once:
            y_uopt = controller_socp_cvxopt(
                np.hstack([[0., 0.], u_ref.detach().numpy()]),
                np.hstack([[1., 1.], np.zeros_like(u_ref)]),
                self._socp_constraints(t, xi, u_ref, u_rand,
                                       convert_out=to_numpy))
            y_uopt = torch.from_numpy(y_uopt).to(dtype=xi.dtype, device=xi.device)
            if len(self.Xtrain) > 20:
                self.constraint_plotter.plot_constraints(
                    self._plottables(t, xi, u_ref, u_rand, y_uopt),
                    xi, y_uopt)
            uopt = y_uopt[2:]
        else:
            uopt = u_ref

        # record the xi, ui pair
        self.Xtrain.append(xi.detach())
        self.Utrain.append(uopt.detach())
        assert len(self.Xtrain) == len(self.Utrain)
        print("Controller took {:.4f} sec".format(time.time()- tic))
        return uopt

    def unsafe_control(self, x):
        return self.unsafe_controller.control(x)

    def random_unsafe_control(self, i, x, min_=None, max_=None):
        randomact = torch.rand(self.u_dim) * (max_ - min_) + min_
        return randomact

    def epsilon_greedy_unsafe_control(self, i, x, min_=-5., max_=5.):
        eps = epsilon(i, interpolate={0: self.egreedy_scheme[0],
                                      self.numSteps: self.egreedy_scheme[1]})
        u0 = self.unsafe_control(x)
        randomact = self.random_unsafe_control(i, x, min_=min_, max_=max_)
        uegreedy = (u0 + randomact
                if (random.random() < eps)
                    else u0)
        return uegreedy # torch.max(torch.min(uegreedy, max_), min_)

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



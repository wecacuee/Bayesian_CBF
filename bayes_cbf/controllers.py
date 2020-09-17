from datetime import datetime
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
from torch.utils.tensorboard import SummaryWriter

from bayes_cbf.misc import (store_args, ZeroDynamicsModel, epsilon, t_jac,
                            variable_required_grad, SumDynamicModels, clip,
                            create_summary_writer)
from bayes_cbf.plotting import plot_results, plot_learned_2D_func, plt_savefig_with_data
from bayes_cbf.cbc2 import cbc2_quadratic_terms, cbc2_gp, cbc2_safety_factor
from bayes_cbf.ilqr import ILQR
from bayes_cbf.optimizers import controller_socp_cvxopt, InfeasibleProblemError


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
    def __init__(self, model, x_quad_goal_cost, u_quad_cost, x_goal, numSteps,
                 dt, ctrl_range):
        self.x_goal = x_goal
        self.model = model
        self.Q = to_numpy(x_quad_goal_cost)
        self.R = to_numpy(u_quad_cost)
        self.numSteps = numSteps
        self.dt = dt
        self.ctrl_range = ctrl_range

    def control(self, x, t=None):
        from bdlqr.lqr import LinearSystem
        with variable_required_grad(x) as xp:
            A = t_jac(self.model.f_func(xp), xp)
        B = to_numpy(self.model.g_func(x)) * self.dt
        Q = self.Q
        R = self.R
        s=(- Q @ to_numpy(self.x_goal))
        Q_T=Q
        s_T=(-  Q @ to_numpy(self.x_goal))
        z=np.zeros(R.shape[-1])
        T=(self.numSteps-t)
        sys = LinearSystem(A=(np.eye(A.shape[-1]) + to_numpy(A) * self.dt),
                           B=B,
                           Q=Q,
                           R=R,
                           s=s,
                           Q_T=Q_T,
                           s_T=s_T,
                           z=z,
                           T=T
        )
        xs, us = sys.solve(to_numpy(x), 1)

        with variable_required_grad(x) as xp:
            A = t_jac(self.model.f_func(xp)
                      + self.model.g_func(xp) @ torch.tensor(us[0], dtype=xp.dtype),
                      xp)
        sys = LinearSystem(A=(np.eye(A.shape[-1]) + to_numpy(A) * self.dt),
                           B=B,
                           Q=Q,
                           R=R,
                           s=s,
                           Q_T=Q_T,
                           s_T=s_T,
                           z=z,
                           T=T
        )
        xs, us = sys.solve(to_numpy(x), 1)

        return torch.tensor(us[0], dtype=xp.dtype)


class ILQRController(Controller):
    def __init__(self, x_goal, model, x_quad_goal_cost, u_quad_cost, numSteps,
                 dt, ctrl_range):
        self.x_goal = x_goal
        self.model = model
        self.Q = x_quad_goal_cost
        self.R = u_quad_cost
        self.numSteps = numSteps
        self.dt = dt
        self.ctrl_range = ctrl_range
        self.u_init = None

    def control(self, x, t=None):
        # https://github.com/locuslab/mpc.pytorch
        from mpc import mpc
        nx = x.shape[-1]
        nu = self.R.shape[-1]
        TIMESTEPS = self.numSteps
        ACTION_LOW = self.ctrl_range[0].repeat(TIMESTEPS, 1, 1)
        ACTION_HIGH = self.ctrl_range[1].repeat(TIMESTEPS, 1, 1)
        LQR_ITER = 5
        N_BATCH = 1
        u_init = self.u_init

        QR = torch.cat((
            torch.cat((self.Q, self.Q.new_zeros((nx, nu))), dim=-1),
            torch.cat((self.R.new_zeros((nu, nx)), self.R), dim=-1)),
                       dim=-2)
        qr = torch.cat((- self.Q @ self.x_goal,
                        self.Q.new_zeros(nu)), dim=-1)
        # T x B x nx+nu (linear component of cost)
        cost = mpc.QuadCost(QR.repeat(TIMESTEPS, N_BATCH, 1, 1),
                            qr.repeat(TIMESTEPS, N_BATCH, 1))

        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW,
                       u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2, n_batch=N_BATCH,
                       backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)

        # compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = ctrl(
            x.unsqueeze(0), cost, self.model)
        action = nominal_actions[0]  # take first planned action
        self.u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)
        return action


class ZeroController(Controller):
    def __init__(self, model, Q, R, x_goal, numSteps, dt, ctrl_range):
        self.u_dim = R.shape[-1]

    def control(self, x, t=None):
        return x.new_zeros(self.u_dim)


class GreedyController(Controller):
    def __init__(self, model, Q, R, x_goal, numSteps, dt, ctrl_range):
        self.x_goal = x_goal
        self.model = model
        self.Q = Q
        self.R = R
        self.numSteps = numSteps
        self.dt = dt
        self.ctrl_range = ctrl_range

    def clf(self, x):
        return (x - self.x_goal)**2

    def grad_clf(self, x):
        return 2*(x - self.x_goal)

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

class TensorboardPlotter:
    @store_args
    def __init__(self, run_dir='data/runs/', exp_tags=[], summary_writer=None):
        if summary_writer is None:
            self.summary_writer = create_summary_writer(run_dir, exp_tags)
        else:
            self.summary_writer = summary_writer

    def plot(self, funcs, x, u, t):
        for i, af in enumerate(funcs):
            self.summary_writer.add_scalar(af.__name__, af(x, u), t)


class ConstraintPlotter:
    @store_args
    def __init__(self,
                 axes=None,
                 constraint_hists=[],
                 plotfile='plots/constraint_hists_{i}.pdf'):
        pass

    def plot(self, funcs, x, u, t=None):
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


class Planner(ABC):
    @abstractmethod
    def plan(self, x, t):
        pass

class EpsilonGreedyController(ABC):
    def __init__(self, base_controller, u_dim, numSteps, egreedy_scheme, ctrl_range):
        self.base_controller = base_controller
        self.u_dim = u_dim
        self.numSteps = numSteps
        self.egreedy_scheme = egreedy_scheme
        self.ctrl_range = ctrl_range

    def random_unsafe_control(self, i, x, min_=None, max_=None):
        return randomact

    def control(self, x, t=None):
        min_, max_ = self.ctrl_range
        eps = epsilon(t, interpolate={0: self.egreedy_scheme[0],
                                      self.numSteps: self.egreedy_scheme[1]})
        u0 = self.base_controller.control(x, t=t)
        randomact = torch.rand(self.u_dim) * (max_ - min_) + min_
        uegreedy = (randomact
                if (random.random() < eps)
                    else u0)
        return clip(uegreedy, min_, max_)



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
                 constraint_plotter_class=TensorboardPlotter,
                 plots_dir='data/runs/',
                 ctrl_range=[-5., 5.],
                 x_goal=None,
                 x_quad_goal_cost=None,
                 u_quad_cost=None,
                 numSteps=1000,
                 unsafe_controller_class=LQRController,
                 planner_class=Planner,
                 cbfs=[],
                 ground_truth_cbfs=[],
                 exp_tags=[], # Experiment tags for tensorboard summary writer (if None)
                 exploration_controller_class=EpsilonGreedyController,
                 clf_class=None, # Control Lyapunov Controller class
                 egreedy_scheme=[1, 0.1], # Scheme for EpsilonGreedyController
                 summary_writer=None, # tensorboard summary writer
                 x0=None, # Initial starting point
                 ctrl_reg=1., # Q control regularizer
                 clf_relax_weight=10000. # ρ CLF relaxation weight compared to ctrl_reg
    ):
        self.Xtrain = []
        self.Utrain = []
        self.mean_dynamics_model = mean_dynamics_model_class()
        self.axes = [None, None]
        self.constraint_plotter = constraint_plotter_class(
            summary_writer=summary_writer,
            run_dir=plots_dir,
            exp_tags=exp_tags)
        self._has_been_trained_once = False
        self.ctrl_range = torch.tensor(ctrl_range)
        self.x_goal = torch.tensor(x_goal)
        self.x_quad_goal_cost = torch.tensor(x_quad_goal_cost)
        self.u_quad_cost = torch.tensor(u_quad_cost)
        self.net_model = SumDynamicModels(model, self.mean_dynamics_model)
        self.unsafe_controller = exploration_controller_class(
            unsafe_controller_class(
                self.net_model, self.x_quad_goal_cost,
                self.u_quad_cost, self.x_goal, numSteps, dt,
                self.ctrl_range
            ),
            u_dim,
            numSteps,
            egreedy_scheme,
            self.ctrl_range
        )
        self.cbfs = cbfs
        self.ground_truth_cbfs = ground_truth_cbfs
        self.planner = planner_class(torch.tensor(x0), self.x_goal, numSteps)
        self.clf = clf_class(self.net_model)


    def train(self):
        LOG.info("Training GP with dataset size {}".format(len(self.Xtrain)))
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

        if False:
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

    def _ctrl_cost_sqrt_Q(self):
        return torch.eye(self.u_dim) * math.sqrt(self.ctrl_reg)

    def _socp_objective(self, i, x, u0, yidx=0, convert_out=to_numpy, extravars=1):
        # s.t. ||[0, 1, Q][y_1; ρ; u] - Q u_0||_2 <= [1, 0, 0] [y_1; ρ; u] + 0
        # s.t. ||R[y_1; ρ; u] + h||_2 <= a' [y_1; ρ; u] + b
        assert yidx < extravars

        # R = [ 0, √λ,  0 ]
        #     [ 0, 0 , √Q ]
        # h = [0, - √Q u₀]
        # a = [1, 0, 0]
        # b = 0
        sqrt_Q = self._ctrl_cost_sqrt_Q()
        λ = self.clf_relax_weight
        assert extravars >= 2
        R = torch.zeros(self.u_dim + 1, self.u_dim + extravars)
        h = torch.zeros(self.u_dim + 1)
        with torch.no_grad():
            assert extravars >= 2
            R[0, 1] = math.sqrt(λ ) # for δ
            R[1:, extravars:] = sqrt_Q
            h[1:] = - sqrt_Q @ u0
        a = torch.zeros((self.u_dim + extravars,))
        a[yidx] = 1
        b = torch.tensor(0.)
        # s.t. ||R[y_1; ρ; u] + h||_2 <= a' [y_1; ρ; u] + b
        return list(map(convert_out, (R, h, a, b)))


    @staticmethod
    def convert_cbc_terms_to_socp_terms(bfe, e, V, bfv, v, extravars,
                                        testing=False):
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
            A = torch.zeros((m + 1, m + extravars))
            try:
                L = torch.cholesky(Asq) # (m+1) x (m+1)
            except RuntimeError as err:
                if "cholesky" in str(err) and "singular" in str(err):
                    if torch.allclose(v, torch.zeros((1,))) and torch.allclose(bfv, torch.zeros(bfv.shape[0])):
                        L = torch.zeros((m+1, m+1))
                        L[1:, 1:] = torch.cholesky(V)
                    else:
                        diag_e, U = torch.symeig(Asq, eigenvectors=True)
                        L = torch.max(torch.diag(diag_e),
                                        torch.tensor(0.)).sqrt() @ U.t()
                else:
                    raise

        if testing:
            np.testing.assert_allclose(L @ L.T, Asq, rtol=1e-2, atol=1e-3)

        A[:, extravars:] = L.T[:, 1:]
        bfb = L.T[:, 0] # (m+1)
        if testing:
            y_u0 = torch.cat((torch.zeros(extravars), u0))
            np.testing.assert_allclose(A @ y_u0 + bfb, L.T @ u0_hom)
            np.testing.assert_allclose(u0_hom @ Asq @ u0_hom, u0_hom @ L @ L.T @ u0_hom, rtol=1e-2, atol=1e-3)
            np.testing.assert_allclose(u0 @ V @ u0 + bfv @ u0 + v, (A @ y_u0 + bfb) @ (A @ y_u0 + bfb), rtol=1e-2, atol=1e-3)
        assert extravars >= 2, "I assumed atleast y and δ "
        bfc[extravars-1] = 1 # For delta the relaxation factor
        bfc[extravars:] = bfe # only affine terms need to be negative?
        d = e
        return A, bfb, bfc, d

    def _socp_stability(self, clc, x, u0, t, extravars=1, convert_out=to_numpy):
        """
        SOCP compatible representation of condition

        d clf(x) / dt + gamma * clf(x) < rho
        """
        # grad_clf(x) @ f(x) + grad_clf(x) @ g(x) @ u + gamma * clf(x) < 0
        # ||[0] [y_1; u] + [0]||_2 <= - [grad_clf(x) @ g(x)] u - grad_clf(x) @ ||f(x) - gamma * clf(x)
        m = u0.shape[-1]
        x_d = self.planner.plan(x, t)
        (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
            lambda u: clc(x_d, u), x, u0)
        A, bfb, bfc, d = self.convert_cbc_terms_to_socp_terms(
            bfe, e, V, bfv, v, extravars)
        # # We want to return in format?
        # (name, (A, b, c, d))
        # s.t. factor * ||A[y_1; u] + b||_2 <= c'u + d
        return list(map(convert_out, (A, bfb, bfc, d)))

    def _socp_safety(self, cbc2, x, u0,
                     safety_factor=None,
                     convert_out=to_numpy,
                     extravars=1):
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
            A = torch.zeros((m + 1, m + extravars))
            try:
                L = torch.cholesky(Asq) # (m+1) x (m+1)
            except RuntimeError as err:
                if "cholesky" in str(err) and "singular" in str(err):
                    diag_e, V = torch.symeig(Asq, eigenvectors=True)
                    L = torch.max(torch.diag(diag_e),
                                    torch.tensor(0.)).sqrt() @ V.t()
                else:
                    raise
            A[:, extravars:] = L[:, 1:]
            b = L[:, 0] # (m+1)
            c = torch.zeros((m+extravars,))
            c[extravars:] = bfe
            # # We want to return in format?
            # (name, (A, b, c, d))
            # s.t. factor * ||A[y_1; u] + b||_2 <= c'x + d
            return list(map(convert_out, (factor * A, factor * b, c, e)))

    def _socp_constraints(self, t, x, u_ref, convert_out=to_numpy, extravars=1):
        constraints =  [
            (#r"$y_1 - \|R (u - u_{ref})\|>0$",
                "Objective",
             self._socp_objective(t, x, u_ref, yidx=0, convert_out=convert_out,
                                  extravars=extravars)),
        ] +  [(#r"$\mbox{CBC}_%d > 0$" % i,
            "SafetyConstraint gt 0",
              self._socp_safety(
                  cbf.cbc,
                  x, u_ref,
                  safety_factor=cbf.safety_factor(),
                  convert_out=convert_out,
                  extravars=extravars))
             for i, cbf in enumerate(self.cbfs)]
        if self.clf is not None:
            constraints += [ (#r"$\mbox{CLF} < 0$",
                "StabilityConstraint gt 0",
                self._socp_stability( self.clf.clc, x, u_ref, t,
                                      convert_out=convert_out, extravars=extravars))
            ]
        return constraints

    def _plottables(self, t, x, u_ref, y_uopt, extravars=1):
        def true_h(gcbf, xp, y_uoptp):
            val = gcbf.cbf(xp)
            return val

        def true_cbc2(gcbf, xp, y_uopt):
            val = (- gcbf.A(xp) @ y_uopt[extravars:]
                    + gcbf.b(xp))
            return val

        def scaled_variance(cbf, xp, y_uopt):
            up = y_uopt[extravars:]
            A, bfb, bfc, d = self._socp_safety(cbf.cbc, xp, u_ref,
                                               safety_factor=cbf.safety_factor(),
                                               convert_out=identity,
                                               extravars=extravars)
            return (A @ y_uopt + bfb).norm(p=2,dim=-1)

        def mean(cbf, xp, y_uopt):
            A, bfb, bfc, d = self._socp_safety(
                cbf.cbc, xp, u_ref,
                safety_factor=cbf.safety_factor(),
                convert_out=identity,
                extravars=extravars)
            return (bfc @ y_uopt + d)

        def relax_cost(xp, y_uopt):
            R, h, a, b = self._socp_objective(t, xp, u_ref,
                                              yidx=0,
                                              convert_out=identity,
                                              extravars=extravars)
            return (R[0, 1] * y_uopt[1] + h[0]).abs()

        def ctrl_ref_cost(xp, y_uopt):
            R, h, a, b = self._socp_objective(t, xp, u_ref,
                                              yidx=0,
                                              convert_out=identity,
                                              extravars=extravars)
            return (R[1:, extravars:] @ y_uopt[extravars:] + h[1:]).norm(p=2, dim=-1)


        def total_cost(xp, y_uopt):
            A, bfb, bfc, d = self._socp_objective(t, xp, u_ref,
                                              yidx=0,
                                              convert_out=identity,
                                              extravars=extravars)
            return (A @ y_uopt + bfb).norm(p=2, dim=-1)


        def clf(xp, y_uopt):
            x_d = self.planner.plan(xp, t)
            return self.clf.clf(xp, x_d)

        def dot_clf(xp, y_uopt):
            x_d = self.planner.plan(xp, t)
            uopt = y_uopt[extravars:]
            return self.clf.grad_clf(xp, x_d) @ (
                self.net_model.f_func(xp)
                + self.net_model.g_func(xp) @ uopt)

        def expected_clc(xp, y_uopt):
            x_d = self.planner.plan(xp, t)
            uopt = y_uopt[extravars:]
            return self.clf.clc(x_d, uopt).mean(xp)

        def clc_socp(xp, y_uopt):
            (A, bfb, bfc, d) = self._socp_stability(self.clf.clc, x, u_ref, t,
                                                    convert_out=identity,
                                                    extravars=extravars)
            return bfc[extravars:] @ y_uopt[extravars:] + d

        def clc_relax(xp, y_uopt):
            (A, bfb, bfc, d) = self._socp_stability(self.clf.clc, x, u_ref, t,
                                                    convert_out=identity,
                                                    extravars=extravars)
            return bfc[extravars-1] * y_uopt[extravars-1]


        return [
            NamedFunc(
                lambda _, y_u, Ap=A, bfbp=bfb, bfcp=bfc, dp=d:
                (bfcp @ y_u + dp - (Ap @ y_u + bfbp).norm(p=2,dim=-1)) , name)
            for name, (A, bfb, bfc, d) in self._socp_constraints(
                    t, x, u_ref, convert_out=identity, extravars=extravars)
        ] + [
            NamedFunc(
                lambda _, y_u, bfcp=bfc, dp=d:
                (bfcp @ y_u + dp) , name + "_affine")
            for name, (A, bfb, bfc, d) in self._socp_constraints(
                    t, x, u_ref, convert_out=identity, extravars=extravars)
        ] + [
            NamedFunc(
                lambda _, y_u, Ap=A, bfbp=bfb:
                (Ap @ y_u + bfbp).norm(p=2,dim=-1), name + "_norm")
            for name, (A, bfb, bfc, d) in self._socp_constraints(
                    t, x, u_ref, convert_out=identity, extravars=extravars)
        ] + [
            NamedFunc(partial(true_cbc2, gcbf),
                      #r"$ \mbox{CBC}_{true} > 0$")
                      "CBCₜᵣᵤₑ")
            for gcbf in self.ground_truth_cbfs
        ] + [
            NamedFunc(partial(true_h, gcbf),
                      #r"$h_{true}(x)> 0$")
                      "hₜᵣᵤₑ")
            for gcbf in self.ground_truth_cbfs
        ] + [
            NamedFunc(partial(scaled_variance, cbf),
                      "ScaledVariance")
            for cbf in self.cbfs
        ] + [
            NamedFunc(partial(mean, cbf), "CBC")
            for cbf in self.cbfs
        ] + [
            NamedFunc(lambda x, u, val=u_ref_j: to_numpy(val), "uᵣₑ_%d" % j)
            for j, u_ref_j in enumerate(u_ref[:])
        ] + [
            NamedFunc(clf, "CLF"),
            NamedFunc(dot_clf, "CLFDot"),
            NamedFunc(relax_cost, "CostRelax"),
            NamedFunc(ctrl_ref_cost, "CostCtrl"),
            NamedFunc(total_cost, "CostTotal")
            # NamedFunc(expected_clc, "CLC"),
            # NamedFunc(clc_socp, "CLC_socp"),
            # NamedFunc(clc_relax, "CLC_Relax")
        ]

    def control(self, xi, t=None, extravars=2):
        if (len(self.Xtrain) > 0
            and len(self.Xtrain) % int(self.train_every_n_steps) == 0):
            # train every n steps
            pass # self.train()

        tic = time.time()
        u_ref = self.unsafe_control(xi, t=t)
        if True or self._has_been_trained_once:
            y_uopt_init = np.hstack([np.zeros(extravars), u_ref.detach().numpy()])
            assert extravars == 2, "I assumed extrvars to be y and δ"
            linear_obj = np.hstack([np.array([1., 0.]), np.zeros_like(u_ref)])
            try:
                y_uopt = controller_socp_cvxopt(
                    y_uopt_init,
                    linear_obj,
                    self._socp_constraints(t, xi, u_ref,
                                           convert_out=to_numpy, extravars=extravars))
            except InfeasibleProblemError as e:
                y_uopt = torch.from_numpy(y_uopt_init).to(dtype=xi.dtype,
                                                          device=xi.device)
                self.constraint_plotter.plot(
                    self._plottables(t, xi, u_ref, y_uopt, extravars=extravars),
                    xi, y_uopt, t)
                raise
            y_uopt = torch.from_numpy(y_uopt).to(dtype=xi.dtype, device=xi.device)
            if len(self.Xtrain) > 20:
                self.constraint_plotter.plot(
                    self._plottables(t, xi, u_ref, y_uopt, extravars=extravars),
                    xi, y_uopt, t)
            uopt = y_uopt[extravars:]
        else:
            uopt = u_ref

        # record the xi, ui pair
        self.Xtrain.append(xi.detach())
        self.Utrain.append(uopt.detach())
        assert len(self.Xtrain) == len(self.Utrain)
        print("Controller took {:.4f} sec".format(time.time()- tic))
        return uopt

    def unsafe_control(self, x, t=None):
        return self.unsafe_controller.control(x, t=t)

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



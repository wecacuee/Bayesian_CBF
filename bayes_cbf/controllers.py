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

from bayes_cbf.cbc2 import cbc2_quadratic_terms, cbc2_gp, cbc2_safety_factor
from bayes_cbf.gp_algebra import DeterministicGP
from bayes_cbf.ilqr import ILQR
from bayes_cbf.misc import (store_args, ZeroDynamicsModel, epsilon, t_jac,
                            variable_required_grad, clip,
                            create_summary_writer, DynamicsModel)
from bayes_cbf.optimizers import (optimizer_socp_cvxopt,
                                  InfeasibleProblemError, optimizer_socp_cvxpy,
                                  optimizer_qp_cvxpy)
from bayes_cbf.plotting import (plot_results, plot_learned_2D_func_from_data,
                                plt_savefig_with_data)
from bayes_cbf.planner import Planner


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
    return torch.cat(
        (torch.cat([Q,                       torch.zeros(Q.shape[0], 1)], dim=0),
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


class EpsilonGreedyController(ABC):
    def __init__(self, base_controller, u_dim, numSteps, egreedy_scheme, ctrl_range):
        self.base_controller = base_controller
        self.u_dim = u_dim
        self.numSteps = numSteps
        self.egreedy_scheme = egreedy_scheme
        self.ctrl_range = ctrl_range

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


class SumDynamicModels(DynamicsModel):
    def __init__(self, *models):
        assert len(models) >= 2
        self.models = models

    @property
    def ctrl_size(self):
        return self.models[0].ctrl_size

    @property
    def state_size(self):
        return self.models[0].state_size

    def f_func(self, x):
        return sum(m.f_func(x) for m in self.models)

    def g_func(self, x):
        return sum(m.g_func(x) for m in self.models)

    def fu_func_gp(self, u):
        fu_func = lambda m, x: m.f_func(x) + m.g_func(x) @ u
        return sum([
            (m.fu_func_gp(u)
             if hasattr(m, "fu_func_gp")
             else
             DeterministicGP(partial(fu_func, m), shape=(self.state_size,)))
            for m in self.models],
                   DeterministicGP(lambda x: 0, shape=(self.state_size,)))


class MeanAdjustedModel(SumDynamicModels):
    def __init__(self, x_dim, u_dim,
                 mean_dynamics_model_class,
                 model,
                 max_train=None,
                 train_every_n_steps=None,
                 enable_learning=None):
        self.Xtrain = []
        self.Utrain = []
        self.mean_dynamics_model = mean_dynamics_model_class()
        super().__init__(model, self.mean_dynamics_model)
        self._has_been_trained_once = False
        self.model = model
        self.max_train = max_train
        self.train_every_n_steps = train_every_n_steps
        self.enable_learning = enable_learning


    def _train(self):
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

    def fu_func_gp(self, u):
        return sum(
            [(m.fu_func_gp(u)
              if hasattr(m, 'fu_func_gp')
              else DeterministicGP(lambda x, m=m, u=u: m.forward(x, u), (self.state_size,)))
            for m in self.models],
            DeterministicGP(lambda x: 0, (self.state_size,))
        )



class SOCPController(Controller):
    def __init__(self, x_dim, u_dim, ctrl_reg, clf_relax_weight, net_model,
                 cbfs, clf, unsafe_controller,
                 summary_writer):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.ctrl_reg = ctrl_reg
        self.clf_relax_weight = clf_relax_weight
        self.net_model = net_model
        self.cbfs = cbfs
        self.clf = clf
        self.unsafe_controller = unsafe_controller
        self.summary_writer = summary_writer

    def _socp_objective(self, i, x, u0, yidx=0, extravars=None):
        # s.t. ||[0, 1, Q][y_1; ρ; u] - Q u_0||_2 <= [1, 0, 0] [y_1; ρ; u] + 0
        # s.t. ||R[y_1; ρ; u] + h||_2 <= a' [y_1; ρ; u] + b
        assert yidx < extravars

        # R = [ 0, √λ,  0 ]
        #     [ 0, 0 , √Q ]
        # h = [0, - √Q u₀]
        # a = [1, 0, 0]
        # b = 0
        sqrt_Q = torch.eye(self.u_dim) * math.sqrt(self.ctrl_reg)
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
        return (R, h, a, b)


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
        assert extravars >= 1, "I assumed atleast δ "
        bfc[extravars-1] = 1 # For delta the relaxation factor
        bfc[extravars:] = bfe # only affine terms need to be negative?
        d = e
        return A, bfb, bfc, d

    def _socp_stability(self, clc, t, x, u0, extravars=None):
        """
        SOCP compatible representation of condition

        d clf(x) / dt + gamma * clf(x) < rho
        """
        # grad_clf(x) @ f(x) + grad_clf(x) @ g(x) @ u + gamma * clf(x) < 0
        # ||[0] [y_1; u] + [0]||_2 <= - [grad_clf(x) @ g(x)] u - grad_clf(x) @ ||f(x) - gamma * clf(x)
        m = u0.shape[-1]
        (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
            lambda u: clc(t, u), x, u0)
        A, bfb, bfc, d = self.convert_cbc_terms_to_socp_terms(
            bfe, e, V, bfv, v, extravars)
        # # We want to return in format?
        # (name, (A, b, c, d))
        # s.t. factor * ||A[y_1; u] + b||_2 <= c'u + d
        return (A, bfb, bfc, d)

    def _socp_safety(self, cbc2, x, u0,
                     safety_factor=None,
                     extravars=None):
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
            return (factor * A, factor * b, c, e)

    def _named_socp_constraints(self, t, x, u_ref, convert_out=to_numpy, extravars=None):
        constraints = [
            ("Objective",
             list(map(to_numpy, self._socp_objective(t, x, u_ref, yidx=0,
                                                     extravars=extravars))))
        ] + [
            ("Safety_%d gt 0" % i,
             list(map(to_numpy, self._socp_safety(
                 cbf.cbc,
                 x, u_ref,
                 safety_factor=cbf.safety_factor(),
                 extravars=extravars))))
             for i, cbf in enumerate(self.cbfs)
        ]
        if self.clf is not None:
            constraints += [ ("Stability gt 0",
                              list(map(convert_out,
                                       self._socp_stability(
                                           self.clf.clc,
                                           t, x,
                                           u_ref,
                                           extravars=extravars))))
            ]
        return constraints

    def control(self, xi, t=None, extravars=2):

        tic = time.time()
        u_ref = self.unsafe_controller.control(xi, t=t)
        y_uopt_init = np.hstack([np.zeros(extravars), u_ref.detach().numpy()])
        assert extravars == 2, "I assumed extravars to be δ"
        linear_obj = np.hstack([np.array([1., 0]), np.zeros(u_ref.shape)])
        try:
            y_uopt = optimizer_socp_cvxpy(
                y_uopt_init,
                linear_obj,
                self._named_socp_constraints(t, xi, u_ref,
                                             convert_out=to_numpy, extravars=extravars))
        except InfeasibleProblemError as e:
            y_uopt = torch.from_numpy(y_uopt_init).to(dtype=xi.dtype,
                                                        device=xi.device)
            raise
        y_uopt_t = torch.from_numpy(y_uopt).to(dtype=xi.dtype, device=xi.device)
        uopt = y_uopt_t[extravars:]
        print("Controller step {0:d} took {1:.4f} sec".format(t, time.time()- tic))
        return uopt


class QPController(Controller):
    def __init__(self, x_dim, u_dim, ctrl_reg, clf_relax_weight, net_model,
                 cbfs, clf, unsafe_controller, summary_writer):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.ctrl_reg = ctrl_reg
        self.clf_relax_weight = clf_relax_weight
        self.net_model = net_model
        self.cbfs = cbfs
        self.clf = clf
        self.unsafe_controller = unsafe_controller
        self.summary_writer = summary_writer

    def _qp_stability(self, clc, t, x, u0, extravars=None):
        """
        SOCP compatible representation of condition

        d clf(x) / dt + gamma * clf(x) < rho
        """
        # grad_clf(x) @ f(x) + grad_clf(x) @ g(x) @ u + gamma * clf(x) < 0
        # ||[0] [y_1; u] + [0]||_2 <= - [grad_clf(x) @ g(x)] u - grad_clf(x) @ ||f(x) - gamma * clf(x)
        m = u0.shape[-1]
        (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
            lambda u: clc(t, u), x, u0)
        A, bfb, bfc, d = SOCPController.convert_cbc_terms_to_socp_terms(
            bfe, e, V, bfv, v, extravars)
        # # We want to return in format?
        # (name, (A, b, c, d))
        # s.t. factor * ||A[y_1; u] + b||_2 <= c'u + d
        return (bfc, d)


    def _plots(self, t, xi, y_uopt_t, extravars):
        uopt = y_uopt_t[extravars:]
        x_p = self.clf._planner.plan(t)
        self.summary_writer.add_scalar('QPController/plan0', x_p[0], t)
        self.summary_writer.add_scalar('QPController/plan1', x_p[1], t)
        self.summary_writer.add_scalar('QPController/plan2', x_p[2], t)
        self.summary_writer.add_scalar('QPController/clf', self.clf._clf(xi, x_p), t)
        self.summary_writer.add_scalar(
            'QPController/clf/dot',
            self.clf._dot_clf_gp(t, x_p, uopt).mean(xi), t)
        self.summary_writer.add_scalar(
            'QPController/clf/dotctrl',
            self.clf._grad_clf_x(xi, x_p) @ self.clf.model.g_func(xi) @ uopt, t)
        self.summary_writer.add_scalar('QPController/clc', self.clf.clc(t, uopt).mean(xi), t)
        self.summary_writer.add_scalar('QPController/ρ', y_uopt_t[0], t)

    def control(self, xi, t=None, extravars=1):
        tic = time.time()
        u_ref = self.unsafe_controller.control(xi, t=t)
        assert extravars == 1, "I assumed extravars to be δ"
        m = u_ref.shape[-1]
        A = np.zeros((extravars+m, extravars+m))
        sqrt_Q = np.eye(self.u_dim) * math.sqrt(self.ctrl_reg)
        A[0, 0] = math.sqrt(self.clf_relax_weight)
        A[extravars:, extravars:] = to_numpy(sqrt_Q)
        bfb = np.zeros((extravars+m,))
        (bfc, d) = list(map(to_numpy, self._qp_stability(
            self.clf.clc, t, xi, u_ref,
            extravars=extravars)))
        y_uopt_init = np.hstack([np.zeros(extravars), u_ref.detach().numpy()])
        y_uopt = optimizer_qp_cvxpy(
            y_uopt_init,
            (A, bfb),
            [('Safety', (bfc, d))])
        # assert bfc @ y_uopt + d >= 0
        y_uopt_t = torch.from_numpy(y_uopt).to(dtype=xi.dtype, device=xi.device)
        uopt = y_uopt_t[extravars:]
        print("Controller step {0:d} took {1:.4f} sec".format(t, time.time()- tic))

        self._plots(t, xi, y_uopt_t, extravars)
        return uopt


class ControlCBFLearned(Controller):
    needs_ground_truth = False
    @store_args
    def __init__(self,
                 x_dim=2,
                 u_dim=1,
                 model=None,
                 train_every_n_steps=10,
                 dt=0.001,
                 constraint_plotter_class=TensorboardPlotter,
                 plots_dir='data/runs/',
                 ctrl_range=[-5., 5.],
                 x_goal=None,
                 x_quad_goal_cost=None,
                 u_quad_cost=None,
                 numSteps=1000,
                 unsafe_controller_class=LQRController,
                 cbfs=[],
                 ground_truth_cbfs=[],
                 exp_tags=[], # Experiment tags for tensorboard summary writer (if None)
                 exploration_controller_class=EpsilonGreedyController,
                 clf_class=None, # Control Lyapunov Controller class
                 egreedy_scheme=[1, 0.1], # Scheme for EpsilonGreedyController
                 summary_writer=None, # tensorboard summary writer
                 x0=None, # Initial starting point
                 ctrl_reg=1., # Q control regularizer
                 clf_relax_weight=100., # ρ CLF relaxation weight compared to ctrl_reg
                 enable_learning=False, # Whether learning from data is enabled
                 mean_dynamics_model_class=None, # A prior known model (mean model)
                 max_train=None, # The sample size of max training dataset
                 controller_class=QPController, # SOCPController or QPController
                 planner_class=Planner, # A Planner that provides time parameterized trajectory
    ):
        self.axes = [None, None]
        self.summary_writer = summary_writer
        self.ctrl_range = torch.tensor(ctrl_range)
        self.x_goal = torch.tensor(x_goal)
        self.x_quad_goal_cost = torch.tensor(x_quad_goal_cost)
        self.u_quad_cost = torch.tensor(u_quad_cost)
        self.net_model = MeanAdjustedModel(x_dim, u_dim,
                                           mean_dynamics_model_class, model,
                                           max_train=max_train,
                                           train_every_n_steps=train_every_n_steps,
                                           enable_learning=enable_learning)
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
        self.clf = clf_class(self.net_model,
                             planner=planner_class(torch.tensor(x0),
                                                   self.x_goal, numSteps, dt))
        self._controller = controller_class(self.x_dim, self.u_dim,
                                            self.ctrl_reg,
                                            self.clf_relax_weight,
                                            self.net_model,
                                            self.cbfs, self.clf,
                                            self.unsafe_controller,
                                            self.summary_writer)

    def control(self, xi, t=None):
        uopt = self._controller.control(xi, t=t)
        self.net_model.train(xi, uopt)
        return uopt


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



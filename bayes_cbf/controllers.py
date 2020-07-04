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
                            variable_required_grad, SumDynamicModels, clip)
from bayes_cbf.plotting import plot_results, plot_learned_2D_func, plt_savefig_with_data
from bayes_cbf.cbc2 import cbc2_quadratic_terms, cbc2_gp, cbc2_safety_factor
from bayes_cbf.ilqr import ILQR


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


class GreedyController(Controller):
    def __init__(self, model, Q, R, x_goal, numSteps, dt, ctrl_range):
        self.x_goal = x_goal
        self.model = model
        self.Q = Q
        self.R = R
        self.numSteps = numSteps
        self.dt = dt
        self.ctrl_range = ctrl_range

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
                 unsafe_controller_class=LQRController,
                 cbfs=[],
                 ground_truth_cbfs=[]
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
        self.net_model = SumDynamicModels(model, self.mean_dynamics_model)
        self.unsafe_controller = unsafe_controller_class(
            self.net_model, self.x_quad_goal_cost,
            self.u_quad_cost, self.x_goal, numSteps, dt,
            torch.tensor(ctrl_range)
        )
        self.cbfs = cbfs
        self.ground_truth_cbfs = ground_truth_cbfs


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


    def _socp_objective(self, i, x, u0, yidx=0, convert_out=to_numpy, extravars=1):
        # s.t. ||[0, Q][y_1; y_2; u] - Q u_0||_2 <= [1, 0, 0] [y_1; y_2; u] + 0
        # s.t. ||R[y_1; y_2; u] + h||_2 <= a' [y_1; y_2; u] + b
        assert yidx < extravars
        Q = torch.eye(self.u_dim)
        R = torch.zeros(self.u_dim, self.u_dim + extravars)
        h = torch.zeros(self.u_dim)
        with torch.no_grad():
            R[:, :extravars] = 0
            R[:, extravars:] = Q
            h = - Q @ u0
        a = torch.zeros((self.u_dim + extravars,))
        a[yidx] = 1
        b = torch.tensor(0.)
        # s.t. ||R[y_1; y_2; u] + h||_2 <= a' [y_1; y_2; u] + b
        return list(map(convert_out, (R, h, a, b)))

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
        return [
            (r"$y_1 - \|R (u - u_{ref})\|>0$",
             self._socp_objective(t, x, u_ref, yidx=0, convert_out=convert_out,
                                  extravars=extravars)),
        ] + [(r"$\mbox{CBC}_%d > 0$" % i,
              self._socp_safety(
                  cbf.cbc,
                  x, u_ref,
                  safety_factor=cbf.safety_factor(),
                  convert_out=convert_out,
                  extravars=extravars))
             for i, cbf in enumerate(self.cbfs)]

    def _plottables(self, i, x, u_ref, y_uopt, extravars=1):
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
                                               convert_out=identity)
            return (A @ y_uopt + bfb).norm(p=2,dim=-1)

        def mean(cbf, xp, y_uopt):
            A, bfb, bfc, d = self._socp_safety(
                cbf.cbc, xp, u_ref,
                safety_factor=cbf.safety_factor(),
                convert_out=identity)
            return (bfc @ y_uopt + d)

        def ref_diff(xp, y_uopt):
            R, h, a, b = self._socp_objective(0, xp, u_ref,
                                              yidx=0,
                                              convert_out=identity)
            return (R @ y_uopt + h).norm(p=2, dim=-1)

        return [
            NamedFunc(
                lambda _, y_u: (bfc @ y_u + d
                                - (A @ y_u + bfb).norm(p=2,dim=-1)) , name)
            for name, (A, bfb, bfc, d) in self._socp_constraints(
                    i, x, u_ref, convert_out=identity, extravars=extravars)
        ] + [
            NamedFunc(partial(true_cbc2, gcbf),
                      r"$ \mbox{CBC}_{true} > 0$")
            for gcbf in self.ground_truth_cbfs
        ] + [
            NamedFunc(partial(true_h, gcbf), r"$h_{true}(x)> 0$")
            for gcbf in self.ground_truth_cbfs
        ] + [
            NamedFunc(scaled_variance, r"$c(\tilde{p}_k)\|V(x,x)[1;u]\|$")
            for cbf in self.cbfs
        ] + [
            NamedFunc(mean, r"$e(x)^\top[1;u]>0$")
            for cbf in self.cbfs
        ] + [
            NamedFunc(lambda x, u: to_numpy(u_ref), r"$u_{ref}$"),
            NamedFunc(ref_diff, r"$\|Q (u - u_{ref})\|$")
        ]

    def control(self, xi, t=None, extravars=1):
        if (len(self.Xtrain) > 0
            and len(self.Xtrain) % int(self.train_every_n_steps) == 0):
            # train every n steps
            LOG.info("Training GP with dataset size {}".format(len(self.Xtrain)))
            self.train()

        tic = time.time()
        u_ref = self.epsilon_greedy_unsafe_control(t, xi,
                                                min_=self.ctrl_range[0],
                                                max_=self.ctrl_range[1])
        y_uopt_init = np.hstack([np.zeros(extravars), u_ref.detach().numpy()])
        linear_obj = np.hstack([np.ones(extravars), np.zeros_like(u_ref)])
        y_uopt = controller_socp_cvxopt(
            y_uopt_init,
            linear_obj,
            self._socp_constraints(t, xi, u_ref,
                                    convert_out=to_numpy, extravars=extravars))
        y_uopt = torch.from_numpy(y_uopt).to(dtype=xi.dtype, device=xi.device)
        if len(self.Xtrain) > 20:
            self.constraint_plotter.plot_constraints(
                self._plottables(t, xi, u_ref, y_uopt, extravars=extravars),
                xi, y_uopt)
        uopt = y_uopt[extravars:]

        # record the xi, ui pair
        self.Xtrain.append(xi.detach())
        self.Utrain.append(uopt.detach())
        assert len(self.Xtrain) == len(self.Utrain)
        print("Controller took {:.4f} sec".format(time.time()- tic))
        return uopt

    def unsafe_control(self, x, t=None):
        return self.unsafe_controller.control(x, t=t)

    def random_unsafe_control(self, i, x, min_=None, max_=None):
        randomact = torch.rand(self.u_dim) * (max_ - min_) + min_
        return randomact

    def epsilon_greedy_unsafe_control(self, t, x, min_=-5., max_=5.):
        eps = epsilon(t, interpolate={0: self.egreedy_scheme[0],
                                      self.numSteps: self.egreedy_scheme[1]})
        u0 = self.unsafe_control(x, t=t)
        randomact = self.random_unsafe_control(t, x, min_=min_, max_=max_)
        uegreedy = (u0 + randomact
                if (random.random() < eps)
                    else u0)
        return clip(uegreedy, min_, max_)

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



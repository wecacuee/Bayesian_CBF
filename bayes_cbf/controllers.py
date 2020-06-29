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

from bayes_cbf.misc import store_args, ZeroDynamicsModel, epsilon
from bayes_cbf.plotting import plot_results, plot_learned_2D_func, plt_savefig_with_data
from bayes_cbf.cbc2 import cbc2_quadratic_terms, cbc2_gp


class NamedFunc:
    def __init__(self, func, name):
        self.__name__ = name
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
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
            shape = ((math.ceil(nplots / 2), 2) if nplots >= 2 else (nplots,))
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
                 train_every_n_steps=10,
                 mean_dynamics_model_class=ZeroDynamicsModel,
                 dt=0.001,
                 constraint_plotter_class=ConstraintPlotter,
                 plotfile='plots/ctrl_cbf_learned_{suffix}.pdf',
                 ctrl_range=[-5., 5.],
    ):
        self.Xtrain = []
        self.Utrain = []
        self.mean_dynamics_model = mean_dynamics_model_class(m=u_dim, n=x_dim)
        self.axes = [None, None]
        self.constraint_plotter = constraint_plotter_class(
            plotfile=plotfile.format(suffix='_constraints_{i}'))
        self._has_been_trained_once = False
        self.ctrl_range = torch.tensor(ctrl_range)


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
        if  self.x_dim == 2:
            #self.axes = axs = plot_results(np.arange(Utrain.shape[0]),
            #                   omega_vec=to_numpy(Xtrain[:, 0]),
            #                   theta_vec=to_numpy(Xtrain[:, 1]),
            #                   u_vec=to_numpy(Utrain[:, 0]),
            #                   axs=self.axes)
            #plt_savefig_with_data(
            #    axs[0,0].figure,
            #    'plots/pendulum_data_{}.pdf'.format(Xtrain.shape[0]))
            assert torch.all((Xtrain[:, 0] <= math.pi) & (-math.pi <= Xtrain[:, 0]))
        LOG.info("Training model with datasize {}".format(XdotTrain.shape[0]))
        if XdotTrain.shape[0] > self.max_train:
            indices = torch.randint(XdotTrain.shape[0], (self.max_train,))
            train_data = Xtrain[indices, :], Utrain[indices, :], XdotError[indices, :],
        else:
            train_data = Xtrain[:-1, :], Utrain[:-1, :], XdotError

        self.model.fit(*train_data, training_iter=100)
        self._has_been_trained_once = True

        if self.x_dim == 2:
            self.axes[0] = plot_learned_2D_func(Xtrain.detach().cpu().numpy(),
                                    self.model.f_func,
                                    self.true_model.f_func,
                                    axtitle="f(x)[{i}]",
                                    axs=self.axes[0])
            plt_savefig_with_data(
                self.axes[0].flatten()[0].figure,
                'plots/online_f_learned_vs_f_true_%d.pdf' % Xtrain.shape[0])
            self.axes[1] = plot_learned_2D_func(Xtrain.detach().cpu().numpy(),
                                    self.model.g_func,
                                    self.true_model.g_func,
                                    axtitle="g(x)[{i}]",
                                    axs=self.axes[1])
            plt_savefig_with_data(
                self.axes[1].flatten()[0].figure,
                'plots/online_g_learned_vs_g_true_%d.pdf' % Xtrain.shape[0])


    def _socp_objective(self, i, x, u0, convert_out=to_numpy):
        # s.t. ||[0, Q][y; u] - Q u_0||_2 <= [1, 0] [y; u] + 0
        # s.t. ||R[y; u] + h||_2 <= a' [y; u] + b
        Q = torch.eye(self.u_dim)
        R = torch.zeros(self.u_dim, self.u_dim + 1)
        h = torch.zeros(self.u_dim)
        with torch.no_grad():
            R[:, :1] = 0
            R[:, 1:] = Q
            h = - Q @ u0
        a = torch.zeros((self.u_dim + 1,))
        a[0] = 1
        b = torch.tensor(0.)
        # s.t. ||R[y, u] + h||_2 <= a' [y, u] + b
        return list(map(convert_out, (R, h, a, b)))

    def _socp_constraints(self, *args, **kw):
        return [(r"$y - \|R u\|>0$", self._socp_objective(*args, **kw)),
                (r"$\mathbf{e}(x)^\top u - \zeta - \frac{\rho}{1-\rho}\|V(x, x')u\|>0$", self.cbf2.as_socp(*args, **kw))]


    def _all_constraints(self, i, x, u0, convert_out=to_numpy):
        if self.model.ground_truth:
            A = self.cbf2.A(x)
            b = self.cbf2.b(x)
            return [("CBC2det", (torch.tensor([[0.]]), A, -b))]
        else:
            return self._socp_constraints(i, x, u0, convert_out=convert_out)

    def _plottables(self, i, x, y_u0):
        def true_h(xp, up):
            val = - self.ground_truth_cbf2.cbf(xp)
            return val

        def true_cbc2(xp, up):
            val = ( self.ground_truth_cbf2.A(xp) @ up[1:]
                    - self.ground_truth_cbf2.b(xp))
            return val
        return [
            NamedFunc(lambda _, y_u: (bfc @ y_u + d - (A @ y_u +
                                                       bfb).norm(p=2,dim=-1)) , name)
            for name, (A, bfb, bfc, d) in self._all_constraints(
                    i, x, y_u0, convert_out=lambda x: x)
        ] + [
            NamedFunc(true_cbc2,
                      r"$ \mathcal{L}_f h(x)^\top F(x) u - [h(x), \mathcal{L}_f h(x)] k_\alpha < 0$"),
            NamedFunc(true_h, r"$-h(x) < 0$"),
        ]

    def control(self, xi, t=None):
        if (len(self.Xtrain) % int(self.train_every_n_steps) == 0):
            # train every n steps
            LOG.info("Training GP with dataset size {}".format(len(self.Xtrain)))
            self.train()

        tic = time.time()
        u0 = self.epsilon_greedy_unsafe_control(t, xi,
                                                min_=self.ctrl_range[0],
                                                max_=self.ctrl_range[1])
        if self._has_been_trained_once:
            y_uopt = controller_socp_cvxopt(
                np.hstack([[1.], u0.detach().numpy()]),
                np.hstack([[1.], np.zeros_like(u0)]),
                self._socp_constraints(t, xi, u0, convert_out=to_numpy))
            y_uopt = torch.from_numpy(y_uopt).to(dtype=xi.dtype, device=xi.device)
            self.constraint_plotter.plot_constraints(
                self._plottables(t, xi, u0),
                xi, y_uopt)
            uopt = y_uopt[1:]
        else:
            uopt = u0

        # record the xi, ui pair
        self.Xtrain.append(xi.detach())
        self.Utrain.append(uopt.detach())
        assert len(self.Xtrain) == len(self.Utrain)
        print("Controller took {:.4f} sec".format(time.time()- tic))
        return uopt

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
            ugreedy = torch.solve(c.unsqueeze(-1), Q).solution.reshape(-1)
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



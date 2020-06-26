import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
from abc import ABC, abstractmethod, abstractproperty
from functools import partial
import math
import random

import numpy as np
import torch

from bayes_cbf.misc import store_args
from bayes_cbf.plotting import plot_results, plot_learned_2D_func, plt_savefig_with_data
from bayes_cbf.cbc2 import cbc2_quadratic_terms, cbc2_gp


class NamedFunc:
    def __init__(self, func, name):
        self.__name__ = name
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)



def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, solver=None,
                    initvals=None, **kwargs):
    import cvxopt
    from cvxopt import matrix
    #P = (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
    else:
        args.extend([None, None])
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    else:
        args.extend([None, None])
    args.extend([initvals, solver])
    solvers = cvxopt.solvers
    old_options = solvers.options.copy()
    solvers.options.update(kwargs)
    try:
        sol = cvxopt.solvers.qp(*args)
    except ValueError:
        return None
    finally:
        solvers.options.update(old_options)
    if 'optimal' not in sol['status']:
        return None
    return np.asarray(sol['x']).reshape((P.shape[1],))


def control_QP_cbf_clf(x,
                       ctrl_aff_constraints,
                       constraint_margin_weights=[]):
    """
    Args:
          A_cbfs: A tuple of CBF functions
          b_cbfs: A tuple of CBF functions
          constraint_margin_weights: Add a margin constant to the constraint
                                     that is maximized.

    """
    #import ipdb; ipdb.set_trace()
    clf_idx = 0
    A_total = np.vstack([af.A(x).detach().numpy()
                         for af in ctrl_aff_constraints])
    b_total = np.vstack([af.b(x).detach().numpy()
                         for af in ctrl_aff_constraints]).flatten()
    D_u = A_total.shape[1]
    N_const = A_total.shape[0]

    # u0 = l*g*sin(theta)
    # uopt = 0.1*g
    # contraints = A_total.dot(uopt) - b_total
    # assert contraints[0] <= 0
    # assert contraints[1] <= 0
    # assert contraints[2] <= 0


    # [A, I][ u ]
    #       [ ρ ] ≤ b for all constraints
    #
    # minimize
    #         [ u, ρ1, ρ2 ] [ 1,     0] [  u ]
    #                       [ 0,   100] [ ρ2 ]
    #         [A_cbf, 1] [ u, -ρ ] ≤ b_cbf
    #         [A_clf, 1] [ u, -ρ ] ≤ b_clf
    N_slack = len(constraint_margin_weights)
    A_total_rho = np.hstack(
        (A_total,
         np.vstack((-np.eye(N_slack),
                    np.zeros((N_const - N_slack, N_slack))))
        ))
    A = A_total
    P_rho = np.eye(D_u + N_slack)
    P_rho[D_u:, D_u:] = np.diag(constraint_margin_weights)
    q_rho = np.zeros(P_rho.shape[0])
    #u_rho_init = np.linalg.lstsq(A_total_rho, b_total - 1e-1, rcond=-1)[0]
    u_rho = cvxopt_solve_qp(P_rho.astype(np.float64),
                            q_rho.astype(np.float64),
                            G=A_total_rho.astype(np.float64),
                            h=b_total.astype(np.float64),
                            show_progress=False,
                            maxiters=1000)
    if u_rho is None:
        raise RuntimeError("""QP is infeasible
        minimize
        u_rhoᵀ {P_rho} u_rho
        s.t.
        {A_total_rho} u_rho ≤ {b_total}""".format(
            P_rho=P_rho,
            A_total_rho=A_total_rho, b_total=b_total))
    # Constraints should be satisfied
    constraint = A_total_rho @ u_rho - b_total
    assert np.all((constraint <= 1e-2) | (constraint / np.abs(b_total) <= 1e-2))
    return torch.from_numpy(u_rho[:D_u]).to(dtype=x.dtype)


def to_numpy(x):
    return x.detach().double().cpu().numpy() if isinstance(x, torch.Tensor) else x


def controller_qcqp(u0, objective, quadratic_constraints):
    import cvxpy as cp
    import gurobipy
    u = cp.Variable(u0.shape)
    cp_obj = objective(u)
    cp_qc = [qc(u) for qc in quadratic_constraints]
    prob = cp.Problem(cp.Minimize(cp_obj), cp_qc)
    prob.solve(solver=cp.GUROBI)
    return u.value

class InfeasibleOptimization(Exception):
    pass

def controller_qcqp_gurobi(u0, quad_objective, quadratic_constraints,
                           DisplayInterval=120,
                           OutputFlag=0,
                           NonConvex=2,
                           **kwargs):
    import gurobipy as gp
    from gurobipy import GRB
    # Create a new model
    m = gp.Model("controller_qcqp_gurobi")
    m.Params.OutputFlag = OutputFlag
    m.Params.DisplayInterval = DisplayInterval
    m.Params.NonConvex = NonConvex
    for k, v in kwargs.items():
        m.setParam(k, v)

    # Create variables
    u = m.addMVar(shape=u0.shape, vtype=GRB.CONTINUOUS, name="u")


    m.setMObjective(*quad_objective, sense=GRB.MINIMIZE)
    for i, (name, (Q, c, const)) in enumerate(quadratic_constraints):
        m.addMQConstr(Q, c, "<", -const, xQ_L=u, xQ_R=u, xc=u, name=name)
    m.optimize()
    if m.getAttr(GRB.Attr.Status) == GRB.OPTIMAL:
        return u.X
    elif m.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
        raise InfeasibleOptimization("Optimal value not found. Problem is infeasible.")
    elif m.getAttr(GRB.Attr.Status) == GRB.UNBOUNDED:
        raise InfeasibleOptimization("Optimal value not found. Problem is unbounded.")
    return u.X


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
    def control(self, xi, i=None):
        pass


class ControlCBFLearned(Controller):
    needs_ground_truth = False
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

        self._has_been_trained_once = True

    def egreedy(self, i):
        se, ee = map(math.log, self.egreedy_scheme)
        T = self.iterations
        return math.exp( i * (ee - se) / T )

    def _socp_safety(self, i, x, u0, convert_out=to_numpy):
        """
        Var(CBC2) = Au² + b' u + c
        E(CBC2) = e' u + e
        """
        δ = self.max_unsafe_prob
        assert δ < 0.5 # Ask for at least more than 50% safety
        ratio = np.sqrt((1-δ)/δ)
        m = self.u_dim

        (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
            self.cbf2.h2_col, self.cbf2.grad_h2_col, self.model, x, u0,
            k_α=self.cbf2.cbf_col_K_alpha)
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
        b = torch.tensor([[0.]])
        # s.t. ||R[y, u] + h||_2 <= a' [y, u] + b
        return list(map(convert_out, (R, h, a, b)))

    def _socp_constraints(self, *args, **kw):
        return [(r"$y - \|R u\|>0$", self._socp_objective(*args, **kw)),
                (r"$\mathbf{e}(x)^\top u - \zeta - \frac{\rho}{1-\rho}\|V(x, x')u\|>0$", self._socp_safety(*args, **kw))]


    def quadratic_constraints(self, i, x, u0, convert_out=to_numpy):
        if self.model.ground_truth:
            A = self.cbf2.A(x)
            b = self.cbf2.b(x)
            return [("CBC2det", (torch.tensor([[0.]]), A, -b))]
        else:
            return self._socp_constraints(i, x, u0, convert_out=convert_out)

    def plottables(self, i, x, y_u0):
        def true_h(xp, up):
            val = - self.ground_truth_cbf2.h2_col(xp)
            return val

        def true_cbc2(xp, up):
            val = ( self.ground_truth_cbf2.A(xp) @ up[1:]
                    - self.ground_truth_cbf2.b(xp))
            return val
        return [
            NamedFunc(lambda _, y_u: (bfc @ y_u + d - A @ y_u - bfb)[1:] , name)
            for name, (A, bfb, bfc, d) in self.quadratic_constraints(
                    i, x, y_u0, convert_out=lambda x: x)
        ] + [
            NamedFunc(true_cbc2,
                      r"$ \mathcal{L}_f h(x)^\top F(x) u - [h(x), \mathcal{L}_f h(x)] k_\alpha < 0$"),
            NamedFunc(true_h, r"$-h(x) < 0$")
        ]

    def control(self, xi, i=None):
        if (not self.model.ground_truth
            and len(self.Xtrain) % int(self.train_every_n_steps) == 0):
            # train every n steps
            LOG.info("Training GP with dataset size {}".format(len(self.Xtrain)))
            self.train()

        assert torch.all((xi[0] <= math.pi) & (-math.pi <= xi[0]))

        u0 = self.epsilon_greedy_unsafe_control(i, xi, min_=-5., max_=5.)
        if self.model.ground_truth or self._has_been_trained_once:
            y_uopt = controller_socp_cvxopt(
                np.hstack([[1.], u0.detach().numpy()]),
                np.hstack([[1.], np.zeros_like(u0)]),
                self._socp_constraints(i, xi, u0, convert_out=to_numpy))
            y_uopt = torch.from_numpy(y_uopt).to(dtype=xi.dtype, device=xi.device)
            self.constraint_plotter.plot_constraints(
                self.plottables(i, xi, u0),
                xi, y_uopt)
            uopt = y_uopt[1:]
        else:
            uopt = u0
        # record the xi, ui pair
        self.Xtrain.append(xi.detach())
        self.Utrain.append(uopt.detach())
        assert len(self.Xtrain) == len(self.Utrain)
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


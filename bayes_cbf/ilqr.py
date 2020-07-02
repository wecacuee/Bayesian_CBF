import math
import warnings
from itertools import starmap, repeat, zip_longest
from functools import partial
from collections import deque
from operator import attrgetter
from logging import getLogger, DEBUG, INFO, basicConfig
basicConfig()
LOG = getLogger(__name__)
LOG.setLevel(DEBUG)

import torch

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from bayes_cbf.misc import t_jac, variable_required_grad

class DiscreteDynamicsModel():
    def __init__(self, model, dt):
        self.dt = dt
        self.model = model

    def f(self, x, u, t=0):
        dx = self.model.f_func(x) + self.model.g_func(x) @ u
        return x + dx * self.dt

    def A(self, x, u, t=0):
        with variable_required_grad(x) as xp:
            dA = t_jac(self.f(xp, u, t=t), xp)
        return torch.eye(x.shape[-1]) + dA * self.dt

    def B(self, x, u, t=0):
        return self.model.g_func(x) * self.dt



def repeat_maybe_inf(a, T):
    return (repeat(a)
            if math.isinf(T)
            else [a] * int(T))

def affine_backpropagation(Q, s, R, z, A, B, P, o, γ=1):
    """
    minimizeᵤ ∑ₜ uₜRₜuₜ + 2 zₜᵀ uₜ + xₜQₜxₜ + 2 sₜᵀ xₜ
    s.t.          xₜ₊₁ = A xₜ₊₁ + B uₜ

    returns Pₜ, oₜᵀ, Kₜ, kₜ

    s.t.
    optimal uₜ = - Kₜxₜ - kₜ

    and

    Value function is quadratic
    xₜPₜxₜ + 2 oₜᵀ xₜ ≡ ∑ₖ₌ₜᵀ uₖRuₖ + 2 zₖᵀ uₖ + xₖQxₖ + 2 sₖᵀ xₖ
                        s.t. xₖ₊₁ = A xₖ₊₁ + B uₖ
    """
    # Solution:
    # (1) Kₜ = (R + BᵀPₜ₊₁B)⁻¹BᵀPₜ₊₁Aₜ
    # (2) Pₜ = Qₜ + AₜᵀPₜ₊₁Aₜ -     AₜᵀPₜ₊₁BKₜ
    # (3) oₜ = sₜ + Aₜᵀoₜ₊₁    - Kₜᵀ(zₜ + Bᵀoₜ₊₁)
    # (4) kₜ = (R + BᵀPₜ₊₁B)⁻¹(zₜ + Bᵀoₜ₊₁)

    # Eq(1)
    P = γ*P
    o = γ*o
    G = R + B.T @ (P) @ (B)
    K = torch.solve(B.T @ (P) @ (A), G)[0]
    # Eq(2)
    P_new = Q + A.T @ (P) @ (A) - A.T @ (P) @ (B) @ (K)
    # Eq(3)
    o_new = s + A.T @ (o) - K.T @ (z + B.T @ (o))
    # Eq(4)
    k = torch.solve((z + B.T @ (o)).unsqueeze(-1), G)[0].squeeze(-1)
    return P_new, o_new, K, k

class ILQR:
    def __init__(self, model, Q, R, x_goal, numSteps,
                 dt,
                 ctrl_range, max_iter=1000, ε=1e-6, γ=1,
                 lqr_iter=1):
        xD = Q.shape[-1]
        uD = R.shape[-1]
        self.model = DiscreteDynamicsModel(model, dt)
        self.Q = Q
        self.R = R
        self.x_goal = x_goal
        self.numSteps = numSteps
        self.ctrl_range = ctrl_range
        self.lqr_iter = lqr_iter

        self.s = self.goal_cost(Q, x_goal)

        self.Q_T = self.Q
        self.s_T = self.s
        self.z = R.new_zeros((uD,))

        self.max_iter = max_iter
        self.ϵ = ϵ
        self.γ = γ
        self._last_trajectory = None

    @staticmethod
    def goal_cost(Q, x_goal):
        return - Q @ x_goal

    def Qs_rev(self):
        return repeat_maybe_inf(self.Q, self.numSteps)

    def ss_rev(self):
        return repeat_maybe_inf(self.s, self.numSteps)

    def Rs_rev(self):
        return repeat_maybe_inf(self.R, self.numSteps)

    def zs_rev(self):
        return repeat_maybe_inf(self.z, self.numSteps)

    def A(self, x, u, t=None):
        return self.model.A(x, u, t=t)

    def B(self, x, u, t=None):
        return self.model.B(x, u, t=t)

    def _backward_pass(self, trajectory, t=0, return_val=False):
        xs, us = trajectory
        eff_backprop = int(min(self.numSteps - t, self.max_iter))
        Ps = deque([self.Q_T], eff_backprop)
        os = deque([self.s_T], eff_backprop)
        Ks = deque([], eff_backprop)
        ks = deque([], eff_backprop)
        # backward
        for t, Q, s, R, z, xt, ut in zip(reversed(range(eff_backprop)),
                                         self.Qs_rev(), self.ss_rev(),
                                         self.Rs_rev(), self.zs_rev(),
                                         xs, us):
            P_t, o_t, K_t, k_t = affine_backpropagation(
                Q, s, R, z, self.A(xt, ut), self.B(xt, ut), Ps[0], os[0], γ=self.γ)
            Ps.appendleft(P_t)
            os.appendleft(o_t)
            Ks.appendleft(K_t)
            ks.appendleft(k_t)

        # forward
        if math.isinf(self.numSteps):
            Ks = repeat(Ks[0])
            ks = repeat(ks[0])
        K_affines = [(Kt, kt) for Kt, kt in zip(Ks, ks)]
        P_quads = [(Pt, ot, ot.new_zeros((1,)))
              for Pt, ot in zip(Ps, os)]
        return ((K_affines, P_quads)
                if return_val
                else K_affines)

    def f(self, x, u, t):
        return self.model.f(x, u, t=0)

    def _forward_pass(self, x0, K_affines, traj_len=100):
        xs = [x0]
        us = []
        eff_traj_len = min(self.numSteps, traj_len)
        for t, (Kt, kt) in zip(range(eff_traj_len), K_affines):
            us.append(- Kt @ xs[t] - kt)
            xs.append(self.f(xs[t], us[t], t))

        assert len(us) == eff_traj_len
        assert len(xs[1:]) == eff_traj_len
        return (xs[1:], us)

    def control(self, x0, t=0, traj_len=100):
        lqr_iter = self.lqr_iter
        if self._last_trajectory is None:
            u0 = torch.ones(self.R.shape[-1])
            trajectory = (repeat(x0, self.numSteps - t),
                        repeat(u0, self.numSteps - t))
        else:
            trajectory = self._last_trajectory[-(self.numSteps -t):]
        K_affines = self._backward_pass(trajectory, t=t)
        for _ in range(lqr_iter):
            trajectory = self._forward_pass(x0, K_affines, traj_len=(self.numSteps - t))
            K_affines, P_quads = self._backward_pass(trajectory, t=t, return_val=True)
        self._last_trajectory = trajectory
        Q, s, c = P_quads[-1]
        LOG.debug("Final Value: {0}".format( x0 @ Q @ x0 + 2 * s @ x0 + c))
        xg = self.x_goal
        xdiff = (xg - x0)
        LOG.debug("Orig Value: {0}".format(xdiff @ Q @ xdiff))
        K0, k0 = K_affines[0]
        return - K0 @ x0 - k0

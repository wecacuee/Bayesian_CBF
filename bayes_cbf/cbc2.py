from abc import ABC, abstractproperty, abstractmethod
import math
import torch
from .gp_algebra import *
from .misc import to_numpy, get_affine_terms, get_quadratic_terms

def cbc2_quadratic_terms(cbc2, x, u):
    """
    cbc2.mean(x) ≥ √(1-δ)/δ cbc2.k(x,x')
    """
    # TODO: Too complicated and opaque. Try to find a way to simplify
    mean = lambda up: cbc2(up).mean(x)
    k_func = lambda up: cbc2(up).knl(x, x)

    #assert mean(u) > 0, 'cbf2 should be at least satisfied in expectation'
    mean_A, mean_b = get_affine_terms(mean, u)
    assert not torch.isnan(mean_A).any()
    assert not torch.isnan(mean_b).any()
    k_Q, k_p, k_r = get_quadratic_terms(k_func, u)
    assert not torch.isnan(k_Q).any()
    assert not torch.isnan(k_p).any()
    assert not torch.isnan(k_r).any()
    return (mean_A, mean_b), (k_Q, k_p, k_r), mean(u), k_func(u)


def cbc2_gp(h, grad_h, learned_model, utest, k_α):
    f_gp = learned_model.f_func_gp()
    fu_gp = learned_model.fu_func_gp(utest)
    h_gp = DeterministicGP(h, shape=(1,), name="h(x)")
    grad_h_gp = DeterministicGP(grad_h, shape=(learned_model.state_size,), name="∇ h(x)")
    L1h = grad_h_gp.t() @ f_gp
    L2h = GradientGP(L1h, x_shape=(learned_model.state_size,)).t() @ fu_gp
    return L2h + h_gp * k_α[0] + L1h * k_α[1]


def cbc2_safety_factor(δ):
    assert δ < 0.5 # Ask for at least more than 50% safety
    factor = math.sqrt((1-δ)/δ)
    assert factor > 1
    return factor


def cbc2_safety_socp(cbc2, x, u0,
                     max_unsafe_prob=0.01,
                     safety_factor=cbc2_safety_factor):
    """
    Var(CBC2) = Au² + b' u + c
    E(CBC2) = e' u + e
    """
    δ = max_unsafe_prob
    factor = safety_factor(δ)
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
        # s.t. factor * ||A[y, u] + b||_2 <= c'x + d
        return (factor * A, factor * b, c, e)

class RelDeg2Safety(ABC):
    @abstractproperty
    def k_alpha(self):
        pass
    @abstractproperty
    def model(self):
        pass

    @abstractmethod
    def cbf(self, x):
        pass

    @abstractmethod
    def grad_cbf(self, x):
        pass

    def cbc2(self, u0):
        return cbc2_gp(self.cbf, self.grad_cbf, self.model, u0, self.k_alpha)

    def as_socp(self, i, xi, u0, convert_out=to_numpy):
        return list(map(convert_out, cbc2_safety_socp(self.cbc2, xi, u0,
                        max_unsafe_prob=self.max_unsafe_prob)))


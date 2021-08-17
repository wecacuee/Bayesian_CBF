import torch

def max_kernel_xbounds(Xtest_grid, A, uBu, d2_kxxp_dx_dxp):
    return torch.max(torch.diag(A) * uBu * d2_kxxp_dx_dxp(Xtest_grid))

def maximal_extension(Xtest_grid):
    """
    Xtest_grid : *pts x d
    """
    *pts, d = Xtest_grid.shape
    ndim = Xtest_grid.dim
    return torch.max(
        torch.norm(Xtest_grid.reshape(*pts, [1]*(ndim-1), d)
                   - Xtest_grid.reshape([1]*(ndim-1), *pts, d)))

def L_f_x(state_dim, delta_L, max_knl_xbounds, max_extension, L_d2_kxxp_dx_dxp):
    """
    state_dim: scalar: State size
    delta_L: scalar: Probability of Lipschitz continuity
    max_knl_xbounds: n x n matrix: with each element max \sqrt( A_ii uBu d2_kxxp_dx_dxp_jj )
    max_extension: scalar: diameter of state space
    L_d2_kxxp_dx_dxp: n vector: Lipschitz constant of d2_kxxp_dx_dxp

    Returns n x n matrix of Lipschitz constant
    """
    n = state_dim
    kX = max_knl_xbounds
    r = max_extension
    return torch.sqrt(2*torch.log(2*n**2/delta_L)) * kX + 12*torch.sqrt(6*n) * max(kX, torch.sqrt(r*L_d2_kxxp_dx_dxp))

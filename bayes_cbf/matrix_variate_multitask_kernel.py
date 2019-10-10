#!/usr/bin/env python3

import torch

from gpytorch.kernels import MultitaskKernel, IndexKernel
from gpytorch.lazy import KroneckerProductLazyTensor, BlockDiagLazyTensor, InterpolatedLazyTensor, lazify

# https://github.com/yulkang/pylabyk/blob/master/numpytorch.py
# Apache licence https://github.com/yulkang/pylabyk/blob/master/LICENSE
def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))

# https://github.com/yulkang/pylabyk/blob/master/numpytorch.py
# Apache licence https://github.com/yulkang/pylabyk/blob/master/LICENSE
def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )


def _eval_covar_matrix(covar_factor, log_var):
    return covar_factor.matmul(covar_factor.transpose(-1, -2)) + log_var.exp().diag()


class MatrixVariateIndexKernel(IndexKernel):
    """
    Wraps IndexKernel to represent
    https://en.wikipedia.org/wiki/Matrix_normal_distribution

    """
    def __init__(self, idxkernel_rows, idxkernel_cols):
        super(MatrixVariateIndexKernel, self).__init__(num_tasks=1)
        self.U = idxkernel_rows
        self.V = idxkernel_cols

    @property
    def covar_matrix(self):
        U = self.U.covar_matrix
        V = self.V.covar_matrix
        return KroneckerProductLazyTensor(V, U)

    def forward(self, i1, i2, **params):
        assert i1.dtype in (torch.int64, torch.int32)
        assert i2.dtype in (torch.int64, torch.int32)
        covar_matrix = self.covar_matrix
        res = InterpolatedLazyTensor(base_lazy_tensor=covar_matrix, left_interp_indices=i1, right_interp_indices=i2)
        return res


def test_MatrixVariateIndexKernel(n=2, m=3):
    mvk = MatrixVariateIndexKernel(IndexKernel(n), IndexKernel(m))
    x = torch.ranint(n*m, size=(10,))
    mvk(x,x).evaluate()


class MatrixVariateKernel(MultitaskKernel):
    """
    Kernel supporting Kronecker style matrix variate Gaussian processes (where every
    data point is evaluated at every task).

    Given a base covariance module to be used for the data, :math:`K_{XX}`,
    this kernel computes a task kernel of specified size :math:`K_{TT}` and
    returns :math:`K = K_{TT} \otimes K_{XX}`. as an
    :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.

    Args:
        task_covar_module (:obj:`gpytorch.kernels.IndexKernel`):
            Kernel to use as the task kernel
        data_covar_module (:obj:`gpytorch.kernels.Kernel`):
            Kernel to use as the data kernel.
        num_tasks (int):
            Number of tasks
        batch_size (int, optional):
            Set if the MultitaskKernel is operating on batches of data (and you
            want different parameters for each batch)
        rank (int):
            Rank of index kernel to use for task covariance matrix.
        task_covar_prior (:obj:`gpytorch.priors.Prior`):
            Prior to use for task kernel. See :class:`gpytorch.kernels.IndexKernel` for details.
    """

    def __init__(self, task_covar_module, data_covar_module, decoder, num_tasks, rank=1, task_covar_prior=None, **kwargs):
        """
        """
        super(MultitaskKernel, self).__init__(**kwargs)
        self.decoder = decoder
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks


class HetergeneousMatrixVariateKernel(MatrixVariateKernel):
    def mask_dependent_covar(self, M1s, U1, M2s, U2, covar_xx):
        # Assume M1s, M2s sorted descending
        B = M1s.shape[:-1]
        M1s = M1s[..., 0]
        idxs1 = torch.nonzero(M1s - torch.ones_like(M1s))
        idxend1 = torch.min(idxs1) if idxs1.numel() else M1s.size(-1)
        # assume sorted
        assert (M1s[..., idxend1:] == 0).all()
        U1s = U1[..., :idxend1, :]

        M2s = M2s[..., 0]
        idxs2 = torch.nonzero(M2s - torch.ones_like(M2s))
        idxend2 = torch.min(idxs2) if idxs2.numel() else M2s.size(-1)
        # assume sorted
        assert (M2s[..., idxend2:] == 0).all()
        U2s = U2[..., :idxend2, :]

        import pdb; pdb.set_trace()
        H1 = BlockDiagLazyTensor(U1s)
        H2 = BlockDiagLazyTensor(U2s)

        Kxx = covar_xx
        # If M1, M2 = (1, 1)
        #    H₁ᵀ [ K ⊗ B ] H₂ ⊗ A
        Kij_xx_11 = H1 @ KroneckerProductLazyTensor(Kxx, B) @ H2 @ A

        # elif M1, M2 = (1, 0)
        #    H₁ᵀ [ K ⊗ B ] ⊗ A
        Kij_xx_12 = H1 @ KroneckerProductLazyTensor(Kxx, B) @ H2 @ A

        # elif M1, M2 = (0, 1)
        #    [ K ⊗ B ] H₂ ⊗ A
        Kij_xx_21 = Kij_12.t()
        # else M1, M2 = (0, 0)
        #    [ K ⊗ B ] ⊗ A
        Kij_xx_22 = KronKroneckerProductLazyTensor(KroneckerProductLazyTensor(Kxx, B), A)
        covar_i = self.task_covar_module.covar_matrix
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        return torch.cat([torch.cat([Kij_xx_11, Kij_xx_12], dim=1),
                          torch.cat([Kij_xx_21, Kij_xx_22], dim=1)], dim=0)


    def forward(self, mxu1, mxu2, diag=False, last_dim_is_batch=False, **params):
        M1, X1, U1 = self.decoder.decode(mxu1)
        M2, X2, U2 = self.decoder.decode(mxu2)

        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        #covar_i = self.mask_dependent_covar(self, M1, U1, M2, U2)
        covar_x = lazify(self.data_covar_module.forward(X1, X2, **params))
        res = self.mask_dependent_covar(M1, U1, M2, U2, covar_x)
        return res.diag() if diag else res

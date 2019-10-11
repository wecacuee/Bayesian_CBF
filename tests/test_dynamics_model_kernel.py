
import numpy as np
import torch
import gpytorch.settings as gpsettings
import pytest

from scipy.linalg import block_diag
from gpytorch.kernels import Kernel

from bayes_cbf.matrix_variate_multitask_kernel import HetergeneousMatrixVariateKernel, MatrixVariateIndexKernel
from bayes_cbf.matrix_variate_multitask_model import CatEncoder


class ConstantIndexKernel(Kernel):
    def __init__(self, A):
        super().__init__()
        self.A = A

    @property
    def raw_var(self):
        return self.A

    @property
    def covar_matrix(self):
        return self.A

    def forward(self, i, j):
        return A[i, j]


class DataKernel(Kernel):
    def forward(self, x1, x2):
        dist = ((x1[:, None, :] - x2[None, :, :])**2).sum(-1)
        return np.exp(-dist)


def kernel_train(H, A, B, X, kerX):
    K = np.kron(H @ np.kron(kerX(X, X), B) @ H.T, A)
    return K


def rand_psd_matrix(n):
    Asqrt = np.random.rand(n,n)
    A = Asqrt.T @ Asqrt + np.diag(np.abs(np.random.rand(n)))
    return A


def encode_from_XU_numpy(Xtrain, Utrain):
    Mtrain = np.ones((Xtrain.shape[0], 1))
    UHtrain = np.concatenate([Mtrain, Utrain], axis=1)
    return CatEncoder.from_data(Mtrain, Xtrain, UHtrain)


def test_dynamics_model_kernel():
    chosen_seed = np.random.randint(10000)
    print(chosen_seed)
    D = 3
    n = 1
    m = 2
    U = np.random.rand(D, m)
    X = np.random.rand(D, n)

    A = rand_psd_matrix(n)
    B = rand_psd_matrix(1 + m)

    UH = np.concatenate((np.ones((D, 1)), U), axis=1)
    H = block_diag(*UH[:, None, :])
    Kexp = kernel_train(H, A, B, X, DataKernel().forward)

    decoder, MXU = encode_from_XU_numpy(X, U)
    MXUtorch = torch.from_numpy(MXU).float()
    with gpsettings.debug(True):
        Kgot = HetergeneousMatrixVariateKernel(
            task_covar_module = MatrixVariateIndexKernel(
                ConstantIndexKernel(torch.from_numpy(A).float()),
                ConstantIndexKernel(torch.from_numpy(B).float())),
            data_covar_module = DataKernel(),
            decoder = decoder
        )(MXUtorch, MXUtorch)
        Kgot_np = Kgot.evaluate().detach().numpy()

    assert Kgot_np == pytest.approx(Kexp)


if __name__ == '__main__':
    test_dynamics_model_kernel()


import numpy as np
import torch
import gpytorch.settings as gpsettings
import pytest

from scipy.linalg import block_diag
from gpytorch.kernels import Kernel

from bayes_cbf.matrix_variate_multitask_kernel import HetergeneousMatrixVariateKernel, MatrixVariateIndexKernel
from bayes_cbf.control_affine_model import CatEncoder


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
        return self.A[i, j]


class DataKernel(Kernel):
    def forward(self, x1, x2):
        dist = ((x1[:, None, :] - x2[None, :, :])**2).sum(-1)
        return np.exp(-dist)


def kernel_train(H, A, B, X, kerX):
    K = np.kron(H @ np.kron(kerX(X, X), B) @ H.T, A)
    return K


def kernel_test(H, A, B, X, kerX):
    return np.kron( np.kron(kerX(X, X), B), A)


def kernel_train_test(Htrain, A, B, Xtrain, Xtest, kerX):
    k_train_test = np.kron(Htrain @ np.kron(kerX(Xtrain, Xtest), B), A)
    return np.vstack((
        np.hstack((kernel_train(Htrain, A, B, Xtrain, kerX), k_train_test)),
        np.hstack((k_train_test.T, kernel_test(Htrain, A, B, Xtest, kerX)))))


def rand_psd_matrix(n):
    Asqrt = np.random.rand(n,n)
    A = Asqrt.T @ Asqrt + np.diag(np.abs(np.random.rand(n)))
    return A


def encode_from_XU_numpy(Xtrain, Utrain, M=1):
    Mtrain = M*np.ones((Xtrain.shape[0], 1))
    if M == 1:
        assert Utrain is not None
    else:
        Utrain = np.zeros((Xtrain.shape[0], Utrain.shape[-1]))
    UHtrain = np.concatenate([Mtrain, Utrain], axis=1)
    return CatEncoder.from_data(Mtrain, Xtrain, UHtrain)


def test_dynamics_model_kernel():
    chosen_seed = np.random.randint(10000)
    print(chosen_seed)
    D = 5
    n = 1
    m = 2
    U = np.random.rand(D, m)
    X = np.random.rand(D, n)
    Xtest = np.random.rand(1, n)

    A = rand_psd_matrix(n)
    B = rand_psd_matrix(1 + m)

    UH = np.concatenate((np.ones((D, 1)), U), axis=1)
    H = block_diag(*UH[:, None, :])
    Kexp = kernel_train(H, A, B, X, DataKernel().forward)

    decoder, MXU = encode_from_XU_numpy(X, U)
    MXUtorch = torch.from_numpy(MXU).float()
    HMVKer = HetergeneousMatrixVariateKernel(
            task_covar_module = MatrixVariateIndexKernel(
                ConstantIndexKernel(torch.from_numpy(A).float()),
                ConstantIndexKernel(torch.from_numpy(B).float())),
            data_covar_module = DataKernel(),
            decoder = decoder)
    with gpsettings.debug(True):
        Kgot = HMVKer(MXUtorch, MXUtorch)
        Kgot_np = Kgot.evaluate().detach().cpu().numpy()

    assert Kgot_np == pytest.approx(Kexp)

    Kexp_test = kernel_test(H, A, B, Xtest, DataKernel().forward)
    _, MXUtest = encode_from_XU_numpy(Xtest, U, M=0)
    MXUtest_torch = torch.from_numpy(MXUtest).float()
    Kgot_test = HMVKer(MXUtest_torch, MXUtest_torch).evaluate().detach().cpu().numpy()

    assert Kgot_test == pytest.approx(Kexp_test)

    Kexp_train_test = kernel_train_test(H, A, B, X, Xtest, DataKernel().forward)
    MXUTrainTest = torch.cat((MXUtorch, MXUtest_torch), dim=0)
    Kgot_train_test = HMVKer(MXUTrainTest, MXUTrainTest).evaluate().detach().cpu().numpy()

    assert Kgot_train_test == pytest.approx(Kexp_train_test)


if __name__ == '__main__':
    test_dynamics_model_kernel()

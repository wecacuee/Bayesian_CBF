import warnings

import numpy as np
import torch
import pytest
import gpytorch.settings as gpsettings

from bayes_cbf.matrix_variate_multitask_model import DynamicModelGP

def sample_generator_trajectory(f, g, D, n, m):
    U = np.random.rand(D, m)
    X = np.zeros((D+1, n))
    X[0, :] = np.random.rand(n)
    Xdot = np.zeros((D, n))
    # Single trajectory
    for i in range(D):
        Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]
        X[i+1, :] = X[i, :] + Xdot[i, :] * dt

def sample_generator_independent(f, g, D, n, m):
    # Idependent random mappings
    U = np.random.rand(D, m)
    X = np.random.rand(D, n)
    Xdot = np.zeros((D, n))
    for i in range(D):
        Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]
    return Xdot, X, U


def test_GP_train_predict(n=2, m=3, dt = 0.001,
                          deterministic=True,
                          sample_generator=sample_generator_independent):
    #chosen_seed = np.random.randint(100000)
    chosen_seed = 18945
    print(chosen_seed)
    np.random.seed(chosen_seed)
    torch.manual_seed(chosen_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    A = np.random.rand(n,n)
    def f(x):
        assert x.shape[-1] == n
        cov = np.eye(n) * 0.0001
        return (A @ x if deterministic
                else np.random.multivariate_normal(A @ x, cov))
    f.A = A

    B = np.random.rand(n, m, n)
    def g(x):
        """
        Returns n x m matrix
        """
        assert x.shape[-1] == n
        cov_A = np.eye(n) * 0.0001
        cov_B = np.eye(m) * 0.0002
        cov = np.kron(cov_A, cov_B)

        return (
            B @ x if deterministic
            else np.random.multivariate_normal(
                    (B @ x).flatten(), cov
            ).reshape((n, m))
        )
    g.B = B

    # Collect training data
    D = 200
    Xdot, X, U = sample_generator(f, g, D, n, m)

    # Test train split
    shuffled_order = np.arange(D)
    np.random.shuffle(shuffled_order)
    train_indices = shuffled_order[:int(D*0.8)]
    test_indices = shuffled_order[int(D*0.8):]

    # Train data
    Xtrain, Utrain, XdotTrain = [Mat[train_indices, :]
                                 for Mat in (X, U, Xdot)]

    # Call the training routine
    dgp = DynamicModelGP()
    with gpsettings.max_cg_iterations(2000), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dgp.fit(Xtrain, Utrain, XdotTrain, training_iter=50, lr=0.01)

    # Test data
    Xtest, Utest, XdotTest = [Mat[test_indices, :]
                              for Mat in (X, U, Xdot)]

    UHtest = np.concatenate((np.ones((Utest.shape[0], 1)), Utest), axis=1)
    FXTexpected = np.empty((Xtest.shape[0], 1+m, n))
    for i in range(Xtest.shape[0]):
        FXTexpected[i, ...] = np.concatenate(
            (f(Xtest[i, :])[None, :], g(Xtest[i,  :]).T), axis=0)
        assert np.allclose(XdotTest[i, :], FXTexpected[i, :, :].T @ UHtest[i, :])

    #FXTexpected = np.concatenate(((f.A @ Xtest.T).T.reshape(-1, 1, n),
    #                              (g.B @ Xtest.T).T.reshape(-1, m, n)), axis=1)
    FXTmean, FXTcov = dgp.predict(Xtest)

    XdotGot = np.empty_like(XdotTest)
    for i in range(Xtest.shape[0]):
        XdotGot[i, :] = FXTmean[i, :, :].T @ UHtest[i, :]
    assert XdotGot == pytest.approx(XdotTest, rel=0.05)


if __name__ == '__main__':
    test_GP_train_predict()

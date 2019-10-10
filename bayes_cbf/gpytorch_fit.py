import torch

from bayes_cbf.matrix_variate_multitask_model import DynamicModelGP

def test_GP_train_predict(n=2, m=3, dt = 0.001):
    import numpy as np
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
        cov = np.ones((n,n)) * 0.01
        return np.random.multivariate_normal(A @ x, cov)
    f.A = A

    B = np.random.rand(n, m, n)
    def g(x):
        """
        Returns n x m matrix
        """
        assert x.shape[-1] == n
        cov_A = np.ones((n, n)) * 0.01
        cov_B = np.ones((m, m)) * 0.02
        cov = np.kron(cov_A, cov_B)

        return np.random.multivariate_normal((B @ x).flatten(), cov).reshape((n, m))
    g.B = B

    # Collect training data
    D = 10
    U = np.random.rand(D, m)
    X = np.zeros((D+1, n))
    X[0, :] = np.random.rand(n)
    Xdot = np.zeros((D, n))

    # # Single trajectory
    # for i in range(D):
    #     Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]
    #     X[i+1, :] = X[i, :] + Xdot[i, :] * dt

    # Idependent random mappings
    X = np.random.rand(D, n)
    for i in range(D):
        Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]

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
    dgp.fit(Xtrain, Utrain, XdotTrain, training_iter=500, lr=0.1)

    # Test data
    Xtest, Utest, XdotTest = [Mat[test_indices, :]
                              for Mat in (X, U, Xdot)]

    FXTexpected = np.concatenate(((f.A @ Xtest.T).T.reshape(-1, 1, n),
                               (g.B @ Xtest.T).T.reshape(-1, m, n)), axis=1)
    FXTmean, FXTcov = dgp.F(Xtest)
    error = np.linalg.norm(FXTmean[:] - FXTexpected[:])
    print("norm(actual - expected) / norm = {} / {}"
          .format(error, np.linalg.norm(FXTexpected[:])))
    assert np.allclose(FXTmean, FXTexpected)


if __name__ == '__main__':
    test_GP_train_predict()
import torch

from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from bayes_cbf.matrix_variate_multitask_model import MatrixVariateGP

def GPTrain(X, U, Y):
    # Initialize model and likelihood
    # Noise model for GPs
    likelihood = MultitaskGaussianLikelihood(num_tasks=X.size(-1))
    # Actual model
    model = MatrixVariateGP(X, U, Y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the amount of training data
    # mll = VariationalELBO(likelihood, model, Y.numel())
    mll = ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 50
    for i in range(training_iter):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(*model.train_inputs)
        # Calc loss and backprop gradients
        loss = -mll(output, Y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()

    return model


def GPPredict(model, x, u):
    return model(x).matmul(u)


def test_GP_train_predict(n=2, m=3, dt = 0.001):
    import numpy as np

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

    # Single trajectory
    for i in range(D):
        Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]
        X[i+1, :] = X[i, :] + Xdot[i, :] * dt

    # Test train split
    shuffled_order = np.arange(D)
    np.random.shuffle(shuffled_order)
    train_indices = shuffled_order[:int(D*0.8)]
    test_indices = shuffled_order[int(D*0.8):]
    Xtrain = X[train_indices, :]
    XdotTrain = Xdot[train_indices, :]
    Utrain = U[train_indices, :]

    # Test train split
    Xtest = X[test_indices, :]
    XdotTest = Xdot[test_indices, :]
    Utest = U[test_indices, :]

    FModel = GPTrain(torch.from_numpy(Xtrain).float(),
                     torch.from_numpy(Utrain).float(),
                     torch.from_numpy(XdotTrain).float())
    np.allclose(FModel(Xtest).mean().numpy(),
                np.concatenate(((np.f.A @ Xtest.T).reshape(n, -1, D), g.B @ Xtest.T)), axis=1)

if __name__ == '__main__':
    test_GP_train_predict()

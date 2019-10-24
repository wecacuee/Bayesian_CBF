from typing import Any
import warnings

import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

import numpy as np
import torch

from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.kernels import ScaleKernel, RBFKernel, IndexKernel
from gpytorch.likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.utils.memoize import cached
import gpytorch.settings as gpsettings

from bayes_cbf.matrix_variate_multitask_kernel import MatrixVariateIndexKernel, HetergeneousMatrixVariateKernel
from bayes_cbf.matrix_variate_multitask_model import HetergeneousMatrixVariateMean


class Namespace:
    """
    Makes a class as a namespace for static functions
    """
    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if isinstance(val, Callable):
            return staticmethod(val)
        else:
            return val


class Arr(Namespace):
    """
    Namespace for functions that works for both numpy as pytorch
    """
    def cat(arrays, axis=0):
        if isinstance(arrays[0], torch.Tensor):
            X = torch.cat(arrays, dim=axis)
        else:
            X = np.concatenate(arrays, axis=axis)
        return X


class CatEncoder:
    """
    Encodes and decodes the arrays by concatenating them
    """
    def __init__(self, *sizes):
        self.sizes = list(sizes)

    @classmethod
    def from_data(cls, *arrays):
        self = cls(*[A.shape[-1] for A in arrays])
        return self, self.encode(*arrays)

    def encode(self, *arrays):
        X = Arr.cat(arrays, axis=-1)
        return X

    def decode(self, X):
        idxs = np.cumsum([0] + self.sizes)
        arrays = [X[..., s:e]
                  for s,e in zip(idxs[:-1], idxs[1:])]
        return arrays


class IdentityLikelihood(_GaussianLikelihoodBase):
    """
    Dummy likelihood class that does not do anything. It tries to be as close
    to identity as possible.

    gpytorch.likelihoods.Likelihood is supposed to model p(y|f(x)).

    GaussianLikelihood model this by y = f(x) + ε, ε ~ N(0, σ²)

    IdentityLikelihood tries to model y = f(x) , without breaking the gpytorch
    `exact_prediction_strategies` function which requires GaussianLikelihood.
    """
    def __init__(self):
        self.min_possible_noise = 1e-6
        super().__init__(noise_covar=FixedGaussianNoise(noise=torch.tensor(self.min_possible_noise)))

    @property
    def noise(self):
        return 0

    @noise.setter
    def noise(self, _):
        LOG.warn("Ignore setting of noise")

    def forward(self, function_samples: torch.Tensor, *params: Any, **kwargs:
                Any) -> base_distributions.Normal:
        # FIXME: How can we get the covariance of the function samples?
        return base_distributions.Normal(
            function_samples,
            self.min_possible_noise * torch.eye(function_samples.size()))

    def marginal(self, function_dist: MultivariateNormal, *params: Any,
                 **kwargs: Any) -> MultivariateNormal:
        return function_dist




class ControlAffineExactGP(ExactGP):
    """
    ExactGP Model to capture the heterogeneous gaussian process

    Given MXU, M, X, U = MXU

        Xdot = F(X)ᵀU    if M = 1
        Y = F(X)ᵀ        if M = 0
    """
    def __init__(self, x_dim, u_dim, likelihood, rank=1):
        super().__init__(None, None, likelihood)
        self.matshape = (1+u_dim, x_dim)
        self.decoder = CatEncoder(1, x_dim, 1+u_dim)
        self.mean_module = HetergeneousMatrixVariateMean(
            ConstantMean(),
            self.decoder,
            self.matshape)

        self.task_covar = MatrixVariateIndexKernel(
            IndexKernel(num_tasks=self.matshape[1]),
            IndexKernel(num_tasks=self.matshape[0]),
        )
        self.input_covar = ScaleKernel(RBFKernel())
        self.covar_module = HetergeneousMatrixVariateKernel(
            self.task_covar,
            self.input_covar,
            self.decoder)

    def set_train_data(self, Xtrain, Utrain, XdotTrain):
        assert self.matshape == (1+Utrain.shape[-1], Xtrain.shape[-1])
        assert Xtrain.shape[-1] == XdotTrain.shape[-1]
        _, MXUtrain = self.encode_from_XU(Xtrain, Utrain, 1)
        super().set_train_data(inputs=(MXUtrain,),
                               targets=XdotTrain.reshape(-1), strict=False)

    def encode_from_XU(self, Xtrain, Utrain=None, M=0):
        Mtrain = Xtrain.new_full([Xtrain.size(0), 1], M)
        if M:
            assert Utrain is not None
            UHtrain = torch.cat([Mtrain, Utrain], dim=1)
        else:
            UHtrain = Xtrain.new_zeros((Xtrain.size(0), self.matshape[0]))
        return CatEncoder.from_data(Mtrain, Xtrain, UHtrain)

    def forward(self, mxu):
        mean_x = self.mean_module(mxu)
        with gpsettings.lazily_evaluate_kernels(False):
            covar_x = self.covar_module(mxu)
        return MultivariateNormal(mean_x, covar_x)


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class ControlAffineRegressor:
    """
    Scikit like wrapper around learning and predicting GaussianProcessRegressor

    Usage:
    F(X), COV(F(X)) = ControlAffineRegressor()
                        .fit(Xtrain, Utrain, XdotTrain)
                        .predict(Xtest, return_cov=True)
    """
    def __init__(self, x_dim, u_dim, device=None, default_device=default_device):
        self.device = device or default_device()

        # Initialize model and likelihood
        # Noise model for GPs
        self.likelihood = IdentityLikelihood()
        # Actual model
        self.model = ControlAffineExactGP(
            x_dim, u_dim, self.likelihood
        ).to(device=self.device)

    def fit(self, *args, max_cg_iterations=2000, **kwargs):
        with warnings.catch_warnings(), \
              gpsettings.max_cg_iterations(max_cg_iterations):
            warnings.simplefilter("ignore")
            return self._fit_with_warnings(*args, **kwargs)

    def _fit_with_warnings(self, Xtrain, Utrain, XdotTrain, training_iter = 50,
                           lr=0.1):
        if Xtrain.shape[0] == 0:
            # Do nothing if no data
            return self

        device = self.device
        model = self.model
        likelihood = self.likelihood

        # Convert to torch
        Xtrain = torch.from_numpy(Xtrain).float().to(device=device)
        Utrain = torch.from_numpy(Utrain).float().to(device=device)
        XdotTrain = torch.from_numpy(XdotTrain).float().to(device=device)

        model.set_train_data(Xtrain, Utrain, XdotTrain)

        # Set in train mode
        model.train()
        likelihood.train()

        # Find optimal model hyperparameters


        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=(torch.tensor([0.3, 0.6, 0.8, 0.90])*training_iter).tolist())


        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the amount of training data
        # mll = VariationalELBO(likelihood, model, Y.numel())
        mll = ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
            # Zero backpropped gradients from previous iteration
            optimizer.zero_grad()
            # Get predictive output
            output = model(*model.train_inputs)
            for p in model.parameters(recurse=True):
                assert not torch.isnan(p).any()
            # Calc loss and backprop gradients
            loss = -mll(output, XdotTrain.reshape(-1))
            assert not torch.isnan(loss).any()
            loss.backward()
            for p in model.parameters(recurse=True):
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any()

            print('Iter %d/%d - Loss: %.3f, lr: %.3g' % (i + 1, training_iter,
                                                         loss.item(),
                                                         scheduler.get_lr()[0]))
            optimizer.step()
            scheduler.step()
        return self

    def predict(self, Xtest, return_cov=True):
        device = self.device
        Xtest = torch.from_numpy(Xtest).float().to(device=device)
        # Switch back to eval mode
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Call fit() with training data before calling predict")

        # Set in eval mode
        self.model.eval()
        self.likelihood.eval()

        # Concatenate the test set
        _, MXUHtest = self.model.encode_from_XU(Xtest)
        output = self.model(MXUHtest)

        mean, cov = (output.mean.reshape(-1, *self.model.matshape),
                     output.covariance_matrix)
        mean_np, cov_np = [arr.detach().cpu().numpy() for arr in (mean, cov)]
        return (mean_np, cov_np) if return_cov else mean_np

    def f_func(self, Xtest, return_cov=False):
        if return_cov:
            mean_FxT, cov_FxT = self.predict(Xtest, return_cov=return_cov)
            cov_fx = cov_FxT[:, :1, :1]
        else:
            mean_FxT = self.predict(Xtest, return_cov=return_cov)
        mean_fx = mean_FxT[:, 0, :]
        return (mean_fx, cov_fx) if return_cov else mean_fx

    def g_func(self, Xtest, return_cov=False):
        if return_cov:
            mean_FxT, cov_FxT = self.predict(Xtest, return_cov=return_cov)
            cov_gx = cov_FxT[:, 1:, 1:]
        else:
            mean_FxT = self.predict(Xtest, return_cov=return_cov)
        mean_gx = mean_FxT[:, 1:, :]
        return (mean_gx, cov_gx) if return_cov else mean_gx

    def state_dict(self):
        return dict(model=self.model.state_dict(),
                    likelihood=self.likelihood.state_dict())

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.likelihood.load_state_dict(state_dict['likelihood'])

    def save(self, path='/tmp/saved.pickle'):
        torch.save(self.state_dict(), path)

    def load(self, path='/tmp/saved.pickle'):
        self.load_state_dict(torch.load(path))

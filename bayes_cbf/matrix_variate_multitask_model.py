from typing import Any

import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


import torch


import numpy as np
import torch

from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal, base_distributions
from gpytorch.kernels import ScaleKernel, RBFKernel, WhiteNoiseKernel, IndexKernel
from gpytorch.likelihoods import _GaussianLikelihoodBase,MultitaskGaussianLikelihood, GaussianLikelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
import gpytorch.settings as gpsettings

from bayes_cbf.matrix_variate_multitask_kernel import MatrixVariateIndexKernel, HetergeneousMatrixVariateKernel, prod


class Namespace:
    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if isinstance(val, Callable):
            return staticmethod(val)
        else:
            return val

class Arr(Namespace):
    def cat(arrays, axis=0):
        if isinstance(arrays[0], torch.Tensor):
            X = torch.cat(arrays, dim=axis)
        else:
            X = np.concatenate(arrays, axis=axis)
        return X


class CatEncoder:
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

class HeterogeneousGaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(self):
        # NOTE: Do nothing
        super().__init__(noise_covar=FixedGaussianNoise(noise=1e-4))

    @property
    def noise(self):
        return 0

    @noise.setter
    def noise(self, _):
        LOG.warn("Ignore setting of noise")

    def forward(self, function_samples: torch.Tensor, *params: Any, **kwargs: Any) -> base_distributions.Normal:
        # FIXME: How can we get the covariance of the function samples?
        return base_distributions.Normal(
            function_samples,
            1e-4 * torch.eye(function_samples.size()))

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        return function_dist


class HetergeneousMatrixVariateMean(MultitaskMean):
    def __init__(self, mean_module, decoder, matshape, **kwargs):
        num_tasks = prod(matshape)
        super().__init__(mean_module, num_tasks, **kwargs)
        self.decoder = decoder
        self.matshape = matshape

    def forward(self, MXU):
        B = MXU.shape[:-1]

        Ms, X, UH = self.decoder.decode(MXU)
        assert Ms.size(-1) == 1
        Ms = Ms[..., 0]
        idxs = torch.nonzero(Ms - Ms.new_ones(Ms.size()))
        idxend = torch.min(idxs) if idxs.numel() else Ms.size(-1)
        # assume sorted
        assert (Ms[..., idxend:] == 0).all()
        UH = UH[..., :idxend, :]
        X1 = X[..., :idxend, :]
        X2 = X[..., idxend:, :]
        mu = torch.cat([sub_mean(MXU).unsqueeze(-1)
                        for sub_mean in self.base_means], dim=-1)
        mu  = mu.reshape(-1, *self.matshape)
        XdotMean = UH.unsqueeze(-2) @ mu[:idxend, ...] # D x n
        output = XdotMean.reshape(-1)
        if Ms.size(-1) != idxend:
            Fmean = mu[idxend:, ...].reshape(-1)
            output = torch.cat([output, Fmean])
        return output


class DynamicsModelExactGP(ExactGP):
    def __init__(self, Xtrain, Utrain, XdotTrain, likelihood, rank=1):
        self.matshape = (1+Utrain.size(-1), Xtrain.size(-1))
        self.decoder, MXUtrain = self.encode_from_XU(Xtrain, Utrain, 1)
        super(DynamicsModelExactGP, self).__init__(MXUtrain, XdotTrain.reshape(-1),
                                              likelihood)
        self.mean_module = HetergeneousMatrixVariateMean(
            ConstantMean(),
            self.decoder,
            self.matshape)

        task_covar = MatrixVariateIndexKernel(
                IndexKernel(num_tasks=self.matshape[1]),
                IndexKernel(num_tasks=self.matshape[0]),
            )
        input_covar = ScaleKernel(RBFKernel())
        self.covar_module = HetergeneousMatrixVariateKernel(
            task_covar,
            input_covar,
            self.decoder)

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


class DynamicModelGP:
    def __init__(self):
        self.likelihood = None
        self.model = None

    def fit(self, Xtrain, Utrain, XdotTrain, training_iter = 50, lr=0.1):
        # Convert to torch
        Xtrain = torch.from_numpy(Xtrain).float()
        Utrain = torch.from_numpy(Utrain).float()
        XdotTrain = torch.from_numpy(XdotTrain).float()

        # Initialize model and likelihood
        # Noise model for GPs
        likelihood = self.likelihood = HeterogeneousGaussianLikelihood()
        # Actual model
        model = self.model = DynamicsModelExactGP(Xtrain, Utrain,
                                                  XdotTrain, likelihood)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the amount of training data
        # mll = VariationalELBO(likelihood, model, Y.numel())
        mll = ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
            # Zero backpropped gradients from previous iteration
            optimizer.zero_grad()
            # Get predictive output
            output = model(*model.train_inputs)
            # Calc loss and backprop gradients
            loss = -mll(output, XdotTrain.reshape(-1))
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()
        return self

    def predict(self, Xtest):
        Xtest = torch.from_numpy(Xtest).float()
        # Switch back to eval mode
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Call train before calling predict_F")

        self.model.eval()
        self.likelihood.eval()

        # Concatenate the test set
        _, MXUHtest = self.model.encode_from_XU(Xtest)
        output = self.model(MXUHtest)

        return (output.mean.reshape(-1, *self.model.matshape).detach().numpy(),
                output.covariance_matrix.detach().numpy())


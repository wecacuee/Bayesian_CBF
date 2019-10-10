import numpy as np
import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, WhiteNoiseKernel, IndexKernel
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.models import ExactGP
import gpytorch.settings as gpsettings

from bayes_cbf.matrix_variate_multitask_kernel import MatrixVariateIndexKernel, HetergeneousMatrixVariateKernel
from bayes_cbf.heterogeneous_mean import HetergeneousMatrixVariateMean


class CatEncoder:
    def __init__(self, *sizes):
        self.sizes = list(sizes)

    @classmethod
    def from_data(cls, *arrays):
        self = cls(*[A.size(-1) for A in arrays])
        return self, self.encode(*arrays)

    def encode(self, *arrays):
        X = torch.cat(arrays, dim=-1)
        return X

    def decode(self, X):
        idxs = np.cumsum([0] + self.sizes)
        arrays = [X[..., s:e]
                  for s,e in zip(idxs[:-1], idxs[1:])]
        return arrays


class MatrixVariateGP(ExactGP):
    def __init__(self, Xtrain, Utrain, XdotTrain, likelihood):
        Mtrain = Xtrain.new_ones([Xtrain.size(0), 1])
        UHtrain = torch.cat([Mtrain, Utrain], dim=1)
        self.decoder, MXUtrain = CatEncoder.from_data(Mtrain, Xtrain, UHtrain)
        super(MatrixVariateGP, self).__init__(MXUtrain, XdotTrain, likelihood)
        D, n = Xtrain.shape
        D, m = Utrain.shape
        num_tasks = (1+m) * n
        self.mean_module = HetergeneousMatrixVariateMean(
            ConstantMean(),
            decoder=self.decoder,
            num_tasks=num_tasks)

        task_covar = MatrixVariateIndexKernel(
                IndexKernel(num_tasks=num_tasks),
                IndexKernel(num_tasks=num_tasks),
            )
        input_covar = ScaleKernel(RBFKernel())
        self.covar_module = HetergeneousMatrixVariateKernel(
            task_covar,
            input_covar,
            self.decoder,
            num_tasks=num_tasks, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        with gpsettings.lazily_evaluate_kernels(False):
            covar_x = self.covar_module(x)
        import pdb; pdb.set_trace()
        return MultitaskMultivariateNormal(mean_x, covar_x)


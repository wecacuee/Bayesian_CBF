import warnings
from typing import Any
from itertools import zip_longest
from collections import namedtuple

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


GaussianProcess = namedtuple('GaussianProcess', ["mean", "k"])


def torch_kron(A, B):
    """
    >>> B = torch.rand(5,3,3)
    >>> A = torch.rand(5,2,2)
    >>> AB = torch_kron(A, B)
    >>> torch.allclose(AB[1, :3, :3] , A[1, 0,0] * B[1, ...])
    True
    >>> BA = torch_kron(B, A)
    >>> torch.allclose(BA[1, :2, :2] , B[1, 0,0] * A[1, ...])
    True
    """
    b = B.shape[0]
    assert A.shape[0] == b
    B_shape = sum([[1, si] for si in B.shape[1:]], [])
    A_shape = sum([[si, 1] for si in A.shape[1:]], [])
    kron_shape = [a*b for a, b in zip_longest(A.shape[1:], B.shape[1:], fillvalue=1)]
    return (A.reshape(b, *A_shape) * B.reshape(b, *B_shape)).reshape(b, *kron_shape)


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

        Xdot = F(X)U    if M = 1
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

    def state_dict(self):
        return dict(matshape=self.matshape,
                    decoder=self.decoder,
                    mean_module=self.mean_module,
                    task_covar=self.task_covar,
                    input_covar=self.input_covar,
                    covar_module=self.covar_module)

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)


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
        self._cache = dict()

    def _ensure_device_dtype(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.float().to(device=self.device)
        return X

    def fit(self, *args, max_cg_iterations=2000, **kwargs):
        with warnings.catch_warnings(), \
              gpsettings.max_cg_iterations(max_cg_iterations):
            warnings.simplefilter("ignore")
            return self._fit_with_warnings(*args, **kwargs)

    def _fit_with_warnings(self, Xtrain_in, Utrain_in, XdotTrain_in, training_iter = 50,
                           lr=0.1):
        if Xtrain_in.shape[0] == 0:
            # Do nothing if no data
            return self

        device = self.device
        model = self.model
        likelihood = self.likelihood

        # Convert to torch
        Xtrain, Utrain, XdotTrain = [
            self._ensure_device_dtype(X)
            for X in (Xtrain_in, Utrain_in, XdotTrain_in)]

        self.clear_cache()
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

            LOG.info('Iter %d/%d - Loss: %.3f, lr: %.3g' % (i + 1, training_iter,
                                                            loss.item(),
                                                            scheduler.get_lr()[0]))
            optimizer.step()
            scheduler.step()
        return self

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def predict(self, Xtest_in, return_cov=True):
        Xtest = self._ensure_device_dtype(Xtest_in)

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
        #mean_np, cov_np = [arr.detach().cpu().numpy() for arr in (mean, cov)]
        mean = mean.to(device=Xtest_in.device, dtype=Xtest_in.dtype)
        cov = cov.to(device=Xtest_in.device, dtype=Xtest_in.dtype)
        return (mean, cov) if return_cov else mean
        #return mean, cov

    def _perturbed_cholesky_compute(self, Kb,
                           cholesky_tries=10,
                           cholesky_perturb_init=1e-5,
                           cholesky_perturb_scale=10):

        # Kb can be singular because of repeated datasamples
        # Add diagonal jitter
        Kb_sqrt = None
        cholesky_perturb_factor = cholesky_perturb_init
        for _ in range(cholesky_tries):
            try:
                Kbp = Kb + cholesky_perturb_factor * Kb.diag() * (
                    torch.eye(Kb.shape[0], dtype=Kb.dtype, device=Kb.device) *
                    torch.rand(Kb.shape[0], dtype=Kb.dtype, device=Kb.device)
                    )
                Kb_sqrt = torch.cholesky(Kbp)
            except RuntimeError as e:
                LOG.warning("Cholesky failed with perturb={} on error {}".format(cholesky_perturb_factor, str(e)))
                cholesky_perturb_factor = cholesky_perturb_factor * cholesky_perturb_scale
                continue

        if Kb_sqrt is None:
            raise
        return Kb_sqrt

    def _perturbed_cholesky(self, Kb, cache_key="perturbed_cholesky" ):
        if cache_key not in self._cache:
            self._cache[cache_key] = self._perturbed_cholesky_compute(Kb)

        return self._cache[cache_key]

    def clear_cache(self):
        self._cache = dict()

    def custom_predict(self, Xtest_in, Utest_in=None, UHfill=1, Xtestp_in=None):
        """
        Gpytorch is complicated. It uses terminology like fantasy something,
        something. Even simple exact prediction strategy uses Laczos. I do not
        understand Laczos and Gpytorch code.
        Let the training be handled by Gpytorch. After that i take things in my
        own hands and predict myself.

        Matrix variate GP: Separate A and B

            f(x; u) ~ 𝕄𝕍ℙ(mean(x)u, A, B k(x, x'))
            vec(f)(x; u) ~ ℕ(μ(x)u, uᵀBu ⊗ A k(x, x'))

            K⁻¹(XU,XU):= [k(xᵢ,xⱼ)uᵢᵀBuⱼ]ᵢⱼ
            k* := [k(xᵢ, x*)uᵀᵢB]ᵢ


            F*(x*)u ~ 𝕄𝕍ℙ( {[k*ᵀ K⁻¹] ⊗ Iₙ}(Y-μ(x)u), A,
                            uᵀ[k(x*,x*)B - k*ᵀK⁻¹k*]u)

        Vector variate GP (preffered):
            Kᶠ(u, u') = uᵀBu' ⊗ A = (uᵀBu)A = bᶠ(u, u') A
            ẋ = f(x;u)
            cov(f(x;u), f(x';u')) = k(x,x')Kᶠ(u, u') = k(x,x')bᶠ(u, u') ⊗ A

            f(x; u) ~ 𝔾ℙ(μ(x)u, k(x, x')bᶠ(u, u') ⊗ A)

            Kb⁻¹:= [k(xᵢ,xⱼ)uᵢᵀBuⱼ]ᵢⱼ
            kb* := [k(xᵢ,xⱼ)uᵢᵀBuⱼ]ᵢⱼ

            f*(x*; u) ~ 𝔾ℙ( {[(kb*ᵀK_b⁻¹) ⊗ Iₙ]}(Y-μ(x)u),
                            [kb(x*,x*) - k*bᵀKb⁻¹kb*] ⊗ A)

        Algorithm (Rasmussen and Williams 2006)
           1. L := cholesky(K)
           2. α := Lᵀ \ ( L \ Y )
           3. μ := kb*ᵀ α
           4. v := L \ kb*
           5. k* := k(x*,x*) - vᵀV
           6. log p(y|X) := -0.5 yᵀ α - ∑ log Lᵢᵢ - 0.5 n log(2π)
        """
        Xtest = self._ensure_device_dtype(Xtest_in)
        Xtestp = self._ensure_device_dtype(Xtestp_in) if Xtestp_in is not None else Xtest
        if Utest_in is None:
            UHtest = Xtest.new_zeros(Xtest.shape[0], self.model.matshape[0])
            UHtest[:, 0] = 1
        else:
            Utest = self._ensure_device_dtype(Utest_in)
            UHtest = torch.cat((Utest.new_full((Utest.shape[0], 1), UHfill), Utest), dim=-1)
        MXUHtrain = self.model.train_inputs[0]
        Mtrain, Xtrain, UHtrain = self.model.decoder.decode(MXUHtrain)
        nsamples = Xtrain.size(0)
        k = self.model.covar_module.data_covar_module
        A = self.model.covar_module.task_covar_module.U.covar_matrix.evaluate()
        B = self.model.covar_module.task_covar_module.V.covar_matrix.evaluate()
        Y = self.model.train_targets.reshape(nsamples, -1) - self.model.mean_module(Xtrain).reshape(nsamples, *self.model.matshape).transpose(-2,-1).bmm(UHtrain.unsqueeze(-1)).squeeze(-1)
        KXX = k(Xtrain, Xtrain).evaluate()
        uBu = UHtrain @ B @ UHtrain.T
        Kb = KXX * uBu

        Kb_sqrt = self._perturbed_cholesky(Kb)
        kb_star = k(Xtrain, Xtest).evaluate() * (UHtrain @ B @ UHtest.t())
        kb_star_p = (k(Xtrain, Xtestp).evaluate() * (UHtrain @ B @ UHtest.t())
                     if Xtestp_in is not None
                     else kb_star)
        kb_star_starp = k(Xtest, Xtestp).evaluate() * (UHtest @ B @ UHtest.t())
        α = torch.cholesky_solve(Y, Kb_sqrt) # check the shape of Y
        mean = self.model.mean_module(Xtest).reshape(
            Xtest.shape[0], *self.model.matshape).transpose(-2, -1).bmm(
                UHtest.unsqueeze(-1)).squeeze(-1) + kb_star.t() @ α
        v = torch.solve(kb_star, Kb_sqrt).solution
        vp = torch.solve(kb_star_p, Kb_sqrt).solution if Xtestp_in is not None else v
        scalar_var = kb_star_starp - v.t() @ vp
        return mean, scalar_var, A


    def predict_flatten(self, Xtest_in, Utest_in):
        """
        Directly predict

        f(x, u) = f(x) + g(x) @ u

        If you need f only, put Utest = [1, 0]
        """
        device = self.device
        if isinstance(Xtest_in, np.ndarray):
            Xtest = torch.from_numpy(Xtest_in)
        else:
            Xtest = Xtest_in
        Xtest = Xtest.float().to(device=device)

        if isinstance(Utest_in, np.ndarray):
            Utest = torch.from_numpy(Utest_in)
        else:
            Utest = Utest_in
        Utest = Utest.float().to(device=device)

        # Switch back to eval mode
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Call fit() with training data before calling predict")

        # Set in eval mode
        self.model.eval()
        self.likelihood.eval()

        # Concatenate the test set
        _, MXUHtest = self.model.encode_from_XU(Xtest, Utest=Utest)
        output = self.model(MXUHtest)

        mean, cov = (output.mean, output.covariance_matrix)
        return (mean.to(device=Xtest_in.device, dtype=Xtest_in.dtype),
                cov.to(device=Xtest_in.device, dtype=Xtest_in.dtype))

    def predict_grad(self, Xtest, Utest):
        """
        Directly predict

        ∇ₓf(x; u) = ∇ₓ f(x) + ∇ₓ g(x) @ u

        One way is to differentiate the kernel

        E[∇ₓf(x; u)] = ∇ₓk(x, X) K(X,X)⁻¹ Y
        Var(∇ₓf(x; u)) = Hₓₓk(x*,x*) - ∇ₓk(x, X)ᵀ K(X,X)⁻¹ ∇ₓk(X, x)

        or let pytorch do all the heavy lifting for us
        """
        self.zero_grad()
        Xtest = Xtest.requires_grad_(True)
        mean, _ = self.predict_flatten(Xtest, Utest)
        mean.backward(torch.eye(mean.shape[-1]))
        grad_mean = Xtest.grad
        self.zero_grad()
        cov = self.model.input_covar(Xtest, Xtest)
        cov.backward(retain_graph=True)
        grad_cov = Xtest.grad
        grad_cov.backward(torch.eye(grad_cov.shape[-1]))
        H_cov = Xtest.grad
        return grad_mean, grad_cov

    def f_func_orig(self, Xtest, return_cov=False):
        if return_cov:
            mean_Fx, cov_Fx = self.predict(Xtest, return_cov=return_cov)
            cov_fx = cov_Fx[:, :1, :1]
        else:
            mean_Fx = self.predict(Xtest, return_cov=return_cov)
        mean_fx = mean_Fx[:, 0, :]
        return (mean_fx, cov_fx) if return_cov else mean_fx

    def f_func_custom(self, Xtest_in, return_cov=False, Xtestp_in=None):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        mean_f, var_f, A =  self.custom_predict(Xtest, Xtestp_in=Xtestp_in)
        var_f = var_f.reshape(-1, 1, 1) * A
        if Xtest_in.ndim == 1:
            mean_f = mean_f.squeeze(0)
            var_f = var_f.squeeze(0)
        mean_f = mean_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        var_f = var_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        return (mean_f, var_f) if return_cov else mean_f

    f_func = f_func_custom

    def f_func_gp(self, Xtest):
        return GaussianProcess(
            self.f_func_custom(Xtest),
            self.f_func_custom(Xtest, return_cov=True)[1])

    def fu_func_gp(self, Xtest_in, Utest_in, Xtestp_in=None):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        Utest = (Utest_in.unsqueeze(0)
                 if Utest_in.ndim == 1
                 else Utest_in)
        mean_f, var_f, A =  self.custom_predict(Xtest, Utest, Xtestp_in=Xtestp_in)
        var_f = var_f.reshape(-1, 1, 1) * A
        if Xtest_in.ndim == 1:
            mean_f = mean_f.squeeze(0)
            var_f = var_f.squeeze(0)

        mean_f = mean_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        var_f = var_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        return GaussianProcess(mean_f, var_f)

    def g_func(self, Xtest_in, return_cov=False):
        assert not return_cov, "Don't know what matrix covariance looks like"
        mean_Fx = self.predict(Xtest_in, return_cov=return_cov)
        mean_gx = mean_Fx[:, 1:, :]
        if Xtest_in.ndim == 1:
            mean_gx = mean_gx.squeeze(0)
        mean_gx = mean_gx.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        return mean_gx.transpose(-2, -1)

    def gu_func(self, Xtest_in, Utest_in, return_cov=False, Xtestp_in=None):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        Utest = (Utest_in.unsqueeze(0)
                 if Utest_in.ndim == 1
                 else Utest_in)
        mean_gu, var_gu, A = self.custom_predict(Xtest, Utest, UHfill=0,
                                                 Xtestp_in=Xtestp_in)
        if Xtest_in.ndim == 1 and Utest_in.ndim == 1:
            mean_gu = mean_gu.squeeze(0)
            var_gu = var_gu.squeeze(0)
        return (mean_gu, var_gu * A) if return_cov else mean_gu

    def cbf_func(self, Xtest, grad_htest, return_cov=False):
        if return_cov:
            mean_Fx, cov_Fx = self.predict(Xtest, return_cov=True)
            cov_hFT = grad_htest.T @ cov_hFT @ grad_htest
        else:
            mean_Fx, cov_Fx = self.predict(Xtest, return_cov=False)
        mean_hFT = grad_htest @ mean_Fx
        return mean_hFT, cov_hFT

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

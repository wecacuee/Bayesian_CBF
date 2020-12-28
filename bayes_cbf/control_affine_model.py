from pathlib import Path
import warnings
from typing import Any, Callable
from itertools import zip_longest
from collections import namedtuple
from functools import partial

import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

import numpy as np
import torch

from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.kernels import ScaleKernel, RBFKernel, IndexKernel, LinearKernel
from gpytorch.likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior
from gpytorch.utils.memoize import cached
import gpytorch.settings as gpsettings

from bayes_cbf.matrix_variate_multitask_kernel import (
    MatrixVariateIndexKernel, HetergeneousMatrixVariateKernel,
    HetergeneousCoregionalizationKernel)
from bayes_cbf.matrix_variate_multitask_model import HetergeneousMatrixVariateMean
from bayes_cbf.misc import (torch_kron, DynamicsModel, t_jac, variable_required_grad,
                            t_hessian, gradgradcheck)
from bayes_cbf.gp_algebra import GaussianProcess


__directory__ = Path(__file__).parent or Path(".")
"""
The directory for this file
"""

class GaussianProcessFunc(namedtuple('GaussianProcessFunc', ["mean", "knl"])):
    @property
    def dtype(self):
        return self.mean.__self__.dtype

    def to(self, dtype):
        self.mean.__self__.to(dtype=dtype)



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

    def state_dict(self):
        return dict(sizes=self.sizes)

    def load_state_dict(self, state_dict):
        self.sizes = state_dict['sizes']


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
    def __init__(self, x_dim, u_dim, likelihood, rank=None,
                 gamma_length_scale_prior=None):
        super().__init__(None, None, likelihood)
        self.matshape = (1+u_dim, x_dim)
        self.decoder = CatEncoder(1, x_dim, 1+u_dim)
        self.mean_module = HetergeneousMatrixVariateMean(
            ConstantMean(),
            self.decoder,
            self.matshape)

        self.task_covar = MatrixVariateIndexKernel(
            IndexKernel(num_tasks=self.matshape[1],
                        rank=(self.matshape[1] if rank is None else rank)),
            IndexKernel(num_tasks=self.matshape[0],
                        rank=(self.matshape[0] if rank is None else rank))
        )
        prior_args = dict() if gamma_length_scale_prior is None else dict(
            lengthscale_prior=GammaPrior(*gamma_length_scale_prior))
        self.input_covar = ScaleKernel(RBFKernel(**prior_args))
            #+ LinearKernel()) # FIXME: how to reduce the variance of LinearKernel
        self.covar_module = HetergeneousMatrixVariateKernel(
            self.task_covar,
            self.input_covar,
            self.decoder,
        )

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
        sd = dict(matshape=self.matshape,
                  decoder=self.decoder.state_dict(),
                  mean_module=self.mean_module.state_dict(),
                  task_covar=self.task_covar.state_dict(),
                  input_covar=self.input_covar.state_dict(),
                  covar_module=self.covar_module.state_dict(),
                  train_inputs=self.train_inputs,
                  train_targets=self.train_targets)
        return sd

    def load_state_dict(self, state_dict):
        self.matshape = state_dict.pop('matshape')
        self.train_inputs = state_dict.pop('train_inputs')
        self.train_targets = state_dict.pop('train_targets')
        for k, v in state_dict.items():
            getattr(self, k).load_state_dict(v)
        return self


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class ControlAffineRegressor(DynamicsModel):
    """
    Scikit like wrapper around learning and predicting GaussianProcessRegressor

    Usage:
    F(X), COV(F(X)) = ControlAffineRegressor()
                        .fit(Xtrain, Utrain, XdotTrain)
                        .predict(Xtest, return_cov=True)
    """
    ground_truth = False
    def __init__(self, x_dim, u_dim, device=None, default_device=default_device,
                 gamma_length_scale_prior=None,
                 model_class=ControlAffineExactGP):
        super().__init__()
        self.device = device or default_device()
        self.x_dim = x_dim
        self.u_dim = u_dim
        # Initialize model and likelihood
        # Noise model for GPs
        self.likelihood = IdentityLikelihood()
        # Actual model
        self.model_class = model_class
        self.model = model_class(
            x_dim, u_dim, self.likelihood,
            gamma_length_scale_prior=gamma_length_scale_prior
        ).to(device=self.device)
        self._cache = dict()
        self._f_func_gp = GaussianProcess(self.f_func_mean, self.f_func_knl, (self.x_dim,), name="f")

    @property
    def ctrl_size(self):
        return self.u_dim

    @property
    def state_size(self):
        return self.x_dim

    def _ensure_device_dtype(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(device=self.device, dtype=next(self.model.parameters())[0].dtype)
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
            loss = -mll(
                output,
                XdotTrain.reshape(-1) * (
                    1 + 1e-6  * torch.rand_like(XdotTrain.reshape(-1))))

            assert not torch.isnan(loss).any()
            assert not torch.isinf(loss).any()
            loss.backward()
            for p in model.parameters(recurse=True):
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any()

            LOG.debug('Iter %d/%d - Loss: %.3f, lr: %.3g' % (i + 1, training_iter,
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

    def _perturbed_cholesky_compute(self, k, B, Xtrain, UHtrain,
                           cholesky_tries=10,
                           cholesky_perturb_init=1e-5,
                           cholesky_perturb_scale=10):
        KXX = k(Xtrain, Xtrain)
        uBu = UHtrain @ B @ UHtrain.T
        Kb = KXX * uBu

        # Kb can be singular because of repeated datasamples
        # Add diagonal jitter
        Kbp, Kb_sqrt = make_psd(Kb)
        return Kb_sqrt

    def _perturbed_cholesky(self, k, B, Xtrain, UHtrain,
                            cache_key="perturbed_cholesky" ):
        if cache_key not in self._cache:
            self._cache[cache_key] = self._perturbed_cholesky_compute(
                k, B, Xtrain, UHtrain)

        return self._cache[cache_key]

    def clear_cache(self):
        self._cache = dict()

    def custom_predict(self, Xtest_in, Utest_in=None, UHfill=1, Xtestp_in=None,
                       Utestp_in=None, UHfillp=1,
                       compute_cov=True,
                       grad_gp=False,
                       grad_check=False,
                       scalar_var_only=False):
        """
        Gpytorch is complicated. It uses terminology like fantasy something,
        something. Even simple exact prediction strategy uses Laczos. I do not
        understand Laczos and Gpytorch code.
        Let the training be handled by Gpytorch. After that i take things in my
        own hands and predict myself.

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
           5. k* := k(x*,x*) - vᵀv
           6. log p(y|X) := -0.5 yᵀ α - ∑ log Lᵢᵢ - 0.5 n log(2π)

        """
        Xtest = self._ensure_device_dtype(Xtest_in)
        Xtestp = (self._ensure_device_dtype(Xtestp_in) if Xtestp_in is not None
                  else Xtest)
        if Utest_in is None:
            UHtest = Xtest.new_zeros(Xtest.shape[0], self.model.matshape[0])
            UHtest[:, 0] = 1
        else:
            Utest = self._ensure_device_dtype(Utest_in)
            UHtest = torch.cat((Utest.new_full((Utest.shape[0], 1), UHfill),
            Utest), dim=-1)
        if Utestp_in is None:
            UHtestp = UHtest
        else:
            Utestp = self._ensure_device_dtype(Utestp_in)
            UHtestp = torch.cat((Utest.new_full((Utestp.shape[0], 1), UHfillp),
                                 Utestp), dim=-1)

        k_xx = lambda x, xp: self.model.covar_module.data_covar_module(
            x, xp).evaluate()
        if not grad_gp:
            k_ss = k_xs = k_sx = k_xx
            mean_s = self.model.mean_module
        else:
            def grad_mean_s(xs):
                with variable_required_grad(xs):
                    # allow_unused=True because the mean_module can be ConstantMean
                    mean_xs = self.model.mean_module(xs)
                    grad_mean_xs = torch.autograd.grad(
                        list(mean_xs.flatten()),
                        xs, allow_unused=True)[0]
                if grad_mean_xs is None:
                    return xs.new_zeros(xs.shape[0], *self.model.matshape,
                                        xs.shape[-1])
                else:
                    return grad_mean_xs.reshape(xs.shape[0],
                                                *self.model.matshape,
                                                xs.shape[-1])

            mean_s = grad_mean_s

            def grad_ksx(xs, xx):
                with variable_required_grad(xs):
                    return torch.autograd.grad(list(k_xx(xs, xx)), xs)[0]
            def grad_kxs(xx, xs):
                with variable_required_grad(xs):
                    return torch.autograd.grad(list(k_xx(xx, xs)), xs)[0]
            k_sx = grad_ksx
            k_xs = grad_kxs
            def Hessian_kxx(xs, xsp):
                if xs is xsp:
                    xsp = xsp.detach().clone()
                return t_hessian(k_xx, xs, xsp)
            k_ss = Hessian_kxx
        A = self.model.covar_module.task_covar_module.U.covar_matrix.evaluate()
        B = self.model.covar_module.task_covar_module.V.covar_matrix.evaluate()

        # Output of mean_s(Xtest) is (B, (1+m)n)
        # Make it (B, (1+m), n, 1) then transpose
        # (B, n, 1, (1+m)) and multiply with UHtest (B, (1+m)) to get
        # (B, n, 1)
        fX_mean_test = mean_s(Xtest)
        fu_mean_test = (
            fX_mean_test
            .reshape(Xtest.shape[0], *self.model.matshape, -1) # (B, 1+m, n, n or 1)
            .permute(0, 2, 3, 1) # (B, n, n or 1, 1+m)
            .reshape(Xtest.shape[0], -1, self.model.matshape[0]) # (B, n(n or 1), 1+m)
            .bmm(UHtest.unsqueeze(-1)) # (B, n(n or 1), 1)
            .squeeze(-1) # (B, n(n or 1))
        )

        if self.model.train_inputs is None:
            # We do not have training data just return the mean and prior covariance
            if fX_mean_test.ndim == 4:
                fu_mean_test = fu_mean_test.reshape(Xtest.shape[0], *self.model.matshape[1:], -1)
            else:
                fu_mean_test = fu_mean_test.reshape(Xtest.shape[0], *self.model.matshape[1:])

            # Compute k(x*,x*) uᵀBu
            kb_star_starp = k_ss(Xtest, Xtestp) * (UHtest @ B @ UHtestp.t())
            # 5. k* := k(x*,x*) uᵀBu
            scalar_var = kb_star_starp
            return fu_mean_test, torch_kron(scalar_var.unsqueeze(0), A.unsqueeze(0))

        MXUHtrain = self.model.train_inputs[0]
        Mtrain, Xtrain, UHtrain = self.model.decoder.decode(MXUHtrain)
        nsamples = Xtrain.size(0)

        if grad_check and not grad_gp:
            with variable_required_grad(Xtest):
                old_dtype = self.dtype
                self.double_()
                torch.autograd.gradcheck(
                    lambda X: self.model.covar_module.data_covar_module(
                            Xtrain.double(), X).evaluate(),
                    Xtest.double())
                gradgradcheck(
                    partial(lambda s, X, Xp: s.model.covar_module.data_covar_module(
                        X, Xp).evaluate(), self),
                    Xtest[:1, :].double())
                self.to(dtype=old_dtype)
        Y = (
            self.model.train_targets.reshape(nsamples, -1)
             - self.model.mean_module(Xtrain).reshape(nsamples,
                                                      *self.model.matshape)
             .transpose(-2,-1)
             .bmm(UHtrain.unsqueeze(-1))
             .squeeze(-1)
        )

        # 1. L := cholesky(K)
        Kb_sqrt = self._perturbed_cholesky(k_xx, B, Xtrain, UHtrain)
        kb_star = k_xs(Xtrain, Xtest) * (UHtrain @ B @ UHtest.t())
        if grad_check:
            old_dtype = self.dtype
            self.double_()
            kb_star_func = lambda X: k_xs(Xtrain.double(), X) * (UHtrain.double() @ B.double() @ UHtest.double().t())
            with variable_required_grad(Xtest):
                torch.autograd.gradcheck(kb_star_func, Xtest.double())
            self.to(dtype=old_dtype)
        # 2. α := Lᵀ \ ( L \ Y )
        α = torch.cholesky_solve(Y, Kb_sqrt) # check the shape of Y
        # 3. μ := μ(x) + kb*ᵀ α
        mean = fu_mean_test + kb_star.t() @ α

        if compute_cov:
            kb_star_p = (k_xs(Xtrain, Xtestp) * (UHtrain @ B @ UHtestp.t())
                     if Xtestp_in is not None
                     else kb_star)
            kb_star_starp = k_ss(Xtest, Xtestp) * (UHtest @ B @ UHtestp.t())
            if grad_check:
                old_dtype = self.dtype
                self.double_()
                kb_star_starp_func = lambda X: k_ss(X, Xtestp.double()) * (UHtest @ B @ UHtestp.t()).double()
                with variable_required_grad(Xtest):
                    torch.autograd.gradcheck(kb_star_starp_func, Xtest.double())
                    kb_star_star_func = lambda X, Xp: k_ss(X, Xp) * (UHtest @ B @ UHtestp.t()).double()
                    gradgradcheck(kb_star_star_func, Xtest.double())
                self.to(dtype=old_dtype)

            # 4. v := L \ kb*
            v = torch.solve(kb_star, Kb_sqrt).solution

            if grad_check:
                old_dtype = self.dtype
                self.double_()
                v_func = lambda X: torch.solve(kb_star_func(X), Kb_sqrt.double()).solution
                with variable_required_grad(Xtest):
                    torch.autograd.gradcheck(v_func, Xtest.double())
                self.to(dtype=old_dtype)

            vp = torch.solve(kb_star_p, Kb_sqrt).solution if Xtestp_in is not None else v

            if grad_check:
                old_dtype = self.dtype
                self.double_()
                v_func = lambda X: torch.solve(kb_star_func(X), Kb_sqrt.double()).solution
                with variable_required_grad(Xtest):
                    torch.autograd.gradcheck(v_func, Xtest.double())
                self.to(dtype=old_dtype)

            # 5. k* := k(x*,x*) - vᵀv
            scalar_var = kb_star_starp - v.t() @ vp
            if grad_check:
                old_dtype = self.dtype
                self.double_()
                scalar_var_func = lambda X: (
                    kb_star_starp_func(X)
                    - v_func(X).t() @ v_func(Xtestp.double()))
                with variable_required_grad(Xtest):
                    torch.autograd.gradcheck(scalar_var_func, Xtest.double())
                    scalar_var_XX_func = lambda X, Xp: (
                        kb_star_star_func(X, Xp)
                        - v_func(X).t() @ v_func(Xp))
                    gradgradcheck(scalar_var_XX_func, Xtest.double())
                self.model.float()
                self.to(dtype=old_dtype)

            covar_mat = torch_kron(scalar_var.unsqueeze(0), A.unsqueeze(0))
            if grad_check:
                old_dtype = self.dtype
                self.double_()
                covar_mat_func = lambda X: (scalar_var_func(X).reshape(-1, 1, 1) * A.double())[0,0]
                with variable_required_grad(Xtest):
                    torch.autograd.gradcheck(covar_mat_func, Xtest.double())
                self.model.float()
                self.to(dtype=old_dtype)
        else: # if not compute_cov
            covar_mat = 0 * A
        return mean, (scalar_var if scalar_var_only else covar_mat)

    @property
    def dtype(self):
        return next(self.model.parameters())[0].dtype

    def to(self, dtype=torch.float64):
        if dtype is torch.float64:
            self.double_()
        else:
            self.float_()

    def double_(self):
        self.model.double()
        assert self.dtype is torch.float64
        self.model.train_inputs = tuple([
            inp.double()
            for inp in self.model.train_inputs])
        self.model.train_targets = self.model.train_targets.double()
        for k, v in self._cache.items():
            self._cache[k] = v.double()

    def float_(self):
        self.model.float()
        assert self.dtype is torch.float32
        self.model.train_inputs = tuple([
            inp.float()
            for inp in self.model.train_inputs])
        self.model.train_targets = self.model.train_targets.float()
        for k, v in self._cache.items():
            self._cache[k] = v.float()

    def _predict_flatten(self, Xtest_in, Utest_in):
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
        Xtest = Xtest.to(device=device, dtype=self.dtype)

        if isinstance(Utest_in, np.ndarray):
            Utest = torch.from_numpy(Utest_in)
        else:
            Utest = Utest_in
        Utest = Utest.to(device=device, dtype=self.dtype)

        # Switch back to eval mode
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Call fit() with training data before calling predict")

        # Set in eval mode
        self.model.eval()
        self.likelihood.eval()

        # Concatenate the test set
        _, MXUHtest = self.model.encode_from_XU(Xtest, Utrain=Utest, M=1)
        output = self.model(MXUHtest)

        mean = output.mean.reshape(Xtest.shape[0], -1)
        cov =  output.covariance_matrix.reshape(Xtest.shape[0],
                                                mean.shape[-1], mean.shape[-1],
                                                Xtest.shape[0])
        return (mean.to(device=Xtest_in.device, dtype=Xtest_in.dtype),
                cov.to(device=Xtest_in.device, dtype=Xtest_in.dtype))

    def f_func(self, Xtest_in, return_cov=False):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        Utest = Xtest.new_zeros((Xtest.shape[0], self.u_dim))
        #mean_fx, cov_fx = self._predict_flatten(Xtest, Utest)
        mean_fx, cov_fx = self.custom_predict(Xtest, Utest)
        if return_cov:
            if Xtest_in.ndim == 1:
                cov_fx = cov_fx.squeeze(0)
            cov_fx = cov_fx.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        if Xtest_in.ndim == 1:
            mean_fx = mean_fx.squeeze(0)
        mean_fx = mean_fx.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        return (mean_fx, cov_fx) if return_cov else mean_fx

    def _A_mat(self):
        return self.model.covar_module.task_covar_module.U.covar_matrix.evaluate()

    def f_func_mean(self, Xtest_in):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        mean_f, _ =  self.custom_predict(Xtest, compute_cov=False)
        if Xtest_in.ndim == 1:
            mean_f = mean_f.squeeze(0)
        return mean_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)

    def f_func_knl(self, Xtest_in, Xtestp_in, grad_check=False):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        Xtestp = (Xtestp_in.unsqueeze(0)
                 if Xtestp_in.ndim == 1
                 else Xtestp_in)
        _, var_f =  self.custom_predict(Xtest, Xtestp_in=Xtestp, compute_cov=True)
        if Xtest_in.ndim == 1:
            var_f = var_f.squeeze(0)
        var_f_out  = var_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)

        if grad_check:
            old_dtype = self.dtype
            self.double_()
            var_f_func = lambda X: self.custom_predict(
                X, Xtestp_in=Xtestp, compute_cov=True)[1][0,0,0]
            with variable_required_grad(Xtest):
                torch.autograd.gradcheck(var_f_func, Xtest.double())
                var_f_func_2 = lambda X, Xp: self.custom_predict(
                    X, Xtestp_in=Xp, compute_cov=True)[1][0,0,0]
                gradgradcheck(var_f_func_2, Xtest.double())
            self.model.float()
            self.to(dtype=old_dtype)
        return var_f_out

    def f_func_gp(self):
        #return GaussianProcess(self.f_func_mean, self.f_func_knl, (self.x_dim,))
        return self._f_func_gp

    def fu_func_mean(self, Utest_in, Xtest_in):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        Utest = (Utest_in.unsqueeze(0)
                 if Utest_in.ndim == 1
                 else Utest_in)
        mean_f, _ =  self.custom_predict(Xtest, Utest, compute_cov=False)
        if Xtest_in.ndim == 1:
            mean_f = mean_f.squeeze(0)
        mean_f = mean_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        return mean_f

    def _grad_fu_func_mean(self, Xtest_in, Utest_in=None):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        Utest = (Utest_in.unsqueeze(0)
                 if Utest_in is not None and Utest_in.ndim == 1
                 else Utest_in)
        mean_f, _ = self.custom_predict(Xtest, Utest, compute_cov=False,
                                        grad_gp=True)
        if Xtest_in.ndim == 1:
            mean_f = mean_f.squeeze(0)
        mean_f = mean_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        return mean_f

    def fu_func_knl(self, Utest_in, Xtest_in, Xtestp_in):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        Utest = (Utest_in.unsqueeze(0)
                 if Utest_in.ndim == 1
                 else Utest_in)
        Xtestp = (Xtestp_in.unsqueeze(0)
                 if Xtestp_in.ndim == 1
                 else Xtestp_in)
        _, var_f = self.custom_predict(Xtest, Utest,
                                       Xtestp_in=Xtestp,
                                       compute_cov=True)
        if Xtest_in.ndim == 1:
            var_f = var_f.squeeze(0)
        var_f = var_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        return var_f


    def fu_func_gp(self, Utest_in):
        gp = GaussianProcess(mean=partial(self.fu_func_mean, Utest_in),
                               knl=partial(self.fu_func_knl, Utest_in),
                               shape=(self.x_dim,),
                             name="F(.)u")
        gp.register_covar(self._f_func_gp, partial(self.covar_fu_f, Utest_in))
        return gp

    def covar_fu_f(self, Utest_in, Xtest_in, Xtestp_in):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        Utest = (Utest_in.unsqueeze(0)
                 if Utest_in.ndim == 1
                 else Utest_in)
        Xtestp = (Xtestp_in.unsqueeze(0)
                 if Xtestp_in.ndim == 1
                 else Xtestp_in)
        Utestp = torch.zeros_like(Utest)
        mean_f, var_f = self.custom_predict(Xtest, Utest,
                                            Xtestp_in=Xtestp,
                                            Utestp_in=Utestp,
                                            compute_cov=True)
        if Xtest_in.ndim == 1:
            var_f = var_f.squeeze(0)
        var_f = var_f.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        return var_f

    def g_func(self, Xtest_in, return_cov=False):
        assert not return_cov, "Don't know what matrix covariance looks like"
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        mean_Fx = self.predict(Xtest, return_cov=return_cov)
        mean_gx = mean_Fx[:, 1:, :]
        if Xtest_in.ndim == 1:
            mean_gx = mean_gx.squeeze(0)
        mean_gx = mean_gx.to(dtype=Xtest_in.dtype, device=Xtest_in.device)
        return mean_gx.transpose(-2, -1)

    def _gu_func(self, Xtest_in, Utest_in=None, return_cov=False, Xtestp_in=None):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        if Utest_in is not None:
            Utest = (Utest_in.unsqueeze(0)
                    if Utest_in.ndim == 1
                    else Utest_in)
        else:
            Utest = Xtest_in.new_ones(Xtest.shape[0], self.u_dim)
        mean_gu, var_gu = self.custom_predict(Xtest, Utest, UHfill=0,
                                              Xtestp_in=Xtestp_in,
                                              compute_cov=True)
        if Xtest_in.ndim == 1 and Utest_in.ndim == 1:
            mean_gu = mean_gu.squeeze(0)
            var_gu = var_gu.squeeze(0)
        return (mean_gu, var_gu) if return_cov else mean_gu

    def g_func_mean(self, Xtest_in):
        return self._gu_func(Xtest_in, return_cov=False)

    def _cbf_func(self, Xtest, grad_htest, return_cov=False):
        if return_cov:
            mean_Fx, cov_Fx = self.predict(Xtest, return_cov=True)
            cov_hFT = grad_htest.T @ cov_Fx @ grad_htest
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


def is_psd(X):
    try:
        torch.cholesky(X)
    except RuntimeError as e:
        print(e)
        return False
    return True

def make_psd(Kb,
             cholesky_tries=10,
             cholesky_perturb_init=1e-5,
             cholesky_perturb_scale=10):
    cholesky_perturb_factor = cholesky_perturb_init
    Kb_sqrt = None
    for ntry in range(cholesky_tries):
        try:
            Kbp = Kb + cholesky_perturb_factor * (
                torch.eye(Kb.shape[0], dtype=Kb.dtype, device=Kb.device) *
                torch.rand(Kb.shape[0], dtype=Kb.dtype, device=Kb.device)
                )
            Kb_sqrt = torch.cholesky(Kbp)
            break
        except RuntimeError as e:
            if ntry == cholesky_tries - 1:
                raise
            else:
                LOG.warning("Cholesky failed with perturb={} on error {}".format(cholesky_perturb_factor, str(e)))
                cholesky_perturb_factor = cholesky_perturb_factor * cholesky_perturb_scale
                continue

    return Kbp, Kb_sqrt

ControlAffineRegressorRankOne = partial(
    ControlAffineRegressor,
    model_class=partial(ControlAffineExactGP,
                        rank=1,
                        gamma_length_scale_prior=(1e-3, 1e-3)))


class ControlAffineRegressorExact(ControlAffineRegressor):
    def custom_predict(self, Xtest_in, Utest_in=None, UHfill=1, Xtestp_in=None,
                       Utestp_in=None, UHfillp=1,
                       compute_cov=True):
        Xtest = self._ensure_device_dtype(Xtest_in)
        Xtestp = (self._ensure_device_dtype(Xtestp_in) if Xtestp_in is not None
                  else Xtest)
        meanFX, A, BkXX = self._custom_predict_matrix(Xtest_in, Xtestp_in,
                                                     compute_cov=compute_cov)
        if Utest_in is None:
            UHtest = Xtest.new_zeros(Xtest.shape[0], self.model.matshape[0])
            UHtest[:, 0] = 1
        else:
            Utest = self._ensure_device_dtype(Utest_in)
            UHtest = torch.cat((Utest.new_full((Utest.shape[0], 1), UHfill),
            Utest), dim=-1)
        if Utestp_in is None:
            UHtestp = UHtest
        else:
            Utestp = self._ensure_device_dtype(Utestp_in)
            UHtestp = torch.cat((Utest.new_full((Utestp.shape[0], 1), UHfillp),
                                 Utestp), dim=-1)
        UHtest_BkXX = UHtest.unsqueeze(-1).unsqueeze(1) #  (k', 1, (1+m), 1)
        UHtestp_BkXX = UHtestp.unsqueeze(-1).unsqueeze(0) #  (1, k', (1+m), 1)
        meanFXU = meanFX.bmm(UHtest.unsqueeze(-1)).squeeze(-1)
        if compute_cov:
            varFXU = torch.matmul(
                torch.matmul(UHtest_BkXX.transpose(-2, -1), BkXX),
                UHtestp_BkXX) * A
        else:
            varFXU = Xtest.new_zeros(Xtest.shape[0], Xtestp.shape[0], *A.shape)
        return (meanFXU, varFXU)

    def custom_predict_fullmat(self, Xtest_in, Xtestp_in=None):
        Xtest = self._ensure_device_dtype(Xtest_in)
        Xtestp = (self._ensure_device_dtype(Xtestp_in) if Xtestp_in is not None
                  else Xtest)
        meanFX, A, BkXX = self._custom_predict_matrix(Xtest_in, Xtestp_in,
                                                      compute_cov=True)
        assert not torch.isnan(meanFX).any()
        b = Xtest.shape[0]
        m = self.u_dim
        n = self.x_dim
        assert meanFX.shape == (b, n, (1+m))
        meanFX = meanFX.transpose(-2, -1) # (b, (1+m), n)
        var_FX = torch_kron(BkXX.transpose(2, 1).reshape(b*(1+m), b*(1+m)), # (b(1+m), b(1+m))
                            A, # (n, n)
                            batch_dims=0) # (b(1+m)n, b(1+m)n)
        # assert is_psd(var_FX)
        # (b(1+m)n), (b(1+m)n, b(1+m)n)
        return meanFX.reshape(-1), var_FX


    def _custom_predict_matrix(self, Xtest_in, Xtestp_in=None, compute_cov=True):
        """

        Matrix variate GP: Separate A and B

            F(x) ~ 𝕄𝕍ℙ(𝐌(x), 𝐀, 𝐁 k(x, x'))                ∈ (n, 1+m)

            𝔅(XU, XU) = [𝐮ᵢᵀB𝐮ⱼ (k(xᵢ, xᵢ)+σ²)]ᵢⱼ            ∈ (k, k)
            𝔅(XU, x*) = [𝐮ᵢᵀB (k(xᵢ, x*)+σ²)]ᵢ               ∈ (k(1+m), k)
            𝐌(XU) = [𝐌(xᵢ)𝐮ᵢ]ᵢ                              ∈ (n, k)

            F*(x*) ~ 𝕄𝕍ℙ(
                       𝐌(x*) + (Ẋ - 𝐌(XU))[𝔅(XU, XU)]⁻¹(𝔅(XU, x*)ᵀ),
                        A,
                       B k(x*, x*) - 𝔅(XU, x*)[𝔅(XU, XU)]⁻¹(𝔅(XU, x*)ᵀ)
                     )

        Algorithm (Rasmussen and Williams 2006)
           1. L := cholesky(𝔅(XU, XU))                                 O(k³)
           2. B† :=  ( (LLᵀ) \ 𝔅(XU, x*)ᵀ )             ∈ (k, k(1+m)))  O(k²(1+m))
           3. Y = (Ẋ - 𝐌(XU))                          ∈ (n, k)        O(kn(1+m))
           3. 𝐌ₖ(x*) := 𝐌(x*) +  Y @ B†               ∈ (n, (1+m))    O(nk²(1+m))
           4. 𝐁ₖ(x*, x*) := B k(x*,x*) - 𝔅(XU, x*) @ B† ∈ (1+m, 1+m)  O(k²(1+m)²)
           5. log p(y|X) := -0.5  Y @ ( (LLᵀ) \ Y )  - ∑ log Lᵢᵢ - 0.5 n log(2π)

        """
        Xtest = self._ensure_device_dtype(Xtest_in)
        Xtestp = (self._ensure_device_dtype(Xtestp_in) if Xtestp_in is not None
                  else Xtest)
        k_xx = lambda x, xp: self.model.covar_module.data_covar_module(
            x, xp).evaluate()
        k_ss = k_xs = k_sx = k_xx
        mean_s = self.model.mean_module
        A = self.model.covar_module.task_covar_module.U.covar_matrix.evaluate()
        B = self.model.covar_module.task_covar_module.V.covar_matrix.evaluate()
        # Output of mean_s(Xtest) is (b, (1+m)n)
        # Make it (b, (1+m), n, 1) then transpose
        # (b, n, 1, (1+m)) and multiply with UHtest (b, (1+m)) to get
        # (b, n, 1)
        fX_mean_test = mean_s(Xtest).reshape(
            Xtest.shape[0], *self.model.matshape).transpose(-2, -1) # (B, 1+m, n) -> (B, n, 1+m)
        if self.model.train_inputs is None:
            # 5. k* := k(x*,x*) B
            return fX_mean_test, A, B * k_ss(Xtest, Xtestp).unsqueeze(-1).unsqueeze(-1)

        MXUHtrain = self.model.train_inputs[0]
        Mtrain, Xtrain, UHtrain = self.model.decoder.decode(MXUHtrain)
        nsamples = Xtrain.size(0)

        # Y₁ₖ = Ẋ₁ₖ - 𝐌₁ₖ𝔘₁ₖ
        MXtrain = self.model.mean_module(Xtrain) # (k, (1+m)n)
        Y = (
            self.model.train_targets.reshape(nsamples, -1) # (k, n)
            - (MXtrain.reshape(nsamples, *self.model.matshape) # (k, (1+m), n)
               .transpose(-2,-1) # (k, n, (1+m))
               .bmm(
                   UHtrain.unsqueeze(-1) # (k, (1+m), 1)
               ) # (k, n, 1)
               .squeeze(-1)) # (k, n)
        ) # (k, n)
        # 1. L := cholesky(𝔅(XU, XU)) or LLᵀ = 𝔅(XU, XU)
        # Kb_sqrt = L
        Kb_sqrt = self._perturbed_cholesky(k_xx, B, Xtrain, UHtrain) # (k, k)

        # kb_star = 𝔅(XU, x*)
        # k_xs(Xtrain, Xtest) \in (k, b)
        # UHtrain \in (k, (1+m))
        # B \in (1+m, 1+m)
        kb_star = k_sx(Xtest, Xtrain).unsqueeze(-1) *  (UHtrain @ B).unsqueeze(0) # (b, k, (1+m))
        # 2. B†(x) := ( LLᵀ) \ (𝔅(XU, x)ᵀ)
        Bdagger = torch.cholesky_solve(kb_star, Kb_sqrt) # (b, k, (1+m))
        # 3. 𝐌ₖ := 𝐌₀(x) + Y₁ₖ B†(x)
        mean_k = fX_mean_test + torch.matmul(Y.t().unsqueeze(0), Bdagger) # (b, n, (1+m))

        if compute_cov:
            # 4. 𝐁ₖ(x*, x*) := B k(x*,x*) - 𝔅(XU, x*) @ B†
            k_xtest_xtestp = k_ss(Xtest, Xtestp) # (b, b)
            #k_xtest_xtestp, _ = make_psd(k_xtest_xtestp)
            #assert is_psd(B)
            KXTestBXTestB = torch_kron(
                k_xtest_xtestp, B, batch_dims=0) # (b(1+m), b(1+m))
            # assert is_psd(KXTestBXTestB)
            # assert is_psd(
            #     torch.cat([
            #         torch.cat([Kb_sqrt @ Kb_sqrt.t(),
            #                    kb_star.reshape(-1, b*(1+m))], dim=-1),
            #         torch.cat([kb_star.transpose(0, 2).reshape(b*(1+m),-1),
            #                    KXTestBXTestB], dim=-1)
            #     ], dim=0)
            # )
            # v = Lᵀ 𝔅(XU, x*)
            # v = torch.solve(kb_star, Kb_sqrt).solution # (b, k, (1+m))
            b = Xtest.shape[0]
            m = self.u_dim
            n = self.x_dim
            k = Xtrain.shape[0]
            BkXX = (
                KXTestBXTestB # (b(1+m), b(1+m))
                - (
                    kb_star.transpose(-2, -1) # (b, (1+m), k)
                   .reshape(b*(1+m), k) # (b (1+m), k)
                   @
                    (Bdagger.transpose(1, 0) # (k, b, (1+m))
                     .reshape(k, b*(1+m))) # (k, b(1+m))
                ) # (b(1+m), b(1+m))
            )
            BkXX, _ = make_psd(BkXX)
            # assert is_psd(BkXX)
            BkXX = BkXX.reshape(b, (1+m), b, (1+m)).transpose(1, 2) # (b, b, (1+m), (1+m))
        else:
            m = self.u_dim
            BkXX = Xtest.new_zeros(Xtest.shape[0], Xtestp.shape[0], (1+m), (1+m))
        # (b, n, (1+m)), (n, n), (b, b, (1+m), (1+m))
        return mean_k, A, BkXX


ControlAffineRegressorExactRankOne = partial(
    ControlAffineRegressorExact,
    model_class=partial(ControlAffineExactGP,
                        rank=1))



class ControlAffineVectorGP(ControlAffineExactGP):
    def __init__(self, x_dim, u_dim, likelihood, rank=None, gamma_length_scale_prior=None):
        ExactGP.__init__(self, None, None, likelihood)
        self.matshape = (1+u_dim, x_dim)
        self.decoder = CatEncoder(1, x_dim, 1+u_dim)
        self.mean_module = HetergeneousMatrixVariateMean(
            ConstantMean(),
            self.decoder,
            self.matshape)
        num_tasks=np.prod(self.matshape)
        self.task_covar = IndexKernel(num_tasks=num_tasks,
                                      rank=(num_tasks if rank is None else rank))

        prior_args = dict() if gamma_length_scale_prior is None else dict(
            lengthscale_prior=GammaPrior(*gamma_length_scale_prior))
        self.input_covar = ScaleKernel(
            RBFKernel(**prior_args) + LinearKernel())
        self.covar_module = HetergeneousCoregionalizationKernel(
            self.task_covar,
            self.input_covar,
            self.decoder,
        )

class ControlAffineRegressorVector(ControlAffineRegressor):
    def __init__(self, *args, model_class=ControlAffineVectorGP, **kwargs):
        super().__init__(*args, model_class=model_class, **kwargs)

    def custom_predict(self, Xtest_in, Utest_in=None, UHfill=1, Xtestp_in=None,
                       Utestp_in=None, UHfillp=1,
                       compute_cov=True):
        Xtest = self._ensure_device_dtype(Xtest_in)
        Xtestp = (self._ensure_device_dtype(Xtestp_in) if Xtestp_in is not None
                  else Xtest)
        meanFX, KkXX = self._custom_predict_matrix(Xtest_in, Xtestp_in,
                                                     compute_cov=compute_cov)
        if Utest_in is None:
            UHtest = Xtest.new_zeros(Xtest.shape[0], self.model.matshape[0])
            UHtest[:, 0] = 1
        else:
            Utest = self._ensure_device_dtype(Utest_in)
            UHtest = torch.cat((Utest.new_full((Utest.shape[0], 1), UHfill),
            Utest), dim=-1)
        if Utestp_in is None:
            UHtestp = UHtest
        else:
            Utestp = self._ensure_device_dtype(Utestp_in)
            UHtestp = torch.cat((Utest.new_full((Utestp.shape[0], 1), UHfillp),
                                 Utestp), dim=-1)
        meanFXU = meanFX.bmm(UHtest.unsqueeze(-1)).squeeze(-1)
        if compute_cov:
            k, n = Xtest.shape
            In = torch.eye(n,
                        dtype=Xtest.dtype,
                        device=Xtest.device) # (n, n)
            UHtest_block = torch_kron(UHtest, In, batch_dims=0).reshape(k, 1, n, -1) # (k, 1, n, (1+m)n)
            UHtest_block_T = UHtest_block.reshape(1, k, n, -1).transpose(-2, -1) # (1, k, (1+m)n), n)

            varFXU = torch.matmul(
                torch.matmul(UHtest_block, KkXX),
                UHtest_block_T)
        else:
            k, n = Xtest.shape
            varFXU = Xtest.new_zeros(Xtest.shape[0], Xtestp.shape[0], n, n)
        return (meanFXU, varFXU)

    def custom_predict_fullmat(self, Xtest_in, Xtestp_in=None):
        Xtest = self._ensure_device_dtype(Xtest_in)
        Xtestp = (self._ensure_device_dtype(Xtestp_in) if Xtestp_in is not None
                  else Xtest)
        meanFX, varFX = self._custom_predict_matrix(Xtest_in, Xtestp_in,
                                                    compute_cov=True)
        b = Xtest.shape[0]
        m = self.u_dim
        n = self.x_dim
        assert meanFX.shape == (b, n, (1+m))
        meanFX = meanFX.transpose(-2, -1) # (b, (1+m), n)
        return (meanFX.reshape(-1), # (b(1+m)n)
                varFX # (b, b, (1+m)n, (1+m)n))
                .transpose(2, 1) # (b, (1+m)n, b, (1+m)n)
                .reshape(b*(1+m)*n, b*(1+m)*n))


    def _perturbed_cholesky_compute(self,
                                    k_xx, # function
                                    Σ, # ((1+m)n, (1+m)n)
                                    Xtrain, # (k, n)
                                    UHtrain, # (k, (1+m))
                                    cholesky_tries=10,
                                    cholesky_perturb_init=1e-5,
                                    cholesky_perturb_scale=10):
        k, n = Xtrain.shape
        In = torch.eye(n,
                       dtype=Xtrain.dtype,
                       device=Xtrain.device) # (n, n)
        KXX = k_xx(Xtrain, Xtrain) # (k, k)
        UHtrain_block = torch_kron(UHtrain, In, batch_dims=0).reshape(k, 1, n, -1) # (k, 1, n, (1+m)n)
        UHtrain_block_T = UHtrain_block.reshape(1, k, n, -1).transpose(-2, -1) # (1, k, (1+m)n), n)
        uΣu = torch.matmul(
            torch.matmul(UHtrain_block, Σ.unsqueeze(-3).unsqueeze(-4)),
            UHtrain_block_T) # (k, k, n, n)
        Kb = ((
            KXX.reshape(k, k, 1, 1) * uΣu # (k, k, n, n)
        ).transpose(1, 2) # (k, n, k, n)
              .reshape(k*n, k*n)
        )

        # Kb can be singular because of repeated datasamples
        # Add diagonal jitter
        Kbp, Kb_sqrt = make_psd(Kb,
                                cholesky_tries=cholesky_tries,
                                cholesky_perturb_init=cholesky_perturb_init,
                                cholesky_perturb_scale=cholesky_perturb_scale)
        return Kb_sqrt

    def _custom_predict_matrix(self, Xtest_in, Xtestp_in=None, compute_cov=True):
        """

        Vector variate GP: Σ                                      ∈ ((1+m)n, (1+m)n)

            vec(F(x)) ~ 𝔾ℙ(vec(𝐌(x)), Σ k(x, x'))                ∈ ((1+m)n, 1)

            𝔎(XU, XU) = [(𝐮ᵢᵀ ⊗ Iₙ) Σ (𝐮ⱼ ⊗ Iₙ) (k(xᵢ, xᵢ)+σ²)]ᵢⱼ ∈ (kn, kn)
            𝔎(XU, x*) = [(𝐮ᵢᵀ ⊗ Iₙ) Σ (k(xᵢ, x*)+σ²)]ᵢ            ∈ (kn, (1+m)n)
            𝐌(XU) = [𝐌(xᵢ)𝐮ᵢ]ᵢ                                   ∈ (k, n)

            vec(F*(x*)) ~ 𝔾ℙ(
                       vec[𝐌(x*) + (Ẋ - 𝐌(XU))[𝔎(XU, XU)]⁻¹(𝔎(XU, x*)ᵀ)],
                       Σ k(x*, x*) - 𝔎(XU, x*)[𝔎(XU, XU)]⁻¹(𝔎(XU, x*)ᵀ)
                     )

        Algorithm (Rasmussen and Williams 2006)
           1. L := cholesky(𝔎(XU, XU))                          O(k³n³)
           2. Y = vec(Ẋ - 𝐌(XU))               ∈ (kn, 1)        O(kn)
           2. α :=  ( (LLᵀ) \ Y )               ∈ (kn, 1)       O(k³n)
           3. 𝐌ₖ(x*) := 𝐌(x*) +  𝔎(XU, x*)ᵀ α ∈  ((1+m)n, 1)   O(k²n³(1+m))
           4. v(x*) = L \ 𝔎(XU, x*)            ∈ (kn, (1+m)n)   O(k²n³(1+m))
           5. Σₖ(x, x') = Σ₀(x, x') - v(x*)ᵀ v(x*)  ∈ ((1+m)n, (1+m)n))   O(k²n⁴(1+m)²))
           6. log p(y|X) := -0.5  Y @ ( (LLᵀ) \ Y )  - ∑ log Lᵢᵢ - 0.5 n log(2π)

        """
        Xtest = self._ensure_device_dtype(Xtest_in)
        Xtestp = (self._ensure_device_dtype(Xtestp_in) if Xtestp_in is not None
                  else Xtest)
        k_xx = lambda x, xp: self.model.covar_module.data_covar_module(
            x, xp).evaluate()
        mean_s = self.model.mean_module
        Σ = self.model.covar_module.task_covar_module.covar_matrix.evaluate() # ((1+m)n, (1+m)n)

        # Output of mean_s(Xtest) is (b, (1+m)n)
        # Make it (b, (1+m), n, 1) then transpose
        # (b, n, 1, (1+m)) and multiply with UHtest (b, (1+m)) to get
        # (b, n, 1)
        fX_mean_test = mean_s(Xtest).reshape(
            Xtest.shape[0], *self.model.matshape).transpose(-2, -1) # (b, 1+m, n) -> (B, n, 1+m)
        if self.model.train_inputs is None:
            # 5. k* := k(x*,x*) B
            return fX_mean_test, Σ * k_xx(Xtest, Xtestp).unsqueeze(-1).unsqueeze(-1)

        MXUHtrain = self.model.train_inputs[0]
        Mtrain, Xtrain, UHtrain = self.model.decoder.decode(MXUHtrain)
        nsamples = Xtrain.size(0)

        # Y₁ₖ = Ẋ₁ₖ - 𝐌₁ₖ𝔘₁ₖ
        MXtrain = self.model.mean_module(Xtrain) # (k, (1+m)n)
        Y = (
            self.model.train_targets.reshape(nsamples, -1) # (k, n)
            - (MXtrain.reshape(nsamples, *self.model.matshape) # (k, (1+m), n)
               .transpose(-2,-1) # (k, n, (1+m))
               .bmm(
                   UHtrain.unsqueeze(-1) # (k, (1+m), 1)
               ) # (k, n, 1)
               .squeeze(-1)) # (k, n)
        ) # (k, n)
        # 1. L := cholesky(𝔎(XU, XU)) or LLᵀ = 𝔎(XU, XU)
        # Kb_sqrt = L
        Kb_sqrt = self._perturbed_cholesky(k_xx, Σ, Xtrain, UHtrain) # (kn, kn)

        # kb_star = 𝔎(XU, x*)
        # k_xs(Xtrain, Xtest) \in (k, b)
        # UHtrain \in (k, (1+m))
        # B \in (1+m, 1+m)
        k, n = Xtrain.shape
        b = Xtest.shape[0]
        In = torch.eye(n, dtype=Xtrain.dtype, device=Xtrain.device) # (n, n)
        UHtrain_block = torch_kron(UHtrain, In, batch_dims=0) # (kn, (1+m)n)
        kb_star = (
            k_xx(Xtest, Xtrain).reshape(b, k, 1, 1) # (b, k, 1, 1)
            *
            (UHtrain_block @ Σ # (kn, (1+m)n)
            ).reshape(1, k, n, -1) # (1, k, n, (1+m)n)
        ).reshape(b, k*n, -1) # (b, kn, (1+m)n)
        # 2. α := ( LLᵀ) \ Y
        α = torch.cholesky_solve(Y.reshape(-1, 1), Kb_sqrt) # (kn, 1)
        # 3. 𝐌ₖ := 𝐌₀(x) + 𝔎(XU, x*)ᵀ α
        mean_k = fX_mean_test + (
            torch.matmul(kb_star.transpose(-2, -1), # (b, (1+m)n, kn)
                         α.unsqueeze(0)) # (1, kn, 1)
            .reshape(b, -1, n) # (b, (1+m)n, 1)
            .transpose(-2, -1) # (b, n, (1+m))
            )

        if compute_cov:
            # 4. v = L \ 𝔎(XU, x*)
            # 5. Bₖ(x, x') = B₀(x, x') - vᵀ v
            v = torch.solve(kb_star, # (b, kn, (1+m)n)
                            Kb_sqrt # (kn, kn)
                            ).solution # (b, kn, (1+m)n)
            b = Xtest.shape[0]
            m = self.u_dim
            n = self.x_dim
            k = Xtrain.shape[0]
            vb = v.transpose(0, 1).reshape(k*n, b*(1+m)*n) # (kn, b(1+m)n)
            KkXX = (
                torch_kron(k_xx(Xtest, Xtestp),
                           Σ, batch_dims=0) # (b(1+m)n, b(1+m)n)
                - vb.t() @ vb # (b(1+m)n, b(1+m)n)
            )
            KkXX, _ = make_psd(KkXX)
            KkXX = KkXX.reshape(b, (1+m)*n, b, (1+m)*n).transpose(2, 1)
        else:
            n = self.model.matshape[1]
            m = self.model.matshape[0]-1
            KkXX = Xtest.new_zeros(Xtest.shape[0], Xtestp.shape[0], (1+m)*n, (1+m)*n)
        # (b, n, (1+m)), (b, b, (1+m)n, (1+m)n)
        return mean_k, KkXX


ControlAffineRegMatrixDiagGP = partial(ControlAffineExactGP, rank=0)
"""
IndexKernel models are BBᵀ + diag(𝐯) where B is a low rank matrix whose rank is
controlled by the rank parameters.
"""

ControlAffineRegMatrixDiag = partial(ControlAffineRegressorExact,
                                  model_class=ControlAffineRegMatrixDiagGP)
"""
Regressor with Independendent REgresssor
"""

ControlAffineRegVectorDiagGP = partial(ControlAffineVectorGP, rank=0)
"""
IndexKernel models are BBᵀ + diag(𝐯) where B is a low rank matrix whose rank is
controlled by the rank parameters.
"""

ControlAffineRegVectorDiag = partial(ControlAffineRegressorVector,
                                     model_class=ControlAffineRegVectorDiagGP)
"""
Regressor with Independendent REgresssor
"""

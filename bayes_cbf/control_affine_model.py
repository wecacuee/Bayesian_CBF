from pathlib import Path
import warnings
from typing import Any
from itertools import zip_longest
from collections import namedtuple
from functools import partial

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
from gpytorch.priors import GammaPrior
from gpytorch.utils.memoize import cached
import gpytorch.settings as gpsettings

from bayes_cbf.matrix_variate_multitask_kernel import MatrixVariateIndexKernel, HetergeneousMatrixVariateKernel
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



class DifferentiableGaussianProcess(torch.autograd.Function):
    @staticmethod
    def forward(ctx, regressor, compute_cov, Utest, Xtest):
        ctx.save_for_backward(regressor, return_cov, Utest, Xtest)
        mean, cov = regressor.custom_predict(Xtest, Utest, compute_cov=compute_cov)
        return mean, cov # 2 inputs to backward

    @staticmethod
    def backward(ctx, grad_mean, grad_cov):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]
        regressor, compute_cov, Utest, Xtest = ctx.saved_tensors

        grad_Xtest = None
        if ctx.needs_input_grad[2]:
            mean, cov = self.regressor.custom_predict( Xtest, Utest,
                grad_gp=True, compute_cov=compute_cov)
            grad_Xtest = grad_mean @ mean + grad_cov @ cov
        return None, None, None, grad_Xtest # 4 inputs to foward


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
    def __init__(self, x_dim, u_dim, likelihood, rank=1, gamma_length_scale_prior=None):
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
        prior_args = dict() if gamma_length_scale_prior is None else dict(
            lengthscale_prior=GammaPrior(*gamma_length_scale_prior))
        self.input_covar = ScaleKernel(
            RBFKernel(**prior_args))
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
                 gamma_length_scale_prior=None):
        self.device = device or default_device()
        self.x_dim = x_dim
        self.u_dim = u_dim
        # Initialize model and likelihood
        # Noise model for GPs
        self.likelihood = IdentityLikelihood()
        # Actual model
        self.model = ControlAffineExactGP(
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

    def _perturbed_cholesky_compute(self, k, B, Xtrain, UHtrain,
                           cholesky_tries=10,
                           cholesky_perturb_init=1e-5,
                           cholesky_perturb_scale=10):
        KXX = k(Xtrain, Xtrain)
        uBu = UHtrain @ B @ UHtrain.T
        Kb = KXX * uBu

        # Kb can be singular because of repeated datasamples
        # Add diagonal jitter
        Kb_sqrt = None
        cholesky_perturb_factor = cholesky_perturb_init
        for _ in range(cholesky_tries):
            try:
                Kbp = Kb + cholesky_perturb_factor * (
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

        TODO: Write the function like this some day
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
            return fu_mean_test, k_ss(Xtest, Xtest) * A

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

    def _gu_func(self, Xtest_in, Utest_in, return_cov=False, Xtestp_in=None):
        Xtest = (Xtest_in.unsqueeze(0)
                 if Xtest_in.ndim == 1
                 else Xtest_in)
        Utest = (Utest_in.unsqueeze(0)
                 if Utest_in.ndim == 1
                 else Utest_in)
        mean_gu, var_gu = self.custom_predict(Xtest, Utest, UHfill=0,
                                              Xtestp_in=Xtestp_in,
                                              compute_cov=True)
        if Xtest_in.ndim == 1 and Utest_in.ndim == 1:
            mean_gu = mean_gu.squeeze(0)
            var_gu = var_gu.squeeze(0)
        return (mean_gu, var_gu * A) if return_cov else mean_gu

    def _cbf_func(self, Xtest, grad_htest, return_cov=False):
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

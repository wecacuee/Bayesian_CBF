from sklearn.gaussian_process.kernels import Kernel, Exponentiation

class AffineScaleKernel(Exponentiation):
    """Scale a kernel by given affine vector

    The resulting kernel splits the input arguments
    k_scaled([x, u], [y, v]) = uᵀv * k(x,y)

    Parameters
    ----------
    kernel: Kernel object
        The base kernel

    xdims   : Integer < |X|
        The index where to split the arguments into two parts
    """
    def __init__(self, kernel, xdims):
        self.kernel = kernel
        self.xdims = xdims

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, xdims=self.xdims)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(('kernel__' + k, val) for k, val in deep_items)
        return params

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        xdims = self.xdims
        X_, U = X[:, :xdims], X[:, xdims:]
        if Y is not None:
            Y_, V = Y[:, :xdims], Y[:, xdims:]
        else:
            Y_, V = None, U

        UdotU = (U * V).sum(axis=-1)
        if eval_gradient:
            K, K_gradient = self.kernel(X_, Y_, eval_gradient=True)
            return UdotU * K, UdotU.reshape(-1, 1) * K_gradient
        else:
            K = self.kernel(X_, Y_, eval_gradient=False)
            return UdotU * K

    def __repr__(self):
        return "AffineScaleKernel(kernel={0}, xdims={1})".format(self.kernel, self.xdims)


class GaussianProcessRegressorAffineLikelihood(GaussianProcessRegressor):
    def __init__(self, *args, **kwargs):
        GaussianProcessRegressor.__init__(self, *args, **kwargs)
        self.U_train_ = None

    def fit(self, X, U, Y):
        self.U_train_ = np.copy(U)
        GaussianProcessRegressor.fit(X, Y)

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        pass

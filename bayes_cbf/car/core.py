import math

import torch

from bayes_cbf.car.HyundaiGenesis import HyundaiGenesisDynamicsModel
from bayes_cbf.misc import t_hstack
from bayes_cbf.sampling import sample_generator_trajectory
from bayes_cbf.plotting import plot_learned_2D_func, plot_results
from bayes_cbf.control_affine_model import ControlAffineRegressor


class ControlRandom:
    needs_ground_truth = False
    def control(self, xi, i=None):
        a, s = torch.rand(2)
        return torch.tensor([a, math.cos(s), math.sin(s)])


def learn_dynamics(
        tau=0.01,
        max_train=50,
        numSteps=2000,
        X0=[1.9,2.5,0, 0,0,0, 0,0,0],
        car_dynamics_class=HyundaiGenesisDynamicsModel):
    car_env = car_dynamics_class()
    dX, X, U = sample_generator_trajectory(
        car_env, D=numSteps, x0=X0,
        dt=tau,
        controller=ControlRandom().control)

    UH = t_hstack((torch.ones((U.shape[0], 1), dtype=U.dtype), U))

    # Do not need the full dataset . Take a small subset
    N = min(numSteps-1, max_train)
    shuffled_range = torch.randint(numSteps - 1, size=(N,))
    XdotTrain = dX[shuffled_range, :]
    Xtrain = X[shuffled_range, :]
    Utrain = U[shuffled_range, :]
    dgp = ControlAffineRegressor(Xtrain.shape[-1], Utrain.shape[-1])
    dgp.fit(Xtrain, Utrain, XdotTrain, training_iter=50)
    dgp.save('/tmp/car-saved.torch')

    # Plot the pendulum trajectory
    Xtrain_numpy = Xtrain.detach().cpu().numpy()
    plot_results(torch.arange(U.shape[0]), omega_vec=X[:, 0],
                 theta_vec=X[:, 1], u_vec=U[:, 0])
    axs = plot_learned_2D_func(Xtrain_numpy, dgp.f_func,
                               pend_env.f_func,
                               axtitle="f(x)[{i}]")
    plt_savefig_with_data(axs.flatten()[0].figure,
                          'plots/f_orig_learned_vs_f_true.pdf')
    axs = plot_learned_2D_func(Xtrain_numpy, dgp.f_func_mean,
                               pend_env.f_func,
                               axtitle="f(x)[{i}]")
    plt_savefig_with_data(axs.flatten()[0].figure,
                          'plots/f_custom_learned_vs_f_true.pdf')
    axs = plot_learned_2D_func(Xtrain_numpy,
                               dgp.g_func,
                               pend_env.g_func,
                               axtitle="g(x)[{i}]")
    plt_savefig_with_data(axs.flatten()[0].figure,
                          'plots/g_learned_vs_g_true.pdf')

    # within train set
    dX_98, _ = dgp.predict_flatten(X[98:99,:], U[98:99, :])
    #dX_98 = FX_98[0, ...].T @ UH[98, :]
    #dXcov_98 = UH[98, :] @ FXcov_98 @ UH[98, :]
    if not torch.allclose(dX[98], dX_98, rtol=0.4, atol=0.1):
        print("Test failed: Train sample: expected:{}, got:{}, cov".format(dX[98], dX_98))

    # out of train set
    dX_Np1, _ = dgp.predict_flatten(X[N+1:N+2,:], U[N+1:N+2,:])
    #dX_Np1 = FXNp1[0, ...].T @ UH[N+1, :]
    if not torch.allclose(dX[N+1], dX_Np1, rtol=0.4, atol=0.1):
        print("Test failed: Test sample: expected:{}, got:{}, cov".format( dX[N+1], dX_Np1))


if __name__ == '__main__':
    learn_dynamics()

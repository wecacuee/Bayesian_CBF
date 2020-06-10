import math
from functools import partial

import torch

from bayes_cbf.car.HyundaiGenesis import HyundaiGenesisDynamicsModel, StateAsArray, rotmat_to_z
from bayes_cbf.misc import t_hstack, store_args
from bayes_cbf.sampling import sample_generator_trajectory, Visualizer
from bayes_cbf.plotting import plot_learned_2D_func, plot_results
from bayes_cbf.control_affine_model import ControlAffineRegressor
from bayes_cbf.controllers import ControlCBFLearned, NamedAffineFunc
from bayes_cbf.car.vis import CarFourObstacles


class CarVisualizer(Visualizer):
    def __init__(self):
        super().__init__()
        self.carworld = CarFourObstacles()
        self.encoder = StateAsArray()

    def setStateCtrl(self, x, u, t=0):
        state, inp = self.encoder.deserialize(x.unsqueeze(0))
        x_ = state.pose.position[:, 0]
        y_ = state.pose.position[:, 1]
        theta_ = rotmat_to_z(state.pose.orientation)
        self.carworld.setCarPose(x_, y_, theta_)
        self.carworld.show()


class CircularObstacle(NamedAffineFunc):
    @store_args
    def __init__(self,
                 model,
                 center,
                 radius,
                 cbf_col_K_alpha=[2, 3],
                 name="cbf-circles",
                 dtype=torch.get_default_dtype()):
        self.model = model
        self.center = center
        self.radius = radius
        self.encoder = StateAsArray()

    def to(self, dtype):
        self.dtype = dtype
        self.model.to(dtype=dtype)

    def value(self, X):
        state, inp = self.encoder.deserialize(X)
        pos = state.pose.position[:, :2]
        distsq = ((pos - self.center)**2).sum(dim=-1)
        return distsq - self.radius**2

    def __call__ (self, x, u):
        return self.A(x) @ u - self.b(x)

    def grad_h_col(self, X_in):
        if X_in.ndim == 1:
            X = X_in.unsqueeze(0)

        with variable_required_grad(X):
            grad_h_x = torch.autograd.grad(self.value(X), X)[0]

        if X_in.ndim == 1:
            grad_h_x = grad_h_x.squeeze(0)
        return grad_h_x

    def lie_f_h_col(self, X):
        return self.grad_h_col(X).bmm( self.model.f_func(X) )

    def grad_lie_f_h_col(self, X):
        with variable_required_grad(X):
            return torch.autograd.grad(self.lie_f_h_col(X), X)[0]

    def lie2_f_h_col(self, X):
        return self.grad_lie_f_h_col(x).bmm( self.model.f_func(x) )

    def lie_g_lie_f_h_col(self, X):
        return self.grad_lie_f_h_col(X).bmm( self.model.g_func(X) )

    def lie2_fu_h_col(self, X, U):
        grad_L1h = self.grad_lie_f_h_col(x)
        return grad_L1h.bmm(self.f_func(X) + self.g_func(X).bmm(U))

    def A(self, X):
        return - self.lie_g_lie_f_h_col(X)

    def b(self, X):
        K_α = torch.tensor(self.cbf_col_K_alpha, dtype=self.dtype)
        η_b_x = torch.cat([self.value(X).unsqueeze(0),
                           self.lie_f_h_col(X).unsqueeze(0)])
        return (self.lie2_f_h_col(x) + K_α @ η_b_x)

class FourCircularObstacles(NamedAffineFunc):
    @store_args
    def __init__(self,
                 model,
                 centers,
                 radii,
                 cbf_col_K_alpha=[2, 3],
                 name="cbf-circles",
                 dtype=torch.get_default_dtype()):
        self.model = model
        self.centers = centers
        self.radii = radii
        self.encoder = StateAsArray()

    def to(self, dtype):
        self.dtype = dtype
        self.model.to(dtype=dtype)

    def value(self, X):
        state, inp = self.encoder.deserialize(X)
        pos = state.pose.position[:, :2]
        distsq = ((pos - self.center)**2).sum(dim=-1)
        return distsq - self.radius**2

    def __call__ (self, x, u):
        return self.A(x) @ u - self.b(x)

    def grad_h_col(self, X_in):
        if X_in.ndim == 1:
            X = X_in.unsqueeze(0)

        with variable_required_grad(X):
            grad_h_x = torch.autograd.grad(self.value(X), X)[0]

        if X_in.ndim == 1:
            grad_h_x = grad_h_x.squeeze(0)
        return grad_h_x

    def lie_f_h_col(self, X):
        return self.grad_h_col(X).bmm( self.model.f_func(X) )

    def grad_lie_f_h_col(self, X):
        with variable_required_grad(X):
            return torch.autograd.grad(self.lie_f_h_col(X), X)[0]

    def lie2_f_h_col(self, X):
        return self.grad_lie_f_h_col(x).bmm( self.model.f_func(x) )

    def lie_g_lie_f_h_col(self, X):
        return self.grad_lie_f_h_col(X).bmm( self.model.g_func(X) )

    def lie2_fu_h_col(self, X, U):
        grad_L1h = self.grad_lie_f_h_col(x)
        return grad_L1h.bmm(self.f_func(X) + self.g_func(X).bmm(U))

    def A(self, X):
        return - self.lie_g_lie_f_h_col(X)

    def b(self, X):
        K_α = torch.tensor(self.cbf_col_K_alpha, dtype=self.dtype)
        η_b_x = torch.cat([self.value(X).unsqueeze(0),
                           self.lie_f_h_col(X).unsqueeze(0)])
        return (self.lie2_f_h_col(x) + K_α @ η_b_x)




class ControlRandom:
    needs_ground_truth = False
    def control(self, xi, i=None):
        a, s = torch.rand(2)
        return torch.tensor([a, math.cos(s), math.sin(s)])


class ControlCarCBFLearned(ControlCBFLearned):
    @store_args
    def __init__(self,
                 dtype=torch.get_default_dtype(),
                 true_model=HyundaiGenesisDynamicsModel,
                 use_ground_truth_model=False):
        if self.use_ground_truth_model:
            self.model = self.true_model
        else:
            self.model = ControlAffineRegressor(
                x_dim, u_dim,
                gamma_length_scale_prior=gamma_length_scale_prior)
        self.cbf2 = CarFourObstacles(self.model, dtype=dtype)
        self.ground_truth_cbf2 = CarFourObstacles(self.true_model, dtype=dtype)


class ControlCarCBFGroundTruth(ControlCarCBFLearned):
    """
    Controller that avoids learning but uses the ground truth model
    """
    needs_ground_truth = False
    def __init__(self, *a, **kw):
        assert kw.pop("use_ground_truth_model", False) is False
        super().__init__(*a, use_ground_truth_model=True, **kw)


def learn_dynamics(
        tau=0.01,
        max_train=100,
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
    deprecate_predict_flatten = True
    if not deprecate_predict_flatten:
        plot_results(torch.arange(U.shape[0]), omega_vec=X[:, 0],
                    theta_vec=X[:, 1], u_vec=U[:, 0])
        axs = plot_learned_2D_func(Xtrain_numpy, dgp.f_func,
                                car_env.f_func,
                                axtitle="f(x)[{i}]")
        plt_savefig_with_data(axs.flatten()[0].figure,
                            'plots/car_f_orig_learned_vs_f_true.pdf')
    # axs = plot_learned_2D_func(Xtrain_numpy, dgp.f_func_mean,
    #                            car_env.f_func,
    #                            axtitle="f(x)[{i}]")
    # plt_savefig_with_data(axs.flatten()[0].figure,
    #                       'plots/car_f_custom_learned_vs_f_true.pdf')
    # axs = plot_learned_2D_func(Xtrain_numpy,
    #                            dgp.g_func,
    #                            car_env.g_func,
    #                            axtitle="g(x)[{i}]")
    # plt_savefig_with_data(axs.flatten()[0].figure,
    #                       'plots/car_g_learned_vs_g_true.pdf')

    # within train set
    dX_98 = dgp.fu_func_mean(U[98:99, :], X[98:99, :])
    #dX_98 = FX_98[0, ...].T @ UH[98, :]
    #dXcov_98 = UH[98, :] @ FXcov_98 @ UH[98, :]
    if not torch.allclose(dX[98], dX_98, rtol=0.4, atol=0.1):
        print("Test failed: Train sample: expected:{}, got:{}, cov".format(dX[98], dX_98))

    # out of train set
    dX_Np1 = dgp.fu_func_mean(U[N+1:N+2,:], X[N+1:N+2,:])
    #dX_Np1 = FXNp1[0, ...].T @ UH[N+1, :]
    if not torch.allclose(dX[N+1], dX_Np1, rtol=0.4, atol=0.1):
        print("Test failed: Test sample: expected:{}, got:{}, cov".format( dX[N+1], dX_Np1))


def run_car_control_ground_truth():
    """
    Run save car control with ground_truth model
    """
    controller = ControlCarCBFGroundTruth()
    return sample_generator_trajectory(
        dynamics_model=HyundaiGenesisDynamicsModel(),
        D=1000,
        controller=controller.control,
        visualizer=CarVisualizer())


if __name__ == '__main__':
    #learn_dynamics()
    run_car_control_ground_truth()

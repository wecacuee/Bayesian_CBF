# stable pendulum
import inspect
from collections import namedtuple
from functools import partial, wraps

import numpy as np
from numpy import cos, sin
#plot the result
import matplotlib.pyplot as plt

from bayes_cbf.matrix_variate_multitask_model import DynamicModelGP

def control_trivial(theta, w, m=None, l=None, g=None):
    assert m is not None
    assert l is not None
    assert g is not None
    u = m*g*sin(theta)
    return u


def env_pendulum(theta0,omega0,tau,m,g,l,numSteps, controller=control_trivial):


    # initialize vectors
    time_vec = np.zeros(numSteps)
    theta_vec = np.zeros(numSteps)
    omega_vec = np.zeros(numSteps)
    u_vec = np.zeros(numSteps)
    #damage indicator
    damage_vec = np.zeros(numSteps)

    # set initial conditions

    theta = theta0
    omega = omega0
    time = 0

    # begin time-stepping

    for i in range(numSteps):
        omega_old = omega
        theta_old = theta
        u= controller(theta, omega)
        # update the values
        omega = omega_old - (g/l)*sin(theta_old)*tau+(u/(m*l))*tau
        theta = theta_old + omega*tau
        # record the values
        time_vec[i] = tau*i
        omega_vec[i] = omega
        u_vec[i] = u
        #record and normalize theta to be in -pi to pi range
        theta_vec[i] = (((theta+np.pi) % (2*np.pi)) - np.pi)
        if 0<theta_vec[i]<np.pi/4:
            damage_vec[i]=1
    damge_perc=damage_vec.sum() * 100/numSteps
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)


def rad2deg(rad):
    return rad / np.pi * 180


def plot_results(time_vec, omega_vec, theta_vec, u_vec):
    #plot thetha
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(time_vec, rad2deg(theta_vec),
                  ":", label = "theta (degrees)",color="blue")
    axs[0,0].set_ylabel("theta (degrees)")
    axs[0,1].plot(time_vec, omega_vec,":", label = "omega (rad/s)",color="blue")
    axs[0,1].set_ylabel("omega")
    axs[1,0].plot(time_vec, u_vec,":", label = "u",color="blue")
    axs[1,0].set_ylabel("u")

    axs[1,1].plot(time_vec, cos(theta_vec),":", label="cos(theta)", color="blue")
    axs[1,1].set_ylabel("cos/sin(theta)")
    axs[1,1].plot(time_vec, sin(theta_vec),":", label="sin(theta)", color="red")
    axs[1,1].set_ylabel("sin(theta)")
    axs[1,1].legend()

    fig.suptitle("Pendulum")
    fig.subplots_adjust(wspace=0.31)
    plt.show()


def run_pendulum_experiment(#parameters
        theta0=5*np.pi/6,
        omega0=-0.01,
        tau=0.01,
        m=1,
        g=10,
        l=1,
        numSteps=7000,
        ground_truth_model=True,
        controller=control_trivial):
    if ground_truth_model:
        controller = partial(controller, m=m, g=g, l=l)
    damge_perc,time_vec,theta_vec,omega_vec,u_vec = env_pendulum(
        theta0,omega0,tau,m,g,l,numSteps, controller=controller)
    plot_results(time_vec, omega_vec, theta_vec, u_vec)
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)


def learn_dynamics(
        theta0=5*np.pi/6,
        omega0=-0.01,
        tau=0.01,
        m=1,
        g=10,
        l=1,
        numSteps=5000):
    #from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    #from bayes_cbf.affine_kernel import AffineScaleKernel
    #from sklearn.gaussian_process import GaussianProcessRegressor

    # kernel_x = 1.0 * RBF(length_scale=np.array([100.0, 100.0]),
    #                      length_scale_bounds=(1e-2, 1e3)) \
    #     + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    # kernel_xu = AffineScaleKernel(kernel_x, 2)

    # xₜ₊₁ = F(xₜ)[1 u]
    # where F(xₜ) = [f(xₜ), g(xₜ)]

    damge_perc,time_vec,theta_vec,omega_vec,u_vec = env_pendulum(
        theta0,omega0,tau,m,g,l,numSteps, controller=partial(control_trivial, m=m, g=g, l=l))
    # X.shape = Nx2
    X = np.vstack((theta_vec, omega_vec)).T
    # XU.shape = Nx3
    U = u_vec.reshape(-1, 1)
    UH = np.hstack((np.ones((U.shape[0], 1), dtype=U.dtype), U))
    XU = np.hstack((X, u_vec.reshape(-1, 1)))

    # compute discrete derivative
    # dxₜ₊₁ = xₜ₊₁ - xₜ / dt
    dX = (X[1:, :] - X[:-1, :]) / tau

    # Do not need the full dataset . Take a small subset
    N = 500
    # Shuffle to make it IID
    shuffled_range = np.arange(N)
    np.random.shuffle(shuffled_range)
    XdotTrain = dX[shuffled_range, :]
    Xtrain = X[shuffled_range, :]
    Utrain = U[shuffled_range, :]
    #gp = GaussianProcessRegressor(kernel=kernel_xu,
    #                              alpha=1e6).fit(Z_shuffled, Y_shuffled)
    dgp = DynamicModelGP(Xtrain.shape[-1], Utrain.shape[-1]
             ).fit(Xtrain, Utrain, XdotTrain, lr=0.01)

    # within train set
    FX_99, FXcov_99 = dgp.predict(X[98:99,:], return_cov=True)
    dX_99 = FX_99[0, ...].T @ UH[98, :]
    print("Train sample: expected:{}, got:{}, cov:{}".format(dX_99, dX[99], FXcov_99))
    assert np.allclose(dX[99], dX_99, rtol=0.01, atol=0.01)

    # out of train set
    FXNp1, FXNp1cov = dgp.predict(X[N+1:N+2,:], return_cov=True)
    dX_Np1 = FXNp1[0, ...].T @ UH[N+1, :]
    print("Test sample: expected:{}, got:{}, cov:{}".format( dX[N+1], dX_Np1, FXNp1cov))
    assert np.allclose(dX[N+1], dX_Np1, rtol=0.01, atol=0.01)
    return dgp, dX, XU


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    import cvxopt
    from cvxopt import matrix
    #P = (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


ControlAffine = namedtuple("ControlAffine",
                           ["A", "b"])

def control_cbf_clf(theta, w,
                    theta_c=np.pi/4,
                    c=1,
                    gamma_sr=1,
                    delta_sr=10,
                    gamma_col=1,
                    delta_col=np.pi/8,
                    m=None,
                    l=None,
                    g=None):
    assert m is not None
    assert l is not None
    assert g is not None
    def A_clf(theta, w):
        return l*w

    def b_clf(theta, w):
        return -c*(0.5*m*l**2*w**2+m*g*l*(1-cos(theta)))

    def A_sr(theta, w):
        return l*w

    def b_sr(theta, w):
        return gamma_sr*(delta_sr-w**2)+(2*g*sin(theta)*w)/(l)

    def A_col(theta, w):
        return -(2*w*(cos(delta_col)-cos(theta-theta_c)))/(m*l)

    def b_col(theta, w):
        return (gamma_col*(cos(delta_col)-cos(theta-theta_c))*w**2
                + w**3*sin(theta-theta_c)
                - (2*g*w*sin(theta)*(cos(delta_col)-cos(theta-theta_c)))/l)

    return control_QP_cbf_clf(theta, w,
                              ctrl_aff_clf=ControlAffine(A_clf, b_clf),
                              ctrl_aff_cbfs=[ControlAffine(A_sr, b_sr),
                                             ControlAffine(A_col, b_col)])


def store_args(method):
    argspec = inspect.getfullargspec(method)
    @wraps(method)
    def wrapped_method(self, *args, **kwargs):
        for name, val in zip(argspec.args[::-1], argspec.defaults[::-1]):
            setattr(self, name, val)
        if argspec.kwonlydefaults and args.kwonlyargs:
            for name, val in zip(argspec.kwonlyargs, argspec.kwonlydefaults):
                setattr(self, name, val)
        for name, val in zip(argspec.args, args):
            setattr(self, name, val)
        for name, val in kwargs.items():
            setattr(self, name, val)

        method(self, *args, **kwargs)

    return wrapped_method


class ControlCBFCLFLearned:
    @store_args
    def __init__(self,
                 x_dim=2,
                 u_dim=1,
                 theta_c=np.pi/4,
                 c=1,
                 gamma_sr=1,
                 delta_sr=10,
                 gamma_col=1,
                 delta_col=np.pi/8,
                 train_every_n_steps=10
    ):
        self.Xtrain = []
        self.Utrain = []
        self.dgp = DynamicModelGP(x_dim, u_dim)

    def f_g(self, theta, w):
        X = np.array([[theta, w]])
        FXTmean, FXTcov = self.dgp.predict(X, return_cov=False)
        fx = FXTmean[0, 0, :]
        gx = FXTmean[0, 1:, :].T
        return fx, gx

    def V_clf(self, theta, w):
        return w**2 / 2 + (1-np.cos(theta))

    def grad_V_clf(self, theta, w):
        return np.array([np.sin(theta), w])

    def A_clf(self, theta, w):
        return self.grad_V_clf(theta, w) @ self.f_g(theta, w)[1]

    def b_clf(self, theta, w):
        c = self.c
        return - self.grad_V_clf(theta, w) @ self.f_g(theta, w)[0] - c * self.V_clf(theta, w)


    def h_col(self, theta, w):
        delta_col = self.delta_col
        theta_c = self.theta_c
        return (np.cos(delta_col) - np.cos(theta - theta_c))*w**2

    def grad_h_col(self, theta, w):
        delta_col = self.delta_col
        theta_c = self.theta_c
        return np.array([w**2*np.sin(theta - theta_c),
                        2*w*(np.cos(delta_col) - np.cos(theta - theta_c))])

    def A_col(self, theta, w):
        return self.grad_h_col(theta, w) @ self.f_g(theta, w)[1]

    def b_col(self, theta, w):
        gamma_col = self.gamma_col
        return - self.grad_h_col(theta, w) @ self.f_g(theta, w)[0] - gamma_col * self.h_col(theta, w)

    def train(self):
        if not len(self.Xtrain):
            return
        Xtrain = np.array(self.Xtrain)
        Utrain = np.array(self.Utrain)
        XdotTrain = Xtrain[1:, :] - Xtrain[:-1, :]
        self.dgp.fit(Xtrain[:-1, :], Utrain[:-1, :], XdotTrain)

    def controller(self, theta, w):
        if len(self.Xtrain) % self.train_every_n_steps == 0:
            # train every n steps
            self.train()
        self.Xtrain.append([theta, w])
        u = control_QP_cbf_clf(theta, w,
                                  ctrl_aff_clf=ControlAffine(self.A_clf, self.b_clf),
                                  ctrl_aff_cbfs=[ControlAffine(self.A_col, self.b_col)])
        self.Utrain.append([u])
        return u


def control_QP_cbf_clf(theta, w,
                    ctrl_aff_clf,
                    ctrl_aff_cbfs,
                    clf_only=False):
    """
    Args:
          A_cbfs: A tuple of CBF functions
          b_cbfs: A tuple of CBF functions
    """
    A_total = np.vstack([A(theta, w) for A, b in [ctrl_aff_clf] + ctrl_aff_cbfs])
    b_total = np.vstack([b(theta, w) for A, b in [ctrl_aff_clf] + ctrl_aff_cbfs])

    # u0 = l*g*sin(theta)
    # uopt = 0.1*g
    # contraints = A_total.dot(uopt) - b_total
    # assert contraints[0] <= 0
    # assert contraints[1] <= 0
    # assert contraints[2] <= 0


    A_total_rho = np.hstack((A_total, np.zeros((A_total.shape[0], 1))))
    A_total_rho[0, -1] = -1
    from cvxopt import matrix
    if clf_only:
        P_rho = np.array([[50., 0],
                          [0, 100.]])
        q_rho = np.array([0., 0.])
        u_rho = cvxopt_solve_qp(P_rho, q_rho,
                            G=A_total_rho[0:1,:],
                            h=b_total[0])
    else:
        P_rho = np.array([[50., 0],
                          [0, 1000.]])
        q_rho = np.array([0., 0.])
        u_rho = cvxopt_solve_qp(P_rho, q_rho,
                                G=A_total_rho,
                                h=b_total)

    u = u_rho[0]
    return u


run_pendulum_control_trival = partial(
    run_pendulum_experiment, controller=control_trivial)

run_pendulum_control_cbf_clf = partial(
    run_pendulum_experiment, controller=control_cbf_clf)


def run_pendulum_control_online_learning():
    return run_pendulum_experiment(
        ground_truth_model=False,
        controller=ControlCBFCLFLearned().controller)

if __name__ == '__main__':
    #run_pendulum_control_trival()
    #run_pendulum_control_cbf_clf()
    #learn_dynamics()
    run_pendulum_control_online_learning()

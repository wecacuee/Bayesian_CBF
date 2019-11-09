# stable pendulum
import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

import sys
import io
import tempfile
import inspect
from collections import namedtuple
from functools import partial, wraps
import pickle
import hashlib
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as mplibrc
mplibrc('text', usetex=True)

from bayes_cbf.control_affine_model import ControlAffineRegressor, LOG as CALOG
CALOG.setLevel(logging.WARNING)

from bayes_cbf.plotting import plot_results, plot_learned_2D_func
from bayes_cbf.sampling import sample_generator_trajectory, controller_sine


def control_trivial(xi, m=1, mass=None, length=None, gravity=None):
    assert mass is not None
    assert length is not None
    assert gravity is not None
    theta, w = xi
    u = mass * gravity * np.sin(theta)
    return np.array([u])


def control_random(xi, m=1, mass=None, length=None, gravity=None):
    assert mass is not None
    assert length is not None
    assert gravity is not None
    return control_trivial(
        xi, mass=mass, length=length, gravity=gravity
    ) * np.abs(np.random.rand()) + np.random.rand()


class PendulumDynamicsModel:
    def __init__(self, m, n, mass=1, gravity=10, length=1, deterministic=True,
                 model_noise=0):
        self.m = m
        self.n = n
        self.mass = mass
        self.gravity = gravity
        self.length = length
        self.model_noise = model_noise

    @property
    def ctrl_size(self):
        return self.m

    @property
    def state_size(self):
        return self.n

    def f_func(self, X):
        m = self.m
        n = self.n
        mass = self.mass
        gravity = self.gravity
        length = self.length
        X = np.asarray(X)
        theta_old, omega_old = X[..., 0:1], X[..., 1:2]
        noise = np.random.normal(scale=self.model_noise) if self.model_noise else 0
        return noise + np.concatenate(
            [omega_old,
             - (gravity/length)*np.sin(theta_old)], axis=-1)

    def g_func(self, x):
        m = self.m
        n = self.n
        mass = self.mass
        gravity = self.gravity
        length = self.length
        size = x.shape[0] if x.ndim == 2 else 1
        noise = np.random.normal(scale=self.model_noise) if self.model_noise else 0
        return noise + np.repeat(
            np.array([[[0], [1/(mass*length)]]]), size, axis=0)


def sampling_pendulum(dynamics_model, numSteps,
                      x0=None,
                      dt=0.01,
                      controller=control_trivial,
                      plot_every_n_steps=100,
                      axs=None):
    m, g, l = (dynamics_model.mass, dynamics_model.gravity,
               dynamics_model.length)
    tau = dt
    f_func, g_func = dynamics_model.f_func, dynamics_model.g_func
    theta0, omega0 = x0

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
        time_vec[i] = tau*i
        theta_vec[i] = theta
        omega_vec[i] = omega
        u= controller((theta, omega))
        u_vec[i] = u

        if 0<theta_vec[i]<np.pi/4:
            damage_vec[i]=1

        omega_old = omega
        theta_old = theta
        # update the values
        omega_direct = omega_old - (g/l)*np.sin(theta_old)*tau+(u[0]/(m*l))*tau
        theta_direct = theta_old + omega_old * tau
        # Update as model
        Xold = np.array([[theta_old, omega_old]])
        Xdot_tau = f_func(Xold) * tau  + g_func(Xold) @ u * tau
        theta_prop, omega_prop = ( Xold + Xdot_tau ).flatten()
        #assert np.allclose(omega_direct, omega_prop, atol=1e-6, rtol=1e-4)
        #assert np.allclose(theta_direct, theta_prop, atol=1e-6, rtol=1e-4)
        LOG.debug("Diff: {}".format(np.abs(theta_direct - theta_prop)))
        theta, omega = theta_prop, omega_prop

        # theta, omega = theta_direct, omega_direct
        # record the values
        #record and normalize theta to be in -pi to pi range
        theta = (((theta+np.pi) % (2*np.pi)) - np.pi)
        if i % plot_every_n_steps == 0:
            axs = plot_results(np.arange(i+1), omega_vec[:i+1], theta_vec[:i+1],
                               u_vec[:i+1], axs=axs)
            plt.pause(0.001)

    assert np.all((theta_vec <= np.pi) & (-np.pi <= theta_vec))
    damge_perc=damage_vec.sum() * 100/numSteps
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)


def sampling_pendulum_data(dynamics_model, D=100, dt=0.01, **kwargs):
    tau = dt
    (damge_perc,time_vec,theta_vec,omega_vec,u_vec) = sampling_pendulum(
        dynamics_model, numSteps=D+1, **kwargs)

    # X.shape = Nx2
    X = np.vstack((theta_vec, omega_vec)).T
    # XU.shape = Nx3
    U = u_vec.reshape(-1, 1)
    XU = np.hstack((X, u_vec.reshape(-1, 1)))

    # compute discrete derivative
    # dxₜ₊₁ = xₜ₊₁ - xₜ / dt
    dX = (X[1:, :] - X[:-1, :]) / tau

    assert np.all((X[:, 0] <= np.pi) & (-np.pi <= X[:, 0]))
    return dX, X, U


def rad2deg(rad):
    return rad / np.pi * 180


def run_pendulum_experiment(#parameters
        theta0=5*np.pi/6,
        omega0=-0.01,
        tau=0.01,
        mass=1,
        gravity=10,
        length=1,
        numSteps=10000,
        ground_truth_model=True,
        controller=control_trivial,
        plotfile='plots/run_pendulum_experiment.pdf'):
    if ground_truth_model:
        controller = partial(controller, mass=mass, gravity=gravity, length=length)
    damge_perc,time_vec,theta_vec,omega_vec,u_vec = sampling_pendulum(
        PendulumDynamicsModel(m=1, n=2, mass=mass, gravity=gravity,
                              length=length),
        numSteps, x0=(theta0,omega0), controller=controller)
    plot_results(time_vec, omega_vec, theta_vec, u_vec)
    plt.savefig(plotfile)
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)


def learn_dynamics(
        theta0=5*np.pi/6,
        omega0=-0.01,
        tau=0.01,
        mass=1,
        gravity=10,
        length=1,
        max_train=200,
        numSteps=1000):
    #from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    #from bayes_cbf.affine_kernel import AffineScaleKernel
    #from sklearn.gaussian_process import GaussianProcessRegressor

    # kernel_x = 1.0 * RBF(length_scale=np.array([100.0, 100.0]),
    #                      length_scale_bounds=(1e-2, 1e3)) \
    #     + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    # kernel_xu = AffineScaleKernel(kernel_x, 2)

    # xₜ₊₁ = F(xₜ)[1 u]
    # where F(xₜ) = [f(xₜ), g(xₜ)]

    pend_env = PendulumDynamicsModel(m=1, n=2, mass=mass, gravity=gravity,
                                     length=length)
    dX, X, U = sampling_pendulum_data(
        pend_env, D=numSteps, x0=(theta0,omega0),
        dt=tau,
        controller=partial(control_random, mass=mass, gravity=gravity,
                           length=length))

    UH = np.hstack((np.ones((U.shape[0], 1), dtype=U.dtype), U))

    # Do not need the full dataset . Take a small subset
    N = min(numSteps-1, max_train)
    shuffled_range = np.random.randint(numSteps - 1, size=N)
    XdotTrain = dX[shuffled_range, :]
    Xtrain = X[shuffled_range, :]
    Utrain = U[shuffled_range, :]
    #gp = GaussianProcessRegressor(kernel=kernel_xu,
    #                              alpha=1e6).fit(Z_shuffled, Y_shuffled)
    dgp = ControlAffineRegressor(Xtrain.shape[-1], Utrain.shape[-1])
    dgp.fit(Xtrain, Utrain, XdotTrain, training_iter=50, lr=0.01)
    dgp.save()

    # Plot the pendulum trajectory
    plot_results(np.arange(U.shape[0]), omega_vec=X[:, 0],
                 theta_vec=X[:, 1], u_vec=U[:, 0])
    plot_learned_2D_func(Xtrain, dgp.f_func, pend_env.f_func,
                         axtitle="f(x)[{i}]")
    plt.savefig('plots/f_learned_vs_f_true.pdf')
    plot_learned_2D_func(Xtrain, dgp.g_func, pend_env.g_func,
                         axtitle="g(x)[{i}]")
    plt.savefig('plots/g_learned_vs_g_true.pdf')

    # within train set
    FX_98, FXcov_98 = dgp.predict(X[98:99,:], return_cov=True)
    dX_98 = FX_98[0, ...].T @ UH[98, :]
    #dXcov_98 = UH[98, :] @ FXcov_98 @ UH[98, :]
    if not np.allclose(dX[98], dX_98, rtol=0.05, atol=0.05):
        print("Train sample: expected:{}, got:{}, cov:{}".format(dX[98], dX_98, FXcov_98))

    # out of train set
    FXNp1, FXNp1cov = dgp.predict(X[N+1:N+2,:], return_cov=True)
    dX_Np1 = FXNp1[0, ...].T @ UH[N+1, :]
    if not np.allclose(dX[N+1], dX_Np1, rtol=0.05, atol=0.05):
        print("Test sample: expected:{}, got:{}, cov:{}".format( dX[N+1], dX_Np1, FXNp1cov))

    return dgp, dX, U


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, solver=None,
                    initvals=None, **kwargs):
    import cvxopt
    from cvxopt import matrix
    #P = (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
    else:
        args.extend([None, None])
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    else:
        args.extend([None, None])
    args.extend([initvals, solver])
    solvers = cvxopt.solvers
    old_options = solvers.options.copy()
    solvers.options.update(kwargs)
    try:
        sol = cvxopt.solvers.qp(*args)
    except ValueError:
        return None
    finally:
        solvers.options.update(old_options)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


NamedAffineFunc = namedtuple("NamedAffineFunc",
                             ["A", "b", "name"])


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


class PendulumCBFCLFDirect:
    @store_args
    def __init__(self, mass=None, length=None, gravity=None,
                 cbf_col_theta=np.pi/4,
                 clf_c=1,
                 cbf_sr_gamma=1,
                 cbf_sr_delta=10,
                 cbf_col_gamma=1,
                 cbf_col_delta=np.pi/8,
                 axes=None,
                 constraint_hists=[]):
        pass

    def A_clf(self, x):
        (theta, w) = x
        m, l, g = self.mass, self.length, self.gravity
        return np.array([l*w])

    def b_clf(self, x):
        (theta, w) = x
        m, l, g = self.mass, self.length, self.gravity
        c = self.clf_c
        return np.array([-c*(0.5*m*l**2*w**2+m*g*l*(1-np.cos(theta)))])

    def A_sr(self, x):
        (theta, w) = x
        m, l, g = self.mass, self.length, self.gravity
        return np.array([l*w])

    def b_sr(self, x):
        (theta, w) = x
        m, l, g = self.mass, self.length, self.gravity
        gamma_sr = self.cbf_sr_gamma
        delta_sr = self.cbf_sr_delta
        return np.array([gamma_sr*(delta_sr-w**2)+(2*g*np.sin(theta)*w)/(l)])

    def A_col(self, x):
        (theta, w) = x
        m, l, g = self.mass, self.length, self.gravity
        delta_col = self.cbf_col_delta
        theta_c = self.cbf_col_theta
        return np.array([-(2*w*(np.cos(delta_col)-np.cos(theta-theta_c)))/(m*l)])

    def b_col(self, x):
        (theta, w) = x
        m, l, g = self.mass, self.length, self.gravity
        gamma_col = self.cbf_col_gamma
        delta_col = self.cbf_col_delta
        theta_c = self.cbf_col_theta
        b = (gamma_col*(np.cos(delta_col)-np.cos(theta-theta_c))*w**2
             + w**3*np.sin(theta-theta_c)
             - (2*g*w*np.sin(theta)*(np.cos(delta_col)-np.cos(theta-theta_c)))/l)
        print("theta: {}".format(x[0]))
        print("cos(theta -theta_c):{} <= delta_c:{}".format(
            np.cos(x[0] - theta_c), np.cos(delta_col)))
        print("ω:{}".format(w))
        print("b:{}".format(b))
        return np.array([b])

    def plot_constraints(self, affine_funcs, x, u):
        axs = self.axes
        if axs is None:
            nplots = len(affine_funcs)
            shape = ((math.ceil(nplots / 2), 2) if nplots >= 2 else (nplots,))
            fig, axs = plt.subplots(*shape)
            fig.subplots_adjust(wspace=0.35, hspace=0.5)
            self.axes = axs.flatten() if hasattr(axs, "flatten") else np.array([axs])

        if len(self.constraint_hists) < len(affine_funcs):
            self.constraint_hists = self.constraint_hists + [
                list() for _ in range(
                len(affine_funcs) - len(self.constraint_hists))]

        for i, af in enumerate(affine_funcs):
            A_func, b_func = af.A, af.b
            self.constraint_hists[i].append(A_func(x) @ u - b_func(x))

        if np.random.rand() < 1e-2:
            for i, (ch, af) in enumerate(zip(self.constraint_hists, affine_funcs)):
                axs[i].clear()
                axs[i].plot(ch)
                axs[i].set_ylabel(af.name)
                axs[i].set_xlabel("time")
                plt.pause(0.0001)


    def control(self, xi, mass=None, gravity=None, length=None):
        assert mass is not None
        assert length is not None
        assert gravity is not None
        self.mass, self.gravity, self.length = mass, gravity, length

        aff_contraints = [NamedAffineFunc(self.A_clf, self.b_clf, "clf"),
                          NamedAffineFunc(self.A_col, self.b_col, "col")]
        u = control_QP_cbf_clf(
            xi,
            ctrl_aff_constraints=aff_contraints,
            constraint_margin_weights=[35.])
        self.plot_constraints(aff_contraints, xi, u)
        return u


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
                 train_every_n_steps=50
    ):
        self.Xtrain = []
        self.Utrain = []
        self.dgp = ControlAffineRegressor(x_dim, u_dim)

    def f_g(self, x):
        X = np.array([x])
        FXTmean = self.dgp.predict(X, return_cov=False)
        fx = FXTmean[0, 0, :]
        gx = FXTmean[0, 1:, :].T
        return fx, gx

    def f(self, x):
        return self.f_g(x)[0]

    def g(self, x):
        return self.f_g(x)[1]

    def V_clf(self, x):
        (theta, w) = x
        return w**2 / 2 + (1-np.cos(theta))

    def grad_V_clf(self, x):
        (theta, w) = x
        return np.array([np.sin(theta), w])

    def A_clf(self, x):
        return self.grad_V_clf(x) @ self.g(x)

    def b_clf(self, x):
        c = self.c
        return - self.grad_V_clf(x) @ self.f(x) - c * self.V_clf(x)

    def h_col(self, x):
        (theta, w) = x
        delta_col = self.delta_col
        theta_c = self.theta_c
        return (np.cos(delta_col) - np.cos(theta - theta_c))*w**2

    def grad_h_col(self, x):
        (theta, w) = x
        delta_col = self.delta_col
        theta_c = self.theta_c
        return np.array([w**2*np.sin(theta - theta_c),
                        2*w*(np.cos(delta_col) - np.cos(theta - theta_c))])

    def A_col(self, x):
        (theta, w) = x
        return self.grad_h_col(x) @ self.g(x)

    def b_col(self, x):
        gamma_col = self.gamma_col
        return - self.grad_h_col(x) @ self.f(x) - gamma_col * self.h_col(x)

    def train(self):
        if not len(self.Xtrain):
            return
        assert len(self.Xtrain) == len(self.Utrain), "Call train when Xtrain and Utrain are balanced"
        Xtrain = np.array(self.Xtrain)
        Utrain = np.array(self.Utrain)
        XdotTrain = Xtrain[1:, :] - Xtrain[:-1, :]
        plot_results(np.arange(Utrain.shape[0]), omega_vec=Xtrain[:, 0],
                     theta_vec=Xtrain[:, 1], u_vec=Utrain[:, 0])
        plt.savefig('plots/pendulum_data_{}.pdf'.format(Xtrain.shape[0]))
        assert np.all((Xtrain[:, 0] <= np.pi) & (-np.pi <= Xtrain[:, 0]))
        LOG.info("Training model with datasize {}".format(XdotTrain.shape[0]))
        try:
            self.dgp.fit(Xtrain[:-1, :], Utrain[:-1, :], XdotTrain)
        except AssertionError:
            train_data = (Xtrain[:-1, :], Utrain[:-1, :], XdotTrain)
            filename = hashlib.sha224(pickle.dumps(train_data)).hexdigest()
            filepath = 'tests/data/{}.pickle'.format(filename)
            pickle.dump(train_data, open(filepath, 'wb'))
            raise

    def controller(self, xi):
        if len(self.Xtrain) % self.train_every_n_steps == 0:
            # train every n steps
            LOG.info("Training GP with dataset size {}".format(len(self.Xtrain)))
            self.train()

        assert np.all((xi[0] <= np.pi) & (-np.pi <= xi[0]))
        self.Xtrain.append(xi)
        u = control_QP_cbf_clf(xi,
                               ctrl_aff_constraints=[
                                   NamedAffineFunc(self.A_col, self.b_col, "col"),
                                   NamedAffineFunc(self.A_clf, self.b_clf, "clf")],
                               constraint_margin_weights=[1000., 1.])
        self.Utrain.append(u)
        return u


def control_QP_cbf_clf(x,
                       ctrl_aff_constraints,
                       constraint_margin_weights=[]):
    """
    Args:
          A_cbfs: A tuple of CBF functions
          b_cbfs: A tuple of CBF functions
          constraint_margin_weights: Add a margin constant to the constraint
                                     that is maximized.

    """
    clf_idx = 0
    A_total = np.vstack([af.A(x) for af in ctrl_aff_constraints])
    b_total = np.vstack([af.b(x) for af in ctrl_aff_constraints]).flatten()
    D_u = A_total.shape[1]
    N_const = A_total.shape[0]

    # u0 = l*g*sin(theta)
    # uopt = 0.1*g
    # contraints = A_total.dot(uopt) - b_total
    # assert contraints[0] <= 0
    # assert contraints[1] <= 0
    # assert contraints[2] <= 0


    # [A, I][ u ]
    #       [ ρ ] ≤ b for all constraints
    #
    # minimize
    #         [ u, ρ1, ρ2 ] [ 1,     0] [  u ]
    #                       [ 0,   100] [ ρ2 ]
    #         [A_cbf, 1] [ u, -ρ ] ≤ b_cbf
    #         [A_clf, 1] [ u, -ρ ] ≤ b_clf
    N_slack = len(constraint_margin_weights)
    A_total_rho = np.hstack(
        (A_total,
         np.vstack((-np.eye(N_slack),
                    np.zeros((N_const - N_slack, N_slack))))
        ))
    A = A_total
    P_rho = np.eye(D_u + N_slack)
    P_rho[D_u:, D_u:] = np.diag(constraint_margin_weights)
    q_rho = np.zeros(P_rho.shape[0])
    u_rho_init = np.linalg.lstsq(A_total_rho, b_total - 1e-1, rcond=-1)[0]
    u_rho = cvxopt_solve_qp(P_rho, q_rho,
                            G=A_total_rho,
                            h=b_total,
                            initvals=dict(x=u_rho_init),
                            show_progress=True,
                            maxiters=100)
    if u_rho is None:
        if np.all(A_total_rho @ u_rho_init - b_total <= 0):
            return u_rho_init[:D_u]
        else:
            raise RuntimeError("""QP is infeasible
            minimize
            u_rhoᵀ {P_rho} u_rho
            s.t.
            {A_total_rho} u_rho ≤ {b_total}""".format(
                P_rho=P_rho,
                A_total_rho=A_total_rho, b_total=b_total))
    # Constraints should be satisfied
    constraint = A_total_rho @ u_rho - b_total
    assert np.all((constraint <= 1e-2) | (constraint / np.abs(b_total) <= 1e-2))
    return u_rho[:D_u]


run_pendulum_control_trival = partial(
    run_pendulum_experiment, controller=control_trivial,
    plotfile='plots/run_pendulum_control_trival.pdf')
"""
Run pendulum with a trivial controller.
"""


run_pendulum_control_cbf_clf = partial(
    run_pendulum_experiment, controller=PendulumCBFCLFDirect().control,
    plotfile='plots/run_pendulum_control_cbf_clf.pdf',
    tau=0.01,
    numSteps=10000)
"""
Run pendulum with a safe CLF-CBF controller.
"""


def run_pendulum_control_online_learning(numSteps=1000):
    """
    Run save pendulum control while learning the parameters online
    """
    return run_pendulum_experiment(
        ground_truth_model=False,
        plotfile='plots/run_pendulum_control_online_learning.pdf',
        controller=ControlCBFCLFLearned().controller,
        numSteps=numSteps)


if __name__ == '__main__':
    #run_pendulum_control_trival()
    run_pendulum_control_cbf_clf()
    #learn_dynamics()
    #run_pendulum_control_online_learning()

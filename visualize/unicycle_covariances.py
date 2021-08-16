import math
import glob
import os.path as osp
import os

from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse, FancyArrowPatch, Arrow
from matplotlib.collections import PatchCollection
from matplotlib import transforms

import bayes_cbf.unicycle_move_to_pose as bayes_cbf_unicycle

from bayes_cbf.misc import TBLogger, to_numpy, load_tensorboard_scalars
from bayes_cbf.unicycle_move_to_pose import (AckermannDrive, ControllerCLF,
                                             NoPlanner, CartesianDynamics,
                                             CLFCartesian, VisualizerZ,
                                             LearnedShiftInvariantDynamics)

from bayes_cbf.control_affine_model import (ControlAffineRegressor, LOG as CALOG,
                                            ControlAffineRegressorExact,
                                            ControlAffineRegressorVector,
                                            ControlAffineRegMatrixDiag,
                                            ControlAffineRegVectorDiag,
                                            is_psd)
from bayes_cbf.sampling import sample_generator_trajectory

def unicycle_plot_covariances_exp(
        max_train=200, # testing GPU
        state_start = [-3, -1, -math.pi/4],
        state_goal = [0, 0, math.pi/4],
        numSteps = 512,
        dt = 0.01,
        true_dynamics_gen=partial(AckermannDrive,
                                  L = 1.0),
        mean_dynamics_gen=partial(AckermannDrive,
                                  L = 12.0),
        logger_class=partial(TBLogger,
                             exp_tags=['unicycle_plot_covariances'],
                             runs_dir='data/runs'),
        exps=dict(
            matrix=dict(
                regressor_class=partial(
                    LearnedShiftInvariantDynamics,
                    learned_dynamics_class=ControlAffineRegressorExact)),
            vector=dict(
                regressor_class=partial(
                    LearnedShiftInvariantDynamics,
                    learned_dynamics_class=ControlAffineRegressorVector))
            ),
):
    logger = logger_class()
    bayes_cbf_unicycle.TBLOG = logger.summary_writer
    true_dynamics_model=true_dynamics_gen()

    # Generate training data
    Xdot, X, U = sample_generator_trajectory(
        dynamics_model=true_dynamics_model,
        D=numSteps,
        controller=ControllerCLF(
            NoPlanner(torch.tensor(state_goal)),
            coordinate_converter = lambda x, x_g: x,
            dynamics = CartesianDynamics(),
            clf = CLFCartesian()
        ).control,
        visualizer=VisualizerZ(),
        x0=state_start,
        dt=dt)

    # Log training data
    for t,  (dx, x, u) in enumerate(zip(Xdot, X, U)):
        logger.add_tensors("traj", dict(dx=dx, x=x, u=u), t)

    shuffled_order = np.arange(X.shape[0]-1)

    dgp = dict()
    # Test train split
    np.random.shuffle(shuffled_order)
    shuffled_order_t = torch.from_numpy(shuffled_order)
    train_indices = shuffled_order_t[:max_train]
    Xtrain = X[train_indices, :]
    Utrain = U[train_indices, :]
    XdotTrain = Xdot[train_indices, :]

    slice_range = []
    for i, num in enumerate([1, 1, 20]):
        x_range = slice(*list(map(float,
                                (Xtrain[:, i].min(),
                                Xtrain[:, i].max(),
                                    (Xtrain[:, i].max() - Xtrain[:, i].min()) / num))))
        slice_range.append(x_range)
    xtest_grid = np.mgrid[tuple(slice_range)]
    Xtest = torch.from_numpy(
        xtest_grid.reshape(-1, Xtrain.shape[-1])).to(
            dtype=Xtrain.dtype,
            device=Xtrain.device)
    # b, n, 1+m
    FX_true = true_dynamics_model.F_func(Xtest).transpose(-2, -1)

    for name, kw in exps.items():
        model = kw['regressor_class'](dt = dt,
                                      mean_dynamics=mean_dynamics_gen(),
                                      max_train=max_train)
        model.fit(Xtrain, Utrain, XdotTrain, training_iter=50)
        # b(1+m)n
        FX_learned, var_FX = model.custom_predict_fullmat(Xtest.reshape(-1, Xtest.shape[-1]))
        b = Xtest.shape[0]
        n = true_dynamics_model.state_size
        m = true_dynamics_model.ctrl_size
        var_FX_t = var_FX.reshape(b, (1+m)*n, b, (1+m)*n)
        var_FX_diag_t = torch.empty((b, (1+m)*n, (1+m)*n),
                                    dtype=var_FX_t.dtype,
                                    device=var_FX_t.device)
        for i in range(b):
            var_FX_diag_t[i, :, :] = var_FX_t[i, :, i, :]

        # log FX_learned and var_FX_diag_t
        logger.add_tensors(name, dict(var_FX_diag_t=to_numpy(var_FX_diag_t)),
                           max_train)

    # Find the latest edited event file from log dir
    events_file = max(
        glob.glob(osp.join(logger.experiment_logs_dir, "*.tfevents*")),
        key=lambda f: os.stat(f).st_mtime)
    return events_file

def plot_pendulum_covariances(
        theta0=5*math.pi/6,
        omega0=-0.01,
        tau=0.01,
        max_train=200,
        ntest=1,
        numSteps=1000,
        mass=1,
        gravity=10,
        length=1,
        logger_class=partial(TBLogger,
                             exp_tags=['cartesian_dynamics'],
                             runs_dir='data/runs'),
        pendulum_dynamics_class=CartesianDynamics,
):
    logger = logger_class()
    pend_env = pendulum_dynamics_class(m=1, n=2, mass=mass, gravity=gravity,
                                       length=length)
    dX, X, U = sampling_pendulum_data(
        pend_env, D=numSteps, x0=torch.tensor([theta0,omega0]),
        dt=tau,
        controller=ControlRandom(mass=mass, gravity=gravity, length=length).control,
        plot_every_n_steps=numSteps)

    shuffled_order = np.arange(X.shape[0]-1)

    learned_models = {}
    shuffled_order = np.arange(X.shape[0]-1)

    # Test train split
    np.random.shuffle(shuffled_order)
    shuffled_order_t = torch.from_numpy(shuffled_order)

    train_indices = shuffled_order_t[:max_train]
    Xtrain = X[train_indices, :]
    Utrain = U[train_indices, :]
    XdotTrain = dX[train_indices, :]

    Xtest = X[shuffled_order_t[-ntest:], :]


    lm_matrix = ControlAffineRegressorExact(Xtrain.shape[-1], Utrain.shape[-1])
    lm_matrix.fit(Xtrain, Utrain, XdotTrain, training_iter=50)
    meanFX, A, BkXX = lm_matrix._custom_predict_matrix(Xtest, Xtest,
                                                       compute_cov=True)
    fig, ax = plt.subplots(1, 2, squeeze=False)
    ax[0, 0].set_title('Var[f(x)]')
    plot_covariance(ax[0, 0], to_numpy(BkXX[0, 0, 0, 0] * A))
    ax[0, 1].set_title('Var[g(x)]')
    plot_covariance(ax[0, 1], to_numpy(BkXX[0, 0, 1, 1] * A))
    # ax[0, 2].set_title('cov[f(x), g(x)]')
    # plot_covariance(ax[0, 2], to_numpy(BkXX[0, 0, 0, 1] * A))

    lm_vector = ControlAffineRegressorVector(Xtrain.shape[-1], Utrain.shape[-1])
    lm_vector.fit(Xtrain, Utrain, XdotTrain, training_iter=50)
    meanFX, KkXX = lm_vector._custom_predict_matrix(Xtest, Xtest,
                                                    compute_cov=True)
    plt.savefig('MVGP_covariances.pdf')

    fig, ax = plt.subplots(1, 2, squeeze=False)
    ax[0, 0].set_title('Var[f(x)]')
    plot_covariance(ax[0, 0], to_numpy(KkXX[0, 0, :2, :2]))
    ax[0, 1].set_title('Var[g(x)]')
    plot_covariance(ax[0, 1], to_numpy(KkXX[0, 0, 2:, 2:]))
    # ax[1, 2].set_title('cov[f(x), g(x)]')
    # plot_covariance(ax[1, 2], to_numpy(KkXX[0, 0, :2, 2:]))

    plt.savefig('Coregionalization_covariances.pdf')

def plot_covariance_3D(ax, cov, n_std=3.0, **kwargs):
    eigval, eigvec = np.linalg.eig(cov)
    a, b, c = np.sqrt(eigval) * n_std
    phi, theta = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = a*np.cos(phi)*np.sin(theta)
    Y = b*np.sin(phi)*np.sin(theta)
    Z = c*np.cos(theta)
    return ax.plot_surface(X, Y, Z)


def cov_ellipse_height(n_std, cov):
    eigval, eigvec = np.linalg.eig(cov)
    width, height = np.sqrt(eigval) * n_std
    return width

def plot_covariance(ax, cov, n_std, axnames=["X", "Y"], scale=1, **kwargs):
    ax.set_aspect('equal')

    eigval, eigvec = np.linalg.eig(cov)
    width, height = np.sqrt(eigval) * n_std


    anglerad = np.arctan2(eigvec[0, 1], eigvec[0, 0])
    ellipse = Ellipse((0, 0), width, height, angle=np.rad2deg(anglerad),
                      fill=False, color='b', linewidth=2*scale)
    axis_x_pt = eigvec.T @ np.array([width, 0]) / 2
    if np.all(axis_x_pt < 0): axis_x_pt = -1 * axis_x_pt
    axis_x = Arrow(0, 0, axis_x_pt[0], axis_x_pt[1], color='g')

    axis_y_pt = eigvec.T @ np.array([0, height]) / 2
    if np.all(axis_y_pt < 0): axis_y_pt = -1 * axis_y_pt
    axis_y = Arrow(0,0, axis_y_pt[0], axis_y_pt[1], color='g')

    return [ ax.add_patch(p) for p in (ellipse, axis_x, axis_y)]

def plot_covariance_projections(axes, cov3D, axtitle, scale=1):
    names = (('$x$', '$y$'), ('$y$', r'$\theta$'), (r'$\theta$', '$x$'))
    covariances = (cov3D[:2, :2],
                   cov3D[1:, 1:],
                   cov3D[np.ix_([2, 0], [2, 0])])
    max_height = max(map(partial(cov_ellipse_height, 3.0), covariances))
    for ax, axname, cov in zip(axes, names, covariances):
        ax.set_title(axtitle + ' on ' + '-'.join(axname), usetex=True, fontsize=10 * scale)
        ax.tick_params(axis='both', labelsize=7 * scale)
        ax.set_xlabel(axname[0])
        ax.set_ylabel(axname[1])
        plot_covariance(ax, cov, n_std=3.0, scale=scale)
        ax.set_ylim(-max_height*1.30/2, max_height*1.30/2)
        ax.set_xlim(-max_height*1.30/2, max_height*1.30/2)


def unicycle_plot_covariances_vis(events_file):
    grouped_by_tag = load_tensorboard_scalars(events_file)
    matrix_var_FX_diag_t = np.asarray(list(zip(*grouped_by_tag['matrix/var_FX_diag_t']))[1])
    vector_var_FX_diag_t = np.asarray(list(zip(*grouped_by_tag['vector/var_FX_diag_t']))[1])
    figs = []
    for name, var_FX_diag_t in (('MVGP', matrix_var_FX_diag_t),
                                ('Coregionalization', vector_var_FX_diag_t)):
        fig, axes = plt.subplots(3, 3, figsize=(5, 6), sharey='row',
                                gridspec_kw=dict(hspace=0.05, wspace=0.30,
                                                 left=0.10, bottom=0.05,
                                                 right=0.98, top=0.95))
        fig.suptitle(name)
        for i in range(3):
            cov3D = var_FX_diag_t[0, 0, i*3:i*3+3, i*3:i*3+3]
            plot_covariance_projections(
                axes[i, :], cov3D,
                ('$\mbox{Var}(f(\mathbf{x}))$'
                if i == 0
                else '$\mbox{Var}(g(\mathbf{x})_{:,%d})$' % i),
                scale=1.0)
        fig.savefig(osp.join(osp.dirname(events_file), '%s_covariances_proj.pdf' % name))
        figs.append(fig)

    plt.show()

if '__main__' == __name__:
    ## Uncomment this to generate experimental data again
    #events_file = unicycle_plot_covariances_exp()
    ## Comment this if you do not want to use saved data
    events_file = "docs/saved-runs/unicycle_plot_covariances_v1.6.2-18-g15ad4c0/events.out.tfevents.1626640030.dwarf.12346.0"
    unicycle_plot_covariances_vis(events_file)

import math
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib import transforms

from bayes_cbf.misc import TBLogger, to_numpy
from bayes_cbf.pendulum import (PendulumDynamicsModel,
                                sampling_pendulum_data, learn_dynamics_from_data,
                                ControlRandom)

from bayes_cbf.control_affine_model import (ControlAffineRegressor, LOG as CALOG,
                                            ControlAffineRegressorExact,
                                            ControlAffineRegressorVector,
                                            ControlAffineRegMatrixDiag,
                                            ControlAffineRegVectorDiag,
                                            is_psd)

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
                             exp_tags=['pendulum_covariances'],
                             runs_dir='data/runs'),
        pendulum_dynamics_class=PendulumDynamicsModel
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

    plt.savefig('Corregionalization_covariances.pdf')

def plot_covariance(ax, cov, n_std=3.0, **kwargs):
    ax.set_aspect('equal')

    eigval, eigvec = np.linalg.eig(cov)
    width, height = np.sqrt(eigval) * n_std
    ellipse = Ellipse((0, 0), width, height, **kwargs)
    axis_x = FancyArrowPatch((0,0), (width, 0), mutation_scale=0.1*width, color='r')
    axis_y = FancyArrowPatch((0,0), (0, height), mutation_scale=0.1*height, color='g')
    patches = PatchCollection([ellipse, axis_x, axis_y])

    Ab = np.eye(3)
    Ab[:2, :2] =  eigvec
    transf = transforms.Affine2D(matrix=Ab)

    patches.set_transform(transf + ax.transData)
    ax.set_xlim(-width * 1.1, width * 1.1)
    ax.set_ylim(-height * 1.1, height * 1.1)
    return ax.add_collection(patches)


if '__main__' == __name__:
    learned_models = plot_pendulum_covariances()

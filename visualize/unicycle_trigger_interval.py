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

from bayes_cbf.control_affine_model import ControlAffineRegressorExact
from bayes_cbf.sampling import sample_generator_trajectory

def unicycle_trigger_interval_exp(
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

    logger.add_tensors('Train', dict(Xtrain=Xtrain, Utrain=Utrain,
                                     XdotTrain=XdotTrain))
    logger.add_tensors('Test', dict(Xtest=Xtest))
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


def unicycle_trigger_interval_vis(events_file):
    grouped_by_tag = load_tensorboard_scalars(events_file)
    matrix_var_FX_diag_t = np.asarray(list(zip(*grouped_by_tag['matrix/var_FX_diag_t']))[1])

if '__main__' == __name__:
    events_file = "docs/saved-runs/unicycle_plot_covariances_v1.6.2-18-g15ad4c0/events.out.tfevents.1626640030.dwarf.12346.0"
    unicycle_trigger_interval_vis(events_file)

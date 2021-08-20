import math
import glob
import os.path as osp
import os

from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse, FancyArrowPatch, Arrow
from matplotlib.collections import PatchCollection
from matplotlib import transforms

import torch

import bayes_cbf.unicycle_move_to_pose as bayes_cbf_unicycle

from bayes_cbf.misc import TBLogger, to_numpy, load_tensorboard_scalars
from bayes_cbf.unicycle_move_to_pose import (AckermannDrive, ControllerCLF,
                                             NoPlanner, CartesianDynamics,
                                             CLFCartesian, VisualizerZ,
                                             LearnedShiftInvariantDynamics,
                                             unicycle_learning_helps_avoid_getting_stuck_exp,
                                             obstacles_at_mid_from_start_and_goal)

from bayes_cbf.control_affine_model import ControlAffineRegressorExact
from bayes_cbf.sampling import sample_generator_trajectory


def rbf_knl(x, xp, sf, ls):
    return sf**2 * np.exp(-0.5*np.sum((x-xp)**2./ls**2,1))

def rbf_d_knl_d_x_xp_i(x, xp, i, sf, ls):
    return -(x[:,i]-xp[:,i])/ls[i]**2 * rbf_knl(x,xp, sf, ls)

def rbf_d2_knl_d_x_xp_i(x, xp, i, sf, ls):
    return ls[i]**(-2) * rbf_knl(x,xp, sf, ls) +  (x[:,i]-xp[:,i])/ls[i]**2 * rbf_d_knl_d_x_xp_i(x, xp, i, sf, ls);

def rbf_d3_knl_d_x_xp_i(x, xp, i, sf, ls):
    return -ls[i]**(-2) * rbf_d_knl_d_x_xp_i(x,xp,i, sf, ls) - ls[i]**(-2) * rbf_d_knl_d_x_xp_i(x,xp,i, sf, ls)
    +  (x[:,i]-xp[:,i])/ls[i]**2 *rbf_d2_knl_d_x_xp_i(x,xp,i, sf, ls)

def pdist(Xtest):
    return np.linalg.norm(Xtest[:, np.newaxis, :] - Xtest[np.newaxis, :, :])

def ndgridj(grid_min, grid_max, ns):
    # generates a grid and returns all combnations of that grid in a list
    # In:
    #   grid_min   1  x D  lower bounds of grid for each dimension separately
    #   grid_max   1  x D  upper bounds of grid for each dimension separately
    #   ns         1  x D  number of points for each dimension separately
    # Out:
    #   grid       Prod(ns) x D
    #   dist        1  x 1   distances in the grid
    #
    # Copyright (c) by Jonas Umlauft (TUM) under BSD License
    # Last modif ied: Jonas Umlauft 10/2018:

    D = len(ns)
    return np.moveaxis(
        np.mgrid[tuple(slice(min_, max_, n*(1j))
                  for min_, max_, n in zip(grid_min, grid_max, ns))],
        0, -1).reshape(-1, D)


def unicycle_trigger_interval_compute(
        events_file,
        out_file_Lfh_traj,
        out_file_tau_traj,
        out_file_xvel_traj,
        Nte = 1e3, # Number of testing samples on the grid
        deltaL = 1e-4, # Prob of f being Lipschitz cont
        zeta = 1e-2, # Margin of CBC separation
        L_alpha = 1, # Lipschitz constant of alpha in alpha(h(x))
        XteMin = [-0.1, -0.1, -np.pi/100], # Min for test grid in neighborhood of x_t
        XteMax = [0.1, 0.1, np.pi/100], # Max for test grid in neighborhood of x_t
        cbfs = partial(
            obstacles_at_mid_from_start_and_goal,
            torch.tensor([-3, -1, -math.pi/4]), # start position
            torch.tensor([0, 0, math.pi/4]), # end positoin
            term_weights=[0.7, 0.3] # weights for pos and orientation term
        ),
        dt=0.01 # trigger interval
):
    events_dir = osp.dirname(events_file)
    grouped_by_tag = load_tensorboard_scalars(events_file)
    knl_lengthscales = np.asarray(list(zip(*grouped_by_tag['vis/knl_lengthscale']))[1])
    knl_scalefactors = np.asarray(list(zip(*grouped_by_tag['vis/knl_scalefactor']))[1])
    knl_As = np.asarray(list(zip(*grouped_by_tag['vis/knl_A']))[1])
    knl_Bs = np.asarray(list(zip(*grouped_by_tag['vis/knl_B']))[1])
    x_traj = np.asarray(list(zip(*grouped_by_tag['vis/state']))[1])
    x_pred_traj = np.asarray(list(zip(*grouped_by_tag['vis/xtp1']))[1])

    E = x_traj.shape[-1]
    Ndte = np.floor(np.power(Nte,1/E)).astype(np.int64)
    Nte = (Ndte**E).astype(np.int64) # Round to a power of E

    nsteps = knl_lengthscales.shape[0]
    Lfh_traj = np.empty(nsteps)
    tau_traj = np.empty(nsteps)
    xvel_traj = np.empty(nsteps)

    Xtest_grid = ndgridj(XteMin, XteMax, Ndte*np.ones(E))
    r = np.max(pdist(Xtest_grid))
    hs = cbfs()
    for t in range(nsteps):
        sf = knl_scalefactors[t]
        ls = knl_lengthscales[t].flatten()
        Lk = np.linalg.norm(sf**2 *np.exp(-0.5)/ls);
        A = knl_As[t]
        B = knl_Bs[t]

        Xtest = Xtest_grid + x_traj[t]
        Lfs = np.zeros((E, 1));
        for e in range(E):
            maxk = max(A[e,e] * rbf_d2_knl_d_x_xp_i(Xtest, Xtest, e, sf, ls));
            Lkds = np.zeros((Nte, 1));
            for nte in range(Nte):
                Lkds[nte] = max(rbf_d3_knl_d_x_xp_i(Xtest, Xtest[nte:nte+1, :], e, sf, ls));
            Lkd = np.max(Lkds)
            Lfs[e] = np.sqrt(2*np.log(2*E/deltaL))*maxk + 12*np.sqrt(6*E)*max(
                maxk, np.sqrt(r*A[e,e]*Lkd)) # Eq (11) from the paper
        Lfh =  np.linalg.norm(Lfs) #  Eq (11) continued
        Lfh_traj[t] = Lfh
        print("Computed Lipschtz constant at %d is " % t, Lfh);

        Lh = max(torch.max(h.grad_cbf(torch.from_numpy(Xtest))).item() for h in hs)
        xvel = np.linalg.norm(x_pred_traj[t] - x_traj[t])/dt
        xvel_traj[t] = xvel
        print("Computed velocity %d is " % t, xvel);
        tau = (1/Lfh)*np.log(1+Lfh * zeta/((Lfh + L_alpha)*Lh*np.linalg.norm(xvel)))
        tau_traj[t] = tau
        print("Computed trigger time at %d is " % t, tau);

    np.savetxt(out_file_xvel_traj, xvel_traj)
    np.savetxt(out_file_Lfh_traj, Lfh_traj)
    np.savetxt(out_file_tau_traj, tau_traj)

def unicycle_trigger_interval_vis(file_Lfh_traj, file_tau_traj, file_xvel_traj):
    Lfh_traj = np.loadtxt(file_Lfh_traj)
    tau_traj = np.loadtxt(file_tau_traj)
    xvel_traj = np.loadtxt(file_xvel_traj)
    fig, ax = plt.subplots(3,1, figsize=(3.5, 6), sharex=True,
                           gridspec_kw=dict(hspace=0.05, left=0.18))
    ax[0].semilogy(range(Lfh_traj.shape[0]), Lfh_traj)
    ax[0].set_ylabel(r'$L_{\mathbf{f}_k}$')
    ax[1].semilogy(range(tau_traj.shape[0]), tau_traj)
    ax[1].set_ylabel(r'$\tau_k$')
    ax[2].semilogy(range(xvel_traj.shape[0]), xvel_traj)
    ax[2].set_ylabel(r'$\|\dot{\mathbf{x}}\|_2$')
    ax[2].set_xlabel('time (k)')
    fig.savefig(osp.join(events_dir, 'triggering_time.pdf'))
    plt.show()

if '__main__' == __name__:
    #events_dir = unicycle_learning_helps_avoid_getting_stuck_exp()
    # events_file = max(glob.glob(osp.join(events_dir, '*tfevents*')),
    #                   key=lambda f: os.stat(f).st_mtime)
    events_file = 'data/runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.6.2-29-gffc84c6/events.out.tfevents.1629403712.dwarf.9630.0'
    events_dir = osp.dirname(events_file)
    file_Lfh_traj = osp.join(events_dir, "Lfh.np.txt")
    file_tau_traj = osp.join(events_dir, "tau.np.txt")
    file_xvel_traj = osp.join(events_dir, "xvel.np.txt")
    data_files = (file_Lfh_traj, file_tau_traj, file_xvel_traj)
    if not all(osp.exists(d) for d in data_files):
        unicycle_trigger_interval_compute(events_file, *data_files)
    unicycle_trigger_interval_vis(*data_files)

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

import bayes_cbf.unicycle_move_to_pose as bayes_cbf_unicycle

from bayes_cbf.misc import TBLogger, to_numpy, load_tensorboard_scalars
from bayes_cbf.unicycle_move_to_pose import (AckermannDrive, ControllerCLF,
                                             NoPlanner, CartesianDynamics,
                                             CLFCartesian, VisualizerZ,
                                             LearnedShiftInvariantDynamics)

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

def unicycle_trigger_interval_vis(events_file):
    grouped_by_tag = load_tensorboard_scalars(events_file)
    knl_lengthscales = np.asarray(list(zip(*grouped_by_tag['vis/knl_lengthscale']))[1])
    knl_scalefactors = np.asarray(list(zip(*grouped_by_tag['vis/knl_scalefactor']))[1])
    knl_As = np.asarray(list(zip(*grouped_by_tag['vis/knl_A']))[1])
    knl_Bs = np.asarray(list(zip(*grouped_by_tag['vis/knl_B']))[1])
    x_traj = np.asarray(list(zip(*grouped_by_tag['vis/state']))[1])

    E = x_traj.shape[-1]
    Nte = 1e4
    deltaL = 1e-2
    Ndte = np.floor(np.power(Nte,1/E)).astype(np.int64)
    Nte = (Ndte**E).astype(np.int64) # Round to a power of E
    XteMin = [-1, -1, -np.pi/10]
    XteMax = [1, 1, np.pi/10] # Xrange and n data points

    nsteps = knl_lengthscales.shape[0]
    Lfh_traj = np.empty(nsteps)
    for t in range(nsteps):
        sf = knl_scalefactors[t]
        ls = knl_lengthscales[t]
        Lk = np.linalg.norm(sf**2 *np.exp(-0.5)/ls);

        Xtest = ndgridj(XteMin, XteMax, Ndte*np.ones(E)) + x_traj[t]

        r = np.max(pdist(Xtest))
        Lfs = np.zeros((E, 1));
        for e in range(E):
            maxk = max(rbf_d2_knl_d_x_xp_i(Xtest, Xtest, e, sf, ls));
            Lkds = np.zeros((Nte, 1));
            for nte in range(Nte):
                Lkds[nte] = max(rbf_d3_knl_d_x_xp_i(Xtest, Xtest[nte:nte+1, :], e, sf, ls));
            Lkd = np.max(Lkds)
            Lfs[e] = np.sqrt(2*np.log(2*E/deltaL))*maxk + 12*np.sqrt(6*E)*max(
                maxk, np.sqrt(r*Lkd)) # Eq (11) from the paper
        end
        Lfh =  np.linalg.norm(Lfs) #  Eq (11) continued
        print("Computed Lipschtz constant is ", Lfh);
        Lfh_traj[t] = Lfh

if '__main__' == __name__:
    events_file = unicycle_learning_helps_avoid_getting_stuck_exp()
    print(events_file)
    # events_file = 'data/runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.6.2-28-g529d7ff/events.out.tfevents.1629329336.dwarf.20145.0'
    unicycle_trigger_interval_vis(events_file)

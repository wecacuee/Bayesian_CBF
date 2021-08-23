import os
import os.path as osp
import glob

import numpy as np
import matplotlib.pyplot as plt

from bayes_cbf.unicycle_move_to_pose import unicycle_learning_helps_avoid_getting_stuck_exp
from bayes_cbf.trigger_interval import unicycle_trigger_interval_compute

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
    events_dir = unicycle_learning_helps_avoid_getting_stuck_exp()
    events_file = max(glob.glob(osp.join(events_dir, '*tfevents*')),
                      key=lambda f: os.stat(f).st_mtime)
    # events_file = 'docs/saved-runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.6.2-29-gffc84c6/events.out.tfevents.1629403712.dwarf.9630.0'
    # events_file = 'data/runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.6.3-1-g5fa08e8/events.out.tfevents.1629489011.dwarf.397.0'
    events_dir = osp.dirname(events_file)
    file_Lfh_traj = osp.join(events_dir, "Lfh.np.txt")
    file_tau_traj = osp.join(events_dir, "tau.np.txt")
    file_xvel_traj = osp.join(events_dir, "xvel.np.txt")
    data_files = (file_Lfh_traj, file_tau_traj, file_xvel_traj)
    if not all(osp.exists(d) for d in data_files):
        unicycle_trigger_interval_compute(events_file, *data_files)
    unicycle_trigger_interval_vis(*data_files)

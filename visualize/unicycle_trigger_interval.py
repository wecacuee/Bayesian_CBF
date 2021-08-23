import os
import os.path as osp
import glob

import numpy as np
import matplotlib.pyplot as plt

from bayes_cbf.unicycle_move_to_pose import unicycle_learning_helps_avoid_getting_stuck_exp
from bayes_cbf.trigger_interval import unicycle_trigger_interval_compute

def unicycle_trigger_interval_vis(data_files):
    Lfh_traj = np.loadtxt(data_files['Lfh.np.txt'])
    Lfh_num_traj = np.loadtxt(data_files['Lfh_num.np.txt'])
    tau_traj = np.loadtxt(data_files['tau.np.txt'])
    tau_num_traj = np.loadtxt(data_files['tau_num.np.txt'])
    xvel_traj = np.loadtxt(data_files['xvel.np.txt'])
    fig, ax = plt.subplots(2,2, figsize=(6.5, 4), sharex=True,
                           gridspec_kw=dict(hspace=0.05, wspace=0.3, left=0.10, right=0.95))
    ax[0, 0].semilogy(range(Lfh_traj.shape[0]), Lfh_traj)
    ax[0, 0].set_ylabel(r'$L_{\mathbf{f}_k}$')
    ax[1, 0].semilogy(range(tau_traj.shape[0]), tau_traj)
    ax[1, 0].set_ylabel(r'$\tau_k$')
    # ax[2].semilogy(range(xvel_traj.shape[0]), xvel_traj)
    # ax[2].set_ylabel(r'$\|\dot{\mathbf{x}}\|_2$')
    ax[1, 0].set_xlabel('time (k)')
    ax[1, 1].set_xlabel('time (k)')
    ax[0, 1].semilogy(range(Lfh_num_traj.shape[0]), Lfh_num_traj)
    ax[0, 1].set_ylabel(r'Numerical $L_{\mathbf{f}_k}$')
    ax[1, 1].semilogy(range(tau_num_traj.shape[0]), tau_num_traj)
    ax[1, 1].set_ylabel(r'Numerical $\tau_{k}$')
    fig.savefig(osp.join(events_dir, 'triggering_time.pdf'))
    plt.show()

if '__main__' == __name__:
    # events_dir = unicycle_learning_helps_avoid_getting_stuck_exp()
    # events_file = max(glob.glob(osp.join(events_dir, '*tfevents*')),
    #                   key=lambda f: os.stat(f).st_mtime)
    # events_file = 'docs/saved-runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.6.2-29-gffc84c6/events.out.tfevents.1629403712.dwarf.9630.0'
    events_file = 'data/runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.6.3-1-g5fa08e8/events.out.tfevents.1629498656.dwarf.4257.0'
    events_dir = osp.dirname(events_file)
    data_file_basenames = [
        "Lfh.np.txt",
        "Lfh_num.np.txt",
        "tau.np.txt",
        "tau_num.np.txt",
        "xvel.np.txt"
        ]
    data_files = {bn : osp.join(events_dir, bn) for bn in data_file_basenames}
    if not all(osp.exists(d) for d in data_files.values()):
        unicycle_trigger_interval_compute(events_file, data_files)
    unicycle_trigger_interval_vis(data_files)

import os.path as osp
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from bayes_cbf.misc import (load_tensorboard_scalars)
from bayes_cbf.plotting import plot_learned_2D_func_from_data

def learn_dynamics_plot_from_log(
        events_file='events.out.tfevents.1606951767.dwarf.13002.0'):
    """
    """
    logdata = load_tensorboard_scalars(events_file)
    events_dir = osp.dirname(events_file)
    fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, squeeze=False,
                            figsize=(14, 4))
    theta_omega_grid = logdata['plot_learned_2D_func/fx/true/theta_omega_grid'][0][1]
    FX_learned = logdata['plot_learned_2D_func/fx/learned/FX'][0][1]
    FX_true = logdata['plot_learned_2D_func/fx/true/FX'][0][1]
    Xtrain = logdata['plot_learned_2D_func/fx/Xtrain'][0][1]
    figtitle='Learned vs True'
    plot_learned_2D_func_from_data(theta_omega_grid, FX_learned, FX_true, Xtrain,
                                   axtitle='f(x)[{i}]',
                                   figtitle=figtitle,
                                   axs=axs[:, :2])

    GX_learned = logdata['plot_learned_2D_func/gx/learned/FX'][0][1]
    GX_true = logdata['plot_learned_2D_func/gx/true/FX'][0][1]
    plot_learned_2D_func_from_data(theta_omega_grid, GX_learned, GX_true, Xtrain,
                                   axtitle='g(x)[{i}]',
                                   figtitle=figtitle,
                                   axs=axs[:, 2:],
                                   ylabel=None)
    xmin = np.min(theta_omega_grid[0, ...])
    xmax = np.max(theta_omega_grid[0, ...])
    ymin = np.min(theta_omega_grid[1, ...])
    ymax = np.max(theta_omega_grid[1, ...])
    for ax in axs.flatten():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    fig.suptitle(figtitle)
    if hasattr(fig, "canvas") and hasattr(fig.canvas, "set_window_title"):
        fig.canvas.set_window_title(figtitle)
    fig.subplots_adjust(wspace=0.2,hspace=0.2, left=0.05, right=0.95)
    plot_file = osp.join(events_dir, 'learned_f_g_vs_true_f_g.pdf')
    fig.savefig(plot_file)
    #subprocess.run(["xdg-open", plot_file])
    return plot_file

if __name__ == '__main__':
    import sys
    default_events_file = 'events.out.tfevents.1606951767.dwarf.13002.0'
    learn_dynamics_plot_from_log(sys.argv[1]
                                 if len(sys.argv) == 2
                                 else default_events_file)

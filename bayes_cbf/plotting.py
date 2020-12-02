#plot the result
from functools import partial
import numpy as np
import torch
import matplotlib.pyplot as plt
import os.path as osp
from matplotlib import rc as mplibrc
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
mplibrc('text', usetex=True)

from bayes_cbf.misc import NoLogger

def rad2deg(rad):
    return rad / np.pi * 180

def plot_2D_f_func(f_func,
                   axes_gen = lambda FX: plt.subplots(1, FX.shape[-1])[1],
                   theta_range = slice(-np.pi, np.pi, np.pi/20),
                   omega_range = slice(-np.pi, np.pi,np.pi/20),
                   axtitle="f(x)[{i}]",
                   figtitle="Learned vs True",
                   xsample=torch.zeros(2,),
                   contour_levels=None,
                   ylabel=r"$\omega$",
                   xlabel=r"$\theta$",
                   logger=NoLogger()):
    # Plot true f(x)
    theta_omega_grid = np.mgrid[theta_range, omega_range]
    _, N, M = theta_omega_grid.shape
    D = xsample.shape[-1]
    Xgrid = torch.empty((N * M, D), dtype=torch.float32)
    Xgrid[:, :] = torch.from_numpy(xsample)
    Xgrid[:, :2] = torch.from_numpy(
        theta_omega_grid.transpose(1, 2, 0).reshape(-1, 2)).to(torch.float32)
    FX = f_func(Xgrid).reshape(N, M, D)
    logger.add_tensors("plot_2D_f_func", dict(theta_omega_grid=theta_omega_grid,
                                              FX=FX), 0)
    axs = axes_gen(FX)
    if contour_levels is None:
        contour_levels = [None]*len(axs)
    contour_sets = []
    for i, lvl in zip(range(len(axs)), contour_levels):
        axs[i].clear()
        ctf0 = axs[i].contourf(theta_omega_grid[0, ...], theta_omega_grid[1, ...],
                               FX[:, :, i].detach().cpu().numpy(),
                               levels=lvl)
        plt.colorbar(ctf0, ax=axs[i])
        contour_sets.append(ctf0)
        if axtitle: axs[i].set_title(axtitle.format(i=i))
        if ylabel and i == 0: axs[i].set_ylabel(ylabel)
        if xlabel: axs[i].set_xlabel(xlabel)

    fig = axs[0].figure
    fig.suptitle(figtitle)
    if hasattr(fig, "canvas") and hasattr(fig.canvas, "set_window_title"):
        fig.canvas.set_window_title(figtitle)
    fig.subplots_adjust(wspace=0.31)
    return contour_sets


def plot_results(time_vec, omega_vec, theta_vec, u_vec,
                 axs=None, figtitle="Pendulum"):
    #plot thetha
    if axs is None:
        fig = plt.figure(figsize=(7, 2.5))
        axs = fig.subplots(1,3)
    axs[0].clear()
    axs[0].plot(time_vec, rad2deg((theta_vec + np.pi) % (2*np.pi) - np.pi),
                ":", label = "theta (degrees)",color="blue")
    axs[0].set_ylabel(r"$\theta$ (degrees)")

    axs[1].clear()
    axs[1].plot(time_vec, omega_vec,":", label = "omega (rad/s)",color="blue")
    axs[1].set_ylabel(r"$\omega$ (rad/s)")

    axs[2].clear()
    axs[2].plot(time_vec, u_vec,":", label = "u", color="blue")
    axs[2].set_ylabel("u")

    #axs[1,1].clear()
    #axs[1,1].plot(time_vec, np.cos(theta_vec),":", label="cos(theta)", color="blue")
    #axs[1,1].set_ylabel("cos/sin(theta)")
    #axs[1,1].plot(time_vec, np.sin(theta_vec),":", label="sin(theta)", color="red")
    #axs[1,1].set_ylabel("sin(theta)")
    #axs[1,1].legend()

    fig = axs[0].figure
    fig.suptitle(figtitle)
    if hasattr(fig, "canvas") and hasattr(fig.canvas, "set_window_title"):
        fig.canvas.set_window_title(figtitle)
    fig.subplots_adjust(wspace=0.67)
    return axs


def plot_learned_2D_func(Xtrain, learned_f_func, true_f_func,
                         axtitle="f(x)[{i}]", figtitle="learned vs true",
                         axs=None,
                         logger=NoLogger()):
    if axs is None:
        fig, axs = plt.subplots(3,2)
    else:
        fig = axs.flatten()[0].figure
        fig.clf()
        axs = fig.subplots(3,2)
    theta_range = slice(Xtrain[:, 0].min(), Xtrain[:, 0].max(),
                        (Xtrain[:, 0].max() - Xtrain[:, 0].min()) / 20)
    omega_range = slice(Xtrain[:, 1].min(), Xtrain[:, 1].max(),
                        (Xtrain[:, 1].max() - Xtrain[:, 1].min()) / 20)
    csets = plot_2D_f_func(learned_f_func, axes_gen=lambda _: axs[1, :],
                           theta_range=theta_range, omega_range=omega_range,
                           axtitle="Learned " + axtitle,
                           xsample=Xtrain[-1, :],
                           logger=logger)
    csets = plot_2D_f_func(true_f_func, axes_gen=lambda _: axs[0, :],
                           theta_range=theta_range, omega_range=omega_range,
                           axtitle="True " + axtitle,
                           xsample=Xtrain[-1, :],
                           contour_levels=[c.levels for c in csets],
                           xlabel=None,
                           logger=logger)
    ax = axs[2,0]
    ax.plot(Xtrain[:, 0], Xtrain[:, 1], marker='*', linestyle='')
    ax.set_ylabel(r"$\omega$")
    ax.set_xlabel(r"$\theta$")
    ax.set_xlim(theta_range.start, theta_range.stop)
    ax.set_ylim(omega_range.start, omega_range.stop)
    #ax.set_title("Training data")
    fig.suptitle(figtitle)
    if hasattr(fig, "canvas") and hasattr(fig.canvas, "set_window_title"):
        fig.canvas.set_window_title(figtitle)
    fig.subplots_adjust(wspace=0.4,hspace=0.4)
    return axs

class LinePlotSerialization:
    @staticmethod
    def serialize(filename, axes):
        xydata = dict()
        for i, ax in enumerate(axes):
            for j, lin in enumerate(ax.lines):
                for xy, method in (("x", lin.get_xdata), ("y", lin.get_ydata)):
                    xydata["ax-{i}_lines-{j}_{xy}".format(i=i,j=j,xy=xy)] = method()
        np.savez_compressed(
            filename,
            **xydata
        )

    @staticmethod
    def example_plot(ax_lines_xydata):
        for i, ax in ax_lines_xydata.items():
            for j, xydata in ax_lines_xydata.items():
                axes[i].plot(xydata["x"], xydata["y"])

    @staticmethod
    def deserialize(filename, axes):
        xydata = np.loadz(filename)
        ax_lines_xydata = {}
        for key, val in xydata.items():
            _, istr,_, jstr, xy = key.split("_")
            i, j = int(istr), int(jstr)
            ax_lines_xydata.setdefault(i, {}).setdefault(j, {})[xy] = val
        return ax_lines_xydata


def plt_savefig_with_data(fig, filename):
    npz_filename = osp.splitext(filename)[0] + ".npz"
    LinePlotSerialization.serialize(npz_filename, fig.get_axes())
    fig.savefig(filename)

def draw_ellipse(ax, scale, theta, x0, **kwargs):
    ellipse = Ellipse((0,0),
                      width=2*scale[0],
                      height=2*scale[1],
                      fill=False,
                      **kwargs)

    transf = transforms.Affine2D() \
        .rotate_deg(theta * 180/np.pi) \
        .translate(x0[0], x0[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def demo_plot_ellipse():
    V = np.asarray([[1., 0.7],
                    [0.7, 1.]])
    mean = np.array([0.5, 0.3])
    fig, ax = plt.subplots()
    ax.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_aspect('equal')
    ax.plot(mean[0], mean[1], '.')
    draw_ellipse(ax, *var_to_scale_theta(V), mean)
    plt.show()


def rotmat2D(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def angle_from_rotmat(R):
    """
    >>> theta = np.random.rand() * 2*np.pi - np.pi
    >>> thetae = angle_from_rotmat(rotmat2D(theta))
    >>> np.allclose(thetae, theta)
    True
    """
    return np.arctan2(R[1, 0], R[0, 0])

def scale_theta_to_var(scale, theta):
    R = rotmat2D(theta)
    return R @ np.diag(scale/3) @ R.T

def var_to_scale_theta(V):
    """
    >>> scale = np.abs(np.random.rand(2)) * 10
    >>> theta = np.random.rand() * (2*np.pi) - np.pi
    >>> s, t = var_to_scale_theta(scale_theta_to_var(scale, theta))
    >>> allclose = partial(np.allclose, rtol=1e-2, atol=1e-5)
    >>> allclose(s, scale)
    True
    >>> allclose(t, theta)
    True
    """
    w, E = np.linalg.eig(V)
    scale = 3*w
    theta = angle_from_rotmat(E)
    return scale, theta

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    demo_plot_ellipse()


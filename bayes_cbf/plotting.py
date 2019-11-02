#plot the result
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as mplibrc
mplibrc('text', usetex=True)

def rad2deg(rad):
    return rad / np.pi * 180


def plot_2D_f_func(f_func,
                   axes_gen = lambda FX: plt.subplots(1, FX.shape[-1])[1],
                   theta_range = slice(-np.pi, np.pi, np.pi/20),
                   omega_range = slice(-np.pi, np.pi,np.pi/20),
                   axtitle="f(x)[{i}]"):
    # Plot true f(x)
    theta_omega_grid = np.mgrid[theta_range, omega_range]
    D, N, M = theta_omega_grid.shape
    FX = f_func(theta_omega_grid.transpose(1, 2, 0).reshape(-1, D)).reshape(N, M, D)
    axs = axes_gen(FX)
    for i in range(FX.shape[-1]):
        ctf0 = axs[i].contourf(theta_omega_grid[0, ...], theta_omega_grid[1, ...],
                               FX[:, :, i])
        plt.colorbar(ctf0, ax=axs[i])
        axs[i].set_title(axtitle.format(i=i))
        axs[i].set_ylabel(r"$\omega$")
        axs[i].set_xlabel(r"$\theta$")


def plot_results(time_vec, omega_vec, theta_vec, u_vec):
    #plot thetha
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(time_vec, rad2deg(theta_vec),
                  ":", label = "theta (degrees)",color="blue")
    axs[0,0].set_ylabel("theta (degrees)")
    axs[0,1].plot(time_vec, omega_vec,":", label = "omega (rad/s)",color="blue")
    axs[0,1].set_ylabel("omega")
    axs[1,0].plot(time_vec, u_vec,":", label = "u",color="blue")
    axs[1,0].set_ylabel("u")

    axs[1,1].plot(time_vec, np.cos(theta_vec),":", label="cos(theta)", color="blue")
    axs[1,1].set_ylabel("cos/sin(theta)")
    axs[1,1].plot(time_vec, np.sin(theta_vec),":", label="sin(theta)", color="red")
    axs[1,1].set_ylabel("sin(theta)")
    axs[1,1].legend()

    fig.suptitle("Pendulum")
    fig.subplots_adjust(wspace=0.31)
    plt.savefig('pendulum_data.pdf')
    #plt.show()


def plot_learned_2D_func(Xtrain, learned_f_func, true_f_func,
                         axtitle="f(x)[{i}]"):
    fig, axs = plt.subplots(3,2)
    theta_range = slice(np.min(Xtrain[:, 0]), np.max(Xtrain[:, 0]),
                        (np.max(Xtrain[:, 0]) - np.min(Xtrain[:, 0])) / 20)
    omega_range = slice(np.min(Xtrain[:, 1]), np.max(Xtrain[:, 1]),
                      (np.max(Xtrain[:, 1]) - np.min(Xtrain[:, 1])) / 20)
    plot_2D_f_func(true_f_func, axes_gen=lambda _: axs[0, :],
                   theta_range=theta_range, omega_range=omega_range,
                   axtitle="True " + axtitle)
    plot_2D_f_func(learned_f_func, axes_gen=lambda _: axs[1, :],
                   theta_range=theta_range, omega_range=omega_range,
                   axtitle="Learned " + axtitle)
    axs[2, 0].plot(Xtrain[:, 0], Xtrain[:, 1], marker='*', linestyle='')
    axs[2, 0].set_ylabel(r"$\omega$")
    axs[2, 0].set_xlabel(r"$\theta$")
    axs[2, 0].set_xlim(theta_range.start, theta_range.stop)
    axs[2, 0].set_ylim(omega_range.start, omega_range.stop)
    fig.subplots_adjust(wspace=0.3,hspace=0.42)

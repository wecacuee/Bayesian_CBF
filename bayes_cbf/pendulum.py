# stable pendulum

import numpy as np
from numpy import cos, sin
#plot the result
import matplotlib.pyplot as plt


def control_trivial(theta, w, m=None, l=None, g=None):
    assert m is not None
    assert l is not None
    assert g is not None
    u = m*g*sin(theta)
    return u


def env_pendulum(theta0,omega0,tau,m,g,l,numSteps, control=control_trivial):


    # initialize vectors
    time_vec = np.zeros(numSteps)
    theta_vec = np.zeros(numSteps)
    omega_vec = np.zeros(numSteps)
    u_vec = np.zeros(numSteps)
    #damage indicator
    damage_vec = np.zeros(numSteps)

    # set initial conditions

    theta = theta0
    omega = omega0
    time = 0

    # begin time-stepping

    for i in range(numSteps):
        omega_old = omega
        theta_old = theta
        u= control(theta, omega, m=m, g=g, l=l)
        # update the values
        omega = omega_old - (g/l)*sin(theta_old)*tau+(u/(m*l))*tau
        theta = theta_old + omega*tau
        # record the values
        time_vec[i] = tau*i
        omega_vec[i] = omega
        u_vec[i] = u
        #record and normalize theta to be in -pi to pi range
        theta_vec[i] = (((theta+np.pi) % (2*np.pi)) - np.pi)
        if 0<theta_vec[i]<np.pi/4:
            damage_vec[i]=1
    damge_perc=damage_vec.sum() * 100/numSteps
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)


def rad2deg(rad):
    return rad / np.pi * 180


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

    axs[1,1].plot(time_vec, cos(theta_vec),":", label="cos(theta)", color="blue")
    axs[1,1].set_ylabel("cos/sin(theta)")
    axs[1,1].plot(time_vec, sin(theta_vec),":", label="sin(theta)", color="red")
    axs[1,1].set_ylabel("sin(theta)")
    axs[1,1].legend()

    fig.suptitle("Pendulum")
    fig.subplots_adjust(wspace=0.31)
    plt.show()


def run_pendulum_experiment(#parameters
        theta0=5*np.pi/6,
        omega0=-0.01,
        tau=0.01,
        m=1,
        g=10,
        l=1,
        numSteps=1000,
        control=control_trivial):
    damge_perc,time_vec,theta_vec,omega_vec,u_vec = env_pendulum(
        theta0,omega0,tau,m,g,l,numSteps, control=control)
    plot_results(time_vec, omega_vec, theta_vec, u_vec)
    return (damge_perc,time_vec,theta_vec,omega_vec,u_vec)


def learn_dynamics(
        theta0=3*np.pi/4,
        omega0=0,
        tau=0.001,
        m=1,
        g=10,
        l=1,
        numSteps=5000):
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.gaussian_process import GaussianProcessRegressor
    damge_perc,time_vec,theta_vec,omega_vec,u_vec = env_pendulum(theta0,omega0,tau,m,g,l,numSteps)
    kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

    total_data = np.vstack((theta_vec, omega_vec, u_vec)).T
    Y= total_data[1:500,:2]
    Z= total_data[:499,:] #-1 remember!

    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0).fit(Z, Y)

    t99, cov = gp.predict(total_data[98:99,:], return_cov=True)
    assert np.allclose(total_data[99], t99)

    t502, cov = gp.predict(total_data[501:502,:], return_cov=True)
    assert np.allclose(total_data[502], t502)


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    import cvxopt
    from cvxopt import matrix
    #P = (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def control_cbf_clf(theta, w,
                    theta_c=np.pi/4,
                    c=1,
                    gamma_sr=1,
                    delta_sr=10,
                    gamma_col=1,
                    delta_col=np.pi/8,
                    m=None,
                    l=None,
                    g=None):
    assert m is not None
    assert l is not None
    assert g is not None
    def A_clf(theta, w):
        return l*w

    def b_clf(theta, w):
        return -c*(0.5*m*l**2*w**2+m*g*l*(1-cos(theta)))

    def A_sr(theta, w):
        return l*w

    def b_sr(theta, w):
        return gamma_sr*(delta_sr-w**2)+(2*g*sin(theta)*w)/(l)

    def A_col(theta, w):
        return -(2*w*(cos(delta_col)-cos(theta-theta_c)))/(m*l)

    def b_col(theta, w):
        return (gamma_col*(cos(delta_col)-cos(theta-theta_c))*w**2
                + w**3*sin(theta-theta_c)
                - (2*g*w*sin(theta)*(cos(delta_col)-cos(theta-theta_c)))/l)

    A_total = np.vstack([A(theta, w) for A in [A_clf, A_sr,A_col]])
    b_total = np.vstack([b(theta, w) for b in [b_clf, b_sr,b_col]])

    # u0 = l*g*sin(theta)
    # uopt = 0.1*g
    # contraints = A_total.dot(uopt) - b_total
    # assert contraints[0] <= 0
    # assert contraints[1] <= 0
    # assert contraints[2] <= 0

    if False:
        A_total_rho = np.hstack((A_total, np.zeros((A_total.shape[0], 1))))
        A_total_rho[0, -1] = -1
        from cvxopt import matrix
        P_rho = np.array([[1., 0],
                          [0, 100.]])
        q_rho = np.array([0., 0.])
        u_rho = cvxopt_solve_qp(P_rho, q_rho,
                            G=A_total_rho,
                            h=b_total)
        u = u_rho[0]
        return u
    else:
        A_clf_val = np.array([[A_clf(theta, w)]],  dtype='f8')
        b_clf_val = np.array([b_clf(theta, w)], dtype='f8')
        u = cvxopt_solve_qp(np.array([[1.]]),
                            np.array([0.]),
                            G=A_clf_val,
                            h=b_clf_val)
        assert A_clf_val.dot(u) - b_clf_val <= 0
        
        return u



if __name__ == '__main__':
    #run_pendulum_experiment(control=control_trivial)
    (damge_perc,time_vec,theta_vec,omega_vec,u_vec) = run_pendulum_experiment(control=control_cbf_clf)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from nm4p import rk4

# define the lorzrk function used by the RK routines
def lorzrk(s, t, param):
    '''
    Returns the RHS of the Lorenz model ODE
    Inputs
    ------
    s = state vector [x, y, z]
    t = independent variable (time, not used here)
    param = parameters [r, sigma, b]
    Output
    ------
    derivs = derivatives [dx/dt, dy/dt, dz/dt]
    '''

    # unravel the input vectors for clarity
    x1, y1, z1 = s[0], s[1], s[2]
    x2, y2, z2 = s[3], s[4], s[5]
    r = param[0]
    sigma = param[1]
    b = param[2]

    # return the derivatives
    derivs = np.empty(len(s))
    derivs[0] = sigma * (y1 - x1)
    derivs[1] = r*x1 - y1 - x1*z1
    derivs[2] = x1*y1 - b*z1
    derivs[3] = sigma * (y2 - x2)
    derivs[4] = r*x2 - y2 - x2*z2
    derivs[5] = x2*y2 - b*z2

    return derivs

def main(pos, param_r):
    state = pos
    r = param_r
    sigma = 10.
    b = 8. / 3.
    param = np.array([r, sigma, b])
    tau = 0.04
    err = 1e-3

    # loop over desired number of steps
    time = 0.
    nsteps = 500
    tplot = np.empty(nsteps)
    tauplot = np.empty(nsteps)
    diffplot = np.empty((3, nsteps))
    x1plot, y1plot, z1plot = np.empty(nsteps), np.empty(nsteps), np.empty(nsteps)
    x2plot, y2plot, z2plot = np.empty(nsteps), np.empty(nsteps), np.empty(nsteps)

    for istep in range(nsteps):
        # Record values for plotting
        x1, y1, z1 = state[0], state[1], state[2]
        x2, y2, z2 = state[3], state[4], state[5]
        tplot[istep] = time
        x1plot[istep] = x1
        y1plot[istep] = y1
        z1plot[istep] = z1
        x2plot[istep] = x2
        y2plot[istep] = y2
        z2plot[istep] = z2
        diffplot[0, istep] = np.abs(x2 - x1)
        diffplot[1, istep] = np.abs(y2 - y1)
        diffplot[2, istep] = np.abs(z2 - z1)

        total_diff = np.linalg.norm(diffplot, axis=0)

        if (istep + 1) % 50 == 0:
            print(f'Completed {istep} steps out of {nsteps}')

        # Find new state using adaptive Runge-Kutta
        state = rk4(state, time, tau, lorzrk, param)
        time += tau

    # Graph the time series x(t)
    fig = plt.figure()
    plt.plot(tplot, x1plot, 'g-')
    plt.plot(tplot, x2plot, 'r--')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title(f'Lorenz model time series, r = {r}')

    fig = plt.figure()
    plt.semilogy(tplot, diffplot[0, :])
    plt.xlabel('time')
    plt.ylabel('Absolute difference in x-coordinates')
    plt.grid(True, which='both')

    fig=plt.figure()
    plt.semilogy(tplot, total_diff)
    plt.xlabel('time')
    plt.ylabel('Norm difference')
    plt.grid(True, which='both')

    # Graph the x, y, z, phase space trajectory
    # Mark the location of the three stredy states
    x_ss, y_ss, z_ss = np.empty(3), np.empty(3), np.empty(3)
    x_ss[0] = 0.
    y_ss[0] = 0.
    z_ss[0] = 0.
    x_ss[1] = np.sqrt(b*(r-1))
    y_ss[1] = x_ss[1]
    z_ss[1] = r - 1
    x_ss[2] = -np.sqrt(b*(r-1))
    y_ss[2] = x_ss[2]
    z_ss[2] = r - 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x1plot, y1plot, z1plot, 'g-')
    ax.plot(x2plot, y2plot, z2plot, 'r--')
    ax.plot(x_ss, y_ss, z_ss, '*')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Lorenz model phase space, r = {r}')
    print(state)
    plt.show()

if __name__ == '__main__':
    pos = [1., 1., 20., 1., 1., 20.001]
    param_r = 28
    main(pos, param_r)
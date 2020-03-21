import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nm4p import rk4
from nm4p import rka

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
    x, y, z = s[0], s[1], s[2]
    r = param[0]
    sigma = param[1]
    b = param[2]

    # return the derivatives
    derivs = np.empty(len(s))
    derivs[0] = sigma * (y - x)
    derivs[1] = r*x - y - x*z
    derivs[2] = x*y - b*z

    return derivs

def main(pos, param_r):
    state = pos
    r = param_r
    sigma = 10.
    b = 8. / 3.
    param = np.array([r, sigma, b])
    tau = 1.
    err = 1e-3

    # loop over desired number of steps
    time = 0.
    nsteps = 500
    tplot = np.empty(nsteps)
    tauplot = np.empty(nsteps)
    xplot, yplot, zplot = np.empty(nsteps), np.empty(nsteps), np.empty(nsteps)

    for istep in range(nsteps):
        # Record values for plotting
        x, y, z = state[0], state[1], state[2]
        tplot[istep] = time
        tauplot[istep] = tau
        xplot[istep] = x
        yplot[istep] = y
        zplot[istep] = z

        if (istep + 1) % 50 == 0:
            print(f'Completed {istep} steps out of {nsteps}')

        # Find new state using adaptive Runge-Kutta
        [state, time, tau] = rka(state, time, tau, err, lorzrk, param)

    # Print max and min time step returned by rka
    taumin = np.min(tauplot[1:nsteps])
    taumax = np.max(tauplot[1:nsteps])
    print(f'Adaptive Runge-Kutta time steps: max = {taumax}, min = {taumin}')

    # Graph the time series x(t)
    plt.plot(tplot, xplot, '-')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title(f'Lorenz model time series, r = {r}')

    # Graph the x, y, z, phase space trajectory
    # Mark the location of the three stredy states
    x_ss, y_ss, z_ss = np.empty(len(state)), np.empty(len(state)), np.empty(len(state))
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
    ax.plot(xplot, yplot, zplot, '-')
    ax.plot(x_ss, y_ss, z_ss, '*')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Lorenz model phase space, r = {r}')
    print(state)
    plt.show()

if __name__ == '__main__':
    pos = [-8.5, -8.5, 27.]
    param_r = 28
    main(pos, param_r)
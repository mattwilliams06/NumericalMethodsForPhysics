import numpy as np

def rk4(x, t, tau, derivsRK, param):
    ''' 4th order Runga-Kutta integrator
    Inputs
    ------
    x = current state vector
    t = independent variable (usually time)
    tau = step size (usually time)
    derivsRK = RHS of the ODE; derivsRK is the name of
               the function that returns dx/dt
               Calling format: derivsRK(x, t param)
    param = extra parameters passed to derivsRK

    Outputs
    -------
    xout = new value of x after a step size of tau
    '''

    half_tau = 0.5 * tau
    F1 = derivsRK(x, t, param)
    xtemp = x + half_tau * F1
    F2 = derivsRK(xtemp, t + half_tau, param)
    xtemp = x + half_tau * F2
    F3 = derivsRK(xtemp, t + half_tau, param)
    xtemp = x + tau * F3
    F4 = derivsRK(xtemp, t + tau, param)
    xout = x + tau/6 * (F1 + 2 * (F2 + F3) + F4)

    return xout

def rka(x, t, tau, err, derivsRK, param):
    ''' Adaptive Runge-Kutta routine
    Inputs
    ------
    x = current state vector
    t = independent variable (usually time)
    tau = step size (usually time)
    err = desired fractional local truncation error
    derivsRK = RHS of the ODE; derivsRK is the name of
               the function that returns dx/dt
               Calling format: derivsRK(x, t param)
    param = extra parameters passed to derivsRK
    Outputs
    -------
    xsmall = new state vector
    t = new value of independent variable
    tau = new step size for the next call to rka
    '''

    # set initial variables
    tsave, xsave = t, x
    s1, s2 = 0.9, 4.0
    eps = 1e-15

    # loop over max attempts to satisfy error bounds
    xtemp = np.empty(len(x))
    xsmall = np.empty(len(x))

    maxtry = 100
    for i in range(maxtry):
        # take two small time steps
        half_tau = 0.5 * tau
        xtemp = rk4(xsave, tsave, half_tau, derivsRK, param)
        t = tsave + half_tau
        xsmall = rk4(xtemp, t, half_tau, derivsRK, param)

        # take one big time step
        t = tsave + tau
        xbig = rk4(xsave, t, tau, derivsRK, param)

        # compute the estimated truncation error
        scale = err * (np.abs(xbig) + np.abs(xsmall)) / 2.
        xdiff = xsmall - xbig
        errRatio = np.max(np.abs(xdiff) / (scale + eps))

        # estimate new tau
        tau_old = tau
        tau = s1 * tau_old * errRatio**(-0.2)
        if tau > s2*tau_old:
            tau = tau_old / 2.
        elif tau < tau_old / s2:
            tau = tau_old / s2
        else:
            tau = tau * 1.

        # if error is acceptable, return computed values
        if errRatio < 1.:
            return np.array([xsmall, t, tau])

    # issue error message if error bound never satisfied
    print('ERROR: Adaptive Runge-Kutta routine failed')
    return np.array([xsmall, t, tau])
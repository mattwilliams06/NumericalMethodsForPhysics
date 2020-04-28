def dftcs():
    '''
    Program to solve the Fourier heat diffusion equation using the forward time-centered space (FTCS) scheme
    '''

    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize parameters
    tau = 1e-4         # time step (sec)
    N = 61            # Number of grid points
    L = 1.            # System length (extends from -L/2 to L/2)
    h = L/(N-1)       # Grid size
    kappa = 1         # Diffusion coefficient
    coeff = kappa * tau /h**2
    if coeff < 0.5:
        print('Solution expected to be stable')
    else:
        print('WARNING: Solution instability expected')

    # Set initial boundary conditions
    tt = np.zeros((N))    # Initialize temperature to zero at all points
    tt[int(N/2)] = 1./h   # Initial condition is delta function in the grid center
    # Boundary conditions are adjusted based on the code below

    # Set up loop and plot variables
    xplot = np.arange(N)*h - L/2     # Record the x-scale for plots
    iplot = 0                        # Counter used to count plots
    nstep = 300                      # Max number of iterations
    nplots = 50                      # Number of snapshots to take
    plot_step = nstep/nplots         # Number of time-steps between plots

    # Loop over the desired number of time steps
    ttplot = np.empty((N, nplots))
    tplot = np.empty(nplots)
    for istep in range(nstep): ## MAIN LOOP ##
        # Compute new temperatures using FTCS scheme
        # If Dirichlet:
        tt[1:(N-1)] = tt[1:(N-1)] + coeff*(tt[2:N] + tt[0:(N-2)] - 2*tt[1:(N-1)])
        # If Neumann, uncomment the 2 lines below:
        tt[0] = tt[0] + coeff*(tt[1] - tt[0])
        tt[-1] = tt[-1] + coeff*(tt[-2] - tt[-1])

        # Periodically record temperature for plotting
        if (istep+1) % plot_step < 1:       # Every plot_step steps
            ttplot[:, iplot] = np.copy(tt)  # Record tt for plotting
            tplot[iplot] = (istep+1) * tau  # Record time for plotting
            iplot += 1

    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    Tp, Xp = np.meshgrid(tplot, xplot)
    ax.plot_surface(Tp, Xp, ttplot, rstride=2, cstride=2, cmap=cm.hot)
    ax.set_xlabel('time')
    ax.set_ylabel('x')
    ax.set_zlabel('T(x, t)')
    ax.set_title('Diffusion of delta spike')
    plt.show()

    # Plot temperature vs x and t as a contour plot
    levels = np.linspace(0., 10., num=21)
    ct = plt.contour(tplot, xplot, ttplot, levels)
    plt.clabel(ct)
    plt.xlabel('time')
    plt.ylabel('x')
    plt.title('Temperature contour plot')
    plt.show()

if __name__ == '__main__':
    dftcs()


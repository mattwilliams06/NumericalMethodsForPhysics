def neutrn3D():
    ''' Program to determine the 3-dimensional critical volume of a nuclear pile. The program uses a
    forward time-centered space (FTCS) scheme to solve the diffusion equation, a system of partial differential
    equations in space and time.

    Author: Matt Williams
    matt.williams@alum.mit.edu
    '''

    # Import the required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize system and solver parameters
    tau = 5e-4                    # Time step
    N = 61                        # Number of nodes (equal in all directions)
    Lx = 4.                       # System lengths
    Ly = 4.
    Lz = np.pi
    hx = Lx / (N - 1)             # Node spacing in all directions
    hy = Ly / (N - 1)
    hz = Lz / (N - 1)
    D = 1.                        # Neutron diffusion coefficent (m^2/s)
    C = 1.                        # Generation rate (1/s)
    coeffx = D * tau / hx ** 2    # Defining coefficients for clarity later
    coeffy = D * tau / hy ** 2
    coeffz = D * tau / hz ** 2
    coeff2 = C * tau
    if (coeffx < 0.5) and (coeffy < 0.5) and (coeffz < 0.5):
        print('Solution expected to be stable')
    else:
        print('WARNING: Solution instability expected')

    # Set up loop and plot variables
    xplot = np.arange(N)*hx - Lx/2
    yplot = np.arange(N)*hy - Ly/2
    zplot = np.arange(N)*hz - Lz/2
    iplot = 0
    nstep = 10000
    nplots = 50
    plot_step = nstep / nplots

    # Set up initial and boundary conditions
    nn = np.zeros((N, N, N))         # Initialize neutron density at all points to be 0
    nn[int(N/4), int(N/2), int(N/4)] =  1 / hx   # Triangular delta function in middle of pile
    # As is, the boundary conditions are Dirichlet on the edges (neutron density = 0 due to leakage)


    # Loop over desired time steps
    nnplot = np.empty((nplots, N, N, N))
    tplot = np.empty(nplots)
    nAve = np.empty(nplots)

    for istep in range(nstep):
        nn[1:(N-1), :, :] = nn[1:(N-1), :, :] + \
                            coeffx * (nn[2:N, :, :] + nn[0:(N-2), :, :] - 2*nn[1:(N-1), :, :]) + \
                            coeff2 * nn[1:(N-1), :, :]

        nn[:, 1:(N - 1), :] = nn[:, 1:(N - 1), :] + \
                              coeffy * (nn[:, 2:N, :] + nn[:, 0:(N - 2), :] - 2 * nn[:, 1:(N - 1), :]) + \
                              coeff2 * nn[:, 1:(N - 1), :]

        nn[:, :, 1:(N - 1)] = nn[:, :, 1:(N - 1)] + \
                              coeffz * (nn[:, :, 2:N] + nn[:, :, 0:(N - 2)] - 2 * nn[:, :, 1:(N - 1)]) + \
                              coeff2 * nn[:, :, 1:(N - 1)]

        # Periodically record density for plotting
        if (istep + 1) % plot_step < 1:
            nnplot[iplot, :, :, :] = np.copy(nn)
            tplot[iplot] = (istep + 1) * tau
            nAve[iplot] = np.mean(nn)
            iplot += 1
            print(f'Finished {istep} of {nstep} steps')

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    xplot = np.arange(N)*hx - Lx/2
    yplot = np.arange(N)*hy - Ly/2
    Xp, Yp = np.meshgrid(xplot, yplot)
    ax.plot_surface(Xp, Yp, nnplot[-1, :, :, int((N-1)/2)], rstride=2, cstride=2, cmap=cm.jet)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('neutron density (x, t)')
    ax.set_title(f'Neutron density at half core-height')

    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # xplot = np.arange(N) * hx - Lx / 2
    # yplot = np.arange(N) * hy - Ly / 2
    # zplot = np.arange(N) * hz - Lz / 2
    # Xp, Yp, Zp = np.meshgrid(xplot, yplot, zplot)
    # ax.contour3D(Xp, Yp, Zp, nnplot[-1, :, :, int((N - 1) / 2)], cmap=cm.jet)
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('neutron density (x, t)')
    # ax.set_title(f'Neutron density at z = {Lz/2}')

    # Plot average neutron density vs time
    fig = plt.figure()
    plt.plot(tplot, nAve, '*')
    plt.xlabel('time')
    plt.ylabel('Total average density')
    plt.title(f'Lx = {Lx}, Ly = {Ly}, Lz = {Lz}')
    plt.show()


if __name__ == '__main__':
    neutrn3D()
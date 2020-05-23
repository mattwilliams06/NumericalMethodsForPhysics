def neutrn():
    # Program to solve neutron diffusion using FTCS
    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize parameters
    tau = 5e-4
    a = 10.
    N = 61
    L = 0.341
    Lext = L*a
    h = Lext/(N-1)
    D = 1.        # Neutron diffusion coefficent (m^2/s)
    C = 1.        # Generation rate (1/s)
    coeff = D * tau / h**2

    ## Diffuser addition ##
    ## Uncomment next 3 lines for diffuser ##
    C = np.zeros(N)
    h_arr = np.array([h*i for i in range(N)])
    C[(h_arr >= Lext/4) & (h_arr <= 3*Lext/4)] = 1.

    coeff2 = C * tau
    if coeff < 0.5:
        print('Solution expected to be stable')
    else:
        print('WARNING: Solution instability expected')

    # Set up initial and boundary conditions
    nn = np.zeros(N)      # Initialize neutron density at all points to be 0
    nn_new = np.zeros(N)  # Temporary array for use in computation
    nn[int(N/2)] = 1/h    # Triangular delta function in middle of pile
    # As is, the boundary conditions are Dirichlet on the edges (neutron density = 0 due to leakage)

    # Set up loop and plot variables
    xplot = np.arange(N)*h - Lext/2
    iplot = 0
    nstep = 12000
    nplots = 50
    plot_step = nstep / nplots

    # Loop over desired time steps
    nnplot = np.empty((N, nplots))
    tplot = np.empty(nplots)
    nAve = np.empty(nplots)
    for istep in range(nstep): ## MAIN LOOP ##
        nn[1:(N-1)] = nn[1:(N-1)] + coeff * (nn[2:N] + nn[0:(N-2)] - 2*nn[1:(N-1)]) + coeff2[1:(N-1)] * nn[1:(N-1)]

        # Periodically record density for plotting
        if (istep + 1) % plot_step < 1:
            nnplot[:, iplot] = np.copy(nn)
            tplot[iplot] = (istep + 1) * tau
            nAve[iplot] = np.mean(nn)
            iplot += 1
            print(f'Finished {istep} of {nstep} steps')

    # Plot density versus x and t as a 3D surface plot
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    Tp, Xp = np.meshgrid(tplot, xplot)
    ax.plot_surface(Tp, Xp, nnplot, rstride=2, cstride=2, cmap=cm.jet)
    ax.set_xlabel('time')
    ax.set_ylabel('x (m)')
    ax.set_zlabel('neutron density (x, t)')
    ax.set_title(f'1D Neutron diffusion with pile length {L:.2f} m')

    # Plot average neutron density vs time
    fig = plt.figure()
    plt.plot(tplot, nAve, '*')
    plt.xlabel('time')
    plt.ylabel('average density')
    plt.title(f'L = {L}, ($L_c = \pi$)')
    plt.show()

if __name__ == '__main__':
    neutrn()
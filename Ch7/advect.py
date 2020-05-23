def advect():
    ''' Program to solve the advection equation using the various hyperbolic PDE schemes '''

    import numpy as np
    import matplotlib.pyplot as plt

    # Select the numerical parameters
    method = 3         # 1 == FTCS; 2 == Lax; 3 == Lax-Wendroff
    N = 50             # Number of grid points
    L = 1.             # System size
    h = L/N            # Grid spacing
    c = 1.             # Wave speed

    print(f'Time for wave to move one grid spacing is {h/c} sec')
    tau = float(input('Enter the time step: '))
    coeff = -c*tau/(2.*h)    # Coefficient used in numerical advection equation
    coefflw = 2*coeff**2     # Coefficient for Lax-Wendroff
    print(f'Wave circles the system in {L/(c*tau)} steps')
    nsteps = int(input('Enter the number of steps: '))

    # Set initial and boundary conditions
    sigma = 0.1              # Width of Gaussian pulse
    k_wave = np.pi/sigma     # Wave number of the cosine
    x = np.arange(N)*h - L/2 # Coordinates of grid points
    # Initial condition is a Gaussian-cosine pulse
    a = np.empty(N)
    for i in range(N):
        a[i] = np.cos(k_wave * x[i]) * np.exp(-x[i]**2/(2*sigma**2))
    # Use periodic boundary conditions
    ip = np.arange(N) + 1
    ip[N-1] = 0        # ip = i + 1 with periodic b.c.
    im = np.arange(N) - 1
    im[0] = N - 1      # im = i - 1 with periodic b.c.

    # Initilize plotting variables
    iplot = 1          # Plot counter
    nplots = 50        # Number of plots
    aplot = np.empty((N, nplots))
    tplot = np.empty(nplots)
    aplot[:, 0] = np.copy(a)     # Copy initial condition
    tplot[0] = 0                 # Initial time
    plotStep = nsteps/nplots + 1 # Number of steps between plots

    # Loop over desired number of steps
    for istep in range(nsteps):   ## MAIN LOOP ##
        # Compute new values of wave amplitude using FTCS, Lax, or Lax-Wendroff
        if method == 1:      ## FTCS ##
            a[:] = a[:] + coeff * (a[ip] - a[im])
        elif method == 2:    ## Lax Method ##
            a[:] = 0.5 * (a[ip] + a[im]) + coeff * (a[ip] - a[im])
        else:                ## Lax-Wendroff ##
            a[:] = (a[:] + coeff * (a[ip] - a[im]) + coefflw * (a[ip] + a[im] - 2*a[:]))

        # Periodically record a(t) for plotting
        if (istep + 1) % plotStep < 1:
            aplot[:, iplot] = np.copy(a)
            tplot[iplot] = tau * (istep + 1)
            iplot += 1

    # Plot the initial and final states
    plt.figure()
    plt.plot(x, aplot[:, 0], label='Initial state')
    plt.plot(x, a, label='Final state')
    plt.xlabel('x')
    plt.ylabel('a(x, t)')
    plt.legend()

    # PLot the wave amplitude versus position and time
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Tp, Xp = np.meshgrid(tplot[0:iplot], x)
    ax.plot_surface(Tp, Xp, aplot[:, 0:iplot], rstride=1, cstride=1, cmap=cm.cool)
    ax.view_init(elev=30, azim=190)
    ax.set_ylabel('Position')
    ax.set_xlabel('Time')
    ax.set_zlabel('Amplitude')
    plt.show()

if __name__ == '__main__':
    advect()
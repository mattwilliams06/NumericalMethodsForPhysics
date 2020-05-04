def main():
    ''' Simulation of a Faraday cage. The problem will be a solution to Laplace's Equation given
    potentials at the walls, and the cage in the middle where the potential will be zero. The SOR method
    will be used.
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize parameters
    N = 60  # Number of grid points on each side
    L = 1.  # Length of system
    h = L / (N - 1)  # Grid spacing
    x = np.arange(N) * h
    y = np.arange(N) * h

    # Set omega for the SOR method
    omegaOpt = 2. / (1 + np.sin(np.pi / N))  # Theoretical optimum
    print(f'Theoretical optimum omega: {omegaOpt}')
    omega = float(input('Enter desired omega: '))

    # Initialize the flux matrix and set the boundary and Faraday Cage conditions
    phi = np.zeros((N, N))
    # Set initial guess for flux
    for i in range(1, N):
        for j in range(1, N):
            phi[i, j] = 4. / (np.pi * np.sinh(np.pi)) * \
                        np.sin(np.pi * x[i] / L) * np.sinh(np.pi * y[j] / L)
    phi[:, 0] = 0.              # Left wall flux
    phi[:, -1] = 100.           # Right wall flux
    for i, y_ in enumerate(y):  # Top/bottom wall flux, varying linearly from left to right
        phi[0, i] = 100. * y_
        phi[-1, i] = 100. * y_

    # Loop until desired fractional change per iteration is achieved
    newphi = np.copy(phi)  # Copy of the solution (used by Jacobi method)
    iterMax = N ** 2  # Set as max to avoid excessively lon runs
    change = np.empty(iterMax)
    changeDesired = 1e-4  # Stop when the change is less than desired
    print(f'Desired fractional change: {changeDesired}')

    for iter in range(iterMax):  ## MAIN LOOP ##
        changeSum = 0.
        for i in range(1, N - 1):  # Loop over interior nodes only
            for j in range(1, N - 1):
                phi[19, 19] = 0.
                phi[29, 19] = 0.
                phi[39, 19] = 0.
                phi[19, 29] = 0.
                phi[19, 39] = 0.
                phi[29, 39] = 0.
                phi[39, 29] = 0.
                phi[39, 39] = 0.
                temp = 0.25 * omega * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1]) + (1 - omega) * \
                       phi[i, j]
                changeSum += (1 - phi[i, j] / temp)
                phi[i, j] = temp

        # Check if iteration change is small enough to halt the loop
        change[iter] = changeSum / (N - 2) ** 2  # Averaging the change sum
        if (iter + 1) % 10 < 1:
            print(f'After {iter + 1} iterations, fractional change = {change[iter]}')
        if change[iter] < changeDesired:
            print(f'Desired accuracy achieved after {iter + 1} iterations')
            print(f'Breaking out of main loop')
            # print(newphi == phi)
            break
        if np.abs(change[iter] - change[iter-1]) < changeDesired*.1:
            print(f'No change in results. Breaking after {iter} iterations.')
            break
    # Plot final estimate of potential as a contour plot
    levels = np.linspace(0, 100, 11)
    ct = plt.contour(x, y, np.flipud(np.rot90(phi)), levels)
    plt.clabel(ct, fmt='%1.2f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Potential after {iter} iterations')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Xp, Yp = np.meshgrid(x, y)
    ax.plot_surface(Yp, Xp, phi, rstride=1, cstride=1, cmap=cm.hot)
    ax.view_init(elev=30, azim=210)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\Phi(x, y)$')
    plt.show()

if __name__ == '__main__':
    main()

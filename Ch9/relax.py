def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize parameters
    method = 3   # 1 == Jacobi, 2 == Gauss-Seidel, 3 == SOR
    N = 50       # Number of grid points on each side
    L = 1.       # Length of system
    h = L/(N-1)  # Grid spacing
    x = np.arange(N)*h
    y = np.arange(N)*h

    # Select over-relaxation factor (SOR only)
    if method == 3:
        omegaOpt = 2./(1+np.sin(np.pi/N))   # Theoretical optimum
        print(f'Theoretical optimum omega: {omegaOpt}')
        omega = float(input('Enter desired omega: '))

    # Set initial guess as first term in separation of variables solution
    phi0 = 1.          # Potential at boundary phi(x, y=Ly)
    phi = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            phi[i, j] = phi0 * .0001/(np.pi*np.sinh(np.pi)) * \
                        np.sin(np.pi*x[i]/L) * np.sinh(np.pi*y[j]/L)

    # Set boundary conditions
    phi[0, :] = 0.
    phi[-1, :] = 0.
    phi[:, 0] = 0.
    phi[:, -1] = phi0*np.ones(N)

    print(f'Potential at y=L is {phi0}')
    print('Potential is zero on all other boundaries')

    # Loop until desired fractional change per iteration is achieved
    newphi = np.copy(phi)  # Copy of the solution (used by Jacobi method)
    iterMax = N ** 2       # Set as max to avoid excessively lon runs
    change = np.empty(iterMax)
    changeDesired = 1e-4   # Stop when the change is less than desired
    print(f'Desired fractional change: {changeDesired}')
    for iter in range(iterMax):
        changeSum = 0.

        if method == 1:  ## Jacobi method MAIN LOOP ##
            for i in range(1, N-1):  # Loop over interior nodes only
                for j in range(1, N-1):
                    newphi[i, j] = 0.25 * (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])

                    changeSum += np.abs(1. - phi[i,j]/newphi[i,j])
            phi = np.copy(newphi)

        elif method == 2:  ## Gauss-Seidel method MAIN LOOP ##
            for i in range(1, N-1):  # Loop over interior nodes only
                for j in range(1, N-1):
                    temp = 0.25 * (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])
                    changeSum += (1 - phi[i,j]/temp)
                    phi[i, j] = temp

        else: ## SOR method MAIN LOOP ##
            for i in range(1, N-1):  # Loop over interior nodes only
                for j in range(1, N-1):
                    temp = 0.25 * omega * (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1]) + (1 - omega) * phi[i, j]
                    changeSum += (1 - phi[i,j]/temp)
                    phi[i, j] = temp

        # Check if iteration change is small enough to halt the loop
        change[iter] = changeSum/(N-2)**2  # Averaging the change sum
        if (iter+1) % 10 < 1:
            print(f'After {iter+1} iterations, fractional change = {change[iter]}')
        if change[iter] < changeDesired:
            print(f'Desired accuracy achieved after {iter+1} iterations')
            print(f'Breaking out of main loop')
            #print(newphi == phi)
            break

    # Plot final estimate of potential as a contour plot
    levels = np.linspace(0, 1, 11)
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
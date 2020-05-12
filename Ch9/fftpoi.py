def main():
    '''Program to solve the Poisson equation using the multiple Fourier transform method.
    Boundary conditions are periodic.
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize parameters
    eps0 = 8.8542e-12           # Permittivity [C^2/(N-m^2)]
    N = 50                      # Grid points per side
    L = 1.                      # Length of each side [m]
    h = L/N                     # Grid spacing for periodic boundary conditions
    x = (np.arange(N) + 1./2)*h  # Coordinates of grid points
    y = np.copy(x)

    # Set up charge density rho(i, j)
    rho = np.zeros((N, N))      # Initialize charge density to zero
    M = 2                       # Number of line charges
    r = np.array([[0.5*L, 0.55*L], [0.5*L, 0.45*L]])
    q = 1.                      # Charge density [C/m]
    for i in range(M):
        ii = int(r[i, 0]/h + 0.5)
        jj = int(r[i, 1]/h + 0.5)  # Place charges at nearest grid point
        rho[ii, jj] += q/h**2

    # Compute the matrix P
    cx = np.cos((2*np.pi/N)*np.arange(N))
    cy = np.copy(cx)
    numerator = -h**2/(2.*eps0)
    tinyNumber = 1e-10          # Prevents division by zero
    P = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = numerator/(cx[i]+cy[j]-2. + tinyNumber)


    # Compute potential with the MFT method
    rhoT = np.fft.fft2(rho)      # Transform rho into wavenumber domain
    phiT = rhoT * P              # Computing phi in the wavenumber domain
    phi = np.fft.ifft2(phiT)     # Inverse transform into the coordinate domain
    phi = np.real(phi)           # Clean up imaginary part due to round-ff

    # Compute electric field as E = -grad phi
    [Ex, Ey] = np.gradient(np.flipud(np.rot90(phi)))
    # for i in range(N):
    #     for j in range(N):
    #         magnitude = np.sqrt(Ex[i, j]**2 + Ey[i, j]**2)
    #         Ex[i, j] = Ex[i, j] / -magnitude        # Normalizing components
    #         Ey[i, j] = Ey[i, j] / -magnitude

    # Plot potential
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Xp, Yp = np.meshgrid(x, y)
    ax.contour(Xp, Yp, np.flipud(np.rot90(phi)), 35)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\Phi(x,y)$')
    plt.show()

    # Plot electric field
    plt.quiver(Xp, Yp, Ey, Ex)
    plt.title('E field (Direction)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.axis([0., L, 0., L])
    plt.show()

if __name__ == '__main__':
    main()

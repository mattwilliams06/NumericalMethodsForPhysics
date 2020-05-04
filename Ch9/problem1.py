def main():
    ''' Evaluate the Laplace equation finite sum solution '''
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    n_tot = 51                     # Number of terms to use in sum
    n = np.arange(1, n_tot+1, 2)   # Odd-terms only
    N = 50                         # Grid size
    L = 1.                         # System length
    h = L / (N-1)                  # Grid spacing
    phi0 = 1.

    x = np.arange(N)*h
    y = np.arange(N)*h

    phi = np.empty((N, N))
    phi_n = np.empty((len(n), N, N))
    ## MAIN LOOP ##
    for idx, n_ in enumerate(n):
        for i in range(N):
            for j in range(N):
                phi[i ,j] = phi0 * 4./(np.pi*n_) * np.sin(n_*np.pi*x[i]/L) * \
                            (np.sinh(n_*np.pi*y[j]/L)/np.sinh(n_*np.pi))
        # Store each result along axis 0 in the rank-3 tensor phi_n
        phi_n[idx, :, :] = np.copy(phi)

    phi_tot = np.sum(phi_n, axis = 0)

    levels = np.linspace(0, 1, 11)
    xx, yy = np.meshgrid(x, y)
    # In meshgrid, xx is the columns and yy are the rows, which is opposite of phi_tot
    ct = plt.contour(x, y, np.flipud(np.rot90(phi_tot)), levels)
    plt.clabel(ct, fmt='%1.2f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Potential of finite sum with n = {n_tot}')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(yy, xx, phi_tot, rstride=1, cstride=1, cmap=cm.hot)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\Phi(x, y)$')
    plt.show()

if __name__ == '__main__':
    main()
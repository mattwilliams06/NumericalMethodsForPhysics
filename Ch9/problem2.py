def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    n_tot = 51  # Number of terms to use in sum
    n = np.arange(1, n_tot + 1)  # For use in series
    N = 50  # Grid size
    L = 1.  # System length
    h = L / (N - 1)  # Grid spacing
    phi1 = 1.
    phi2 = 0.
    phi3 = 1.
    phi4 = 0.

    x = np.arange(N) * h
    y = np.arange(N) * h

    phi = np.empty((N, N))
    phi_n = np.empty((len(n), N, N))
    for idx, n_ in enumerate(n):
        ks = 1j*(2*np.pi*n_+np.pi+1j*np.arcsinh(1))/L
        k = (4*np.pi*n_ + np.pi)/(2*L)
        for i in range(N):
            for j_ in range(N):
                X = phi2*np.sin(k*x[i]) + phi1*np.cos(k*x[i])
                Y = (phi4 + np.sqrt(2)*phi3)*np.sinh(ks*y[j_]) + phi3*np.cosh(ks*y[j_])
                phi[i, j_] = X * Y.real
        phi_n[idx, :, :] = np.copy(phi)

    phi_tot = np.sum(phi_n, axis=0)
    # phi_real = phi_tot.real
    # phi_imag = phi_tot.imag

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

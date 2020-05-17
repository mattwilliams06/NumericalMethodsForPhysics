def main():
    ''' Crank-Nicolson scheme for solving the Schrodinger equation'''
    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize parameters
    i_imag = 1j               # Imaginary number i
    N = 60                    # Number of grid points
    L = 100.                  # System size
    h = L/(N-1)               # Grid spacing
    x = np.arange(N)*h - L/2  # Grid points, -L/2 to L/2
    h_bar = 1.                # Planck
    mass = 1.
    tau = 0.01                # Time step

    # Set up Hamiltonian operator matrix
    ham = np.zeros((N, N))
    coeff = -h_bar**2/(2*mass*h**2)
    for i in range(1, N-1):
        ham[i, i-1] = coeff
        ham[i, i] = -2*coeff
        ham[i, i+1] = coeff

    # First and last rows for periodic boundary conditions
    ham[0, -1] = coeff
    ham[0, 0] = -2*coeff
    ham[0, 1] = coeff
    ham[-1, -2] = coeff
    ham[-1, -1] = -2*coeff
    ham[-1, 0] = coeff

    # Dirichlet boundary conditions
    # ham[0, 0] = 0
    # ham[-1, -1] = 0

    # Compute the Crank-Nicolson matrix
    dCN = np.dot(np.linalg.inv(np.identity(N)+0.5*i_imag*tau/h_bar*ham),
                 (np.identity(N) - 0.5*i_imag*tau/h_bar*ham))

    # Initialize the wave function
    x0 = 0.                     # Wave packet center location
    velocity = 0.5              # Average velocity
    k0 = mass*velocity/h_bar    # Average wavenumber
    sigma0 = L/10               # Standard deviation of the wavefunction
    norm_psi = 1./(np.sqrt(sigma0*np.sqrt((np.pi))))
    psi = np.empty(N, dtype=complex)
    for i in range(N):
        psi[i] = norm_psi * np.exp(i_imag*k0*x[i]) * np.exp(-(x[i] - x0)**2/(2*sigma0**2))

    # Plot the initial wavefunction
    plt.plot(x, np.real(psi), '-', x, np.imag(psi), '--')
    plt.xlabel('x')
    plt.ylabel(r'$\psi(x)$')
    plt.legend(['Real', 'Imag'])
    plt.title('Initial wave function')
    plt.show()

    # Initialize loop variables
    max_iter = int(L/(velocity*tau) + 0.5)
    plot_iter = max_iter/8
    p_plot = np.empty((N, max_iter+1))
    e_plot = np.empty(max_iter, dtype=complex)
    p_plot[:, 0] = np.absolute(psi[:])**2
    iplot = 0
    axisV = [-L/2., L/2., 0., max(p_plot[:, 0])]
    # Need velocity to compute momentum
    momentum = np.empty((max_iter, N))
    psi_vec = np.empty((max_iter+1, N))
    psi_vec[0, :] = np.real(psi)
    # Loop over desired number of steps (wave circles system once)
    for iter in range(max_iter):

        # Compute new wave function
        psi = np.dot(dCN, psi)
        # Dirichlet conditions
        psi[0] = 0
        psi[-1] = 0
        # Compute velocity
        psi_vec[iter+1, :] = np.real(psi)
        vel = (psi_vec[iter+1] - psi_vec[iter])/tau
        momentum[iter, :] = mass * vel

        # Compute the energy of the wave
        energy_num = np.dot(np.conj(psi), np.dot(dCN, psi))
        energy_denom = np.dot(np.conj(psi), psi)
        energy = energy_num/energy_denom
        e_plot[iter] = energy

        # Periodically record values for plotting
        if (iter+1) % plot_iter < 1:
            iplot+=1
            p_plot[:, iplot] = np.absolute(psi[:])**2
            plt.plot(x, p_plot[:, iplot])
            plt.xlabel('x')
            plt.ylabel('P(t, x)')
            plt.title(f'Finished {iter} of {max_iter} iterations')
            plt.axis(axisV)
            plt.show()

    # Plot probability density
    pFinal = np.empty(N)
    pFinal = np.absolute(psi[:])**2
    for i in range(iplot+1):
        plt.plot(x, p_plot[:, i])
    plt.xlabel('x')
    plt.ylabel('Probability density at various times')
    plt.show()

    # Plot energy
    plt.plot(np.arange(1, len(e_plot)+1)*tau, np.real(e_plot), np.arange(1, len(e_plot)+1)*tau, np.imag(e_plot), '--')
    plt.xlabel('Time (sec)')
    plt.ylabel('Energy')
    plt.legend(['Real', 'Imaginary'])
    plt.show()

    # Plot momentum
    plt.plot(x, momentum[-1, :])
    plt.xlabel('Position')
    plt.ylabel('Momentum')
    plt.show()
if __name__ == '__main__':
    main()
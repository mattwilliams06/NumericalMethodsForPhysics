import numpy as np
import matplotlib.pyplot as plt
from nm4p import rk4, rka

# The ratio B / (1 + A^2) is the effective damping of the system.
# When the ratio is 1, the system is critically damped, and slowly
# oscillates and converges to equilibrium
# When < 1, th system is overdamped, and quickly converges to equilibrium.
# When > 1, it is underdamped, and slowly converges, or never does if the ratio is
# too small

def derivs_bruss(s, t, params):
	''' Compute the RHS of the Belousov-Zhabotinski ODEs '''
	A, B = params
	x, y = s
	derivs = np.empty(len(s))
	derivs[0] = A + x**2 * y - (B + 1) * x
	derivs[1] = B * x - x**2 * y

	return derivs

def main(s, params):
	state = s
	tau = 0.1
	err = 1e-3
	time = 0.

	nsteps = 250
	xplot = np.empty(nsteps)
	yplot = np.empty(nsteps)
	tplot = np.empty(nsteps)

	A, B = params
	x_ss = A
	y_ss = B/A

	for i in range(nsteps):
		x, y = state
		xplot[i] = x
		yplot[i] = y
		tplot[i] = time

		[state, time, tau] = rka(state, time, tau, err, derivs_bruss, params)

		time += tau

	plt.figure()
	plt.plot(xplot, yplot)
	plt.plot(x_ss, y_ss, 'g*', label='Steady state concetrations')
	plt.xlabel('Concentration of species x')
	plt.ylabel('Concentration of species y')
	plt.title('Phase plot of chemical concetrations')
	plt.legend()

	plt.figure()
	plt.plot(tplot, xplot, 'b--', label='Concentration of species x')
	plt.hlines(x_ss, tplot[0], tplot[-1], 'g', label='Steady state concetration')
	plt.xlabel('time')
	plt.ylabel('Concentration of species x')
	plt.legend()

	plt.figure()
	plt.plot(tplot, yplot, 'b--', label='Concentration of species y')
	plt.hlines(y_ss, tplot[0], tplot[-1], 'g', label='Steady state')
	plt.xlabel('time')
	plt.ylabel('Concentration of species y')
	plt.legend()

	plt.show()

if __name__ == '__main__':
	params = np.array([5., 24])
	state_init = np.array([3., 3.])
	main(state_init, params)



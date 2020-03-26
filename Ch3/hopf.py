import numpy as np
import matplotlib.pyplot as plt

from nm4p import rka, rk4

def hopf_derivs(s, t, params):
	''' Return RHS of Hopf model
	Inputs
	------
	s = current state vector [x, y]
	t = independent variable (usually time, not used here)
	params = Hopf parameters [a]

	Output
	------
	derivs = derivatives of the state vector [dx/dt, dy/dt]
	'''

	x, y = s
	a = params

	derivs = np.empty(len(s))
	derivs[0] = a*x + y - x*(x**2 + y**2)
	derivs[1] = -x + a*y - y*(x**2 + y**2)

	return derivs

def main(init, param):
	state = init
	a = param
	tau = 0.1
	err = 1e-3

	nsteps = 100
	time = 0.
	tplot = np.empty(nsteps)
	xplot = np.empty(nsteps)
	yplot = np.empty(nsteps)
	tauplot = np.empty(nsteps)

	for i in range(nsteps):
		x, y = state
		tplot[i] = time
		xplot[i] = x
		yplot[i] = y
		tauplot[i] = tau
		if (i+1) % 50 == 0:
			print(f'Completed step {i+1} of {nsteps}')

		[state, time, tau] = rka(state, time, tau, err, hopf_derivs, a)
	tau_max = np.max(tauplot[1:])
	tau_min = np.min(tauplot[1:])
	if a > 0:
		print(f'Trajectories circle around origin with radius {np.sqrt(a):.3f}')
	print(f'Max and min adaptive time step values: {tau_max}, {tau_min}')
	plt.figure(1)
	plt.plot(xplot, yplot, 'b-')
	plt.plot(xplot[0], yplot[0], 'g*', label='starting position')
	plt.plot(0., 0., 'r*', label='origin')
	if a > 0:
		plt.vlines(0., 0., np.sqrt(a), label='radius')
	plt.xlabel('x-position')
	plt.ylabel('y-position')
	plt.title(f'Hopf model solution, a = {a}')
	plt.legend()

	plt.figure(2)
	plt.plot(tplot, xplot)
	plt.xlabel('time')
	plt.ylabel('x-position')
	plt.show()


if __name__ == '__main__':
	a = .03
	init = np.array([0., -1.])
	main(init, a)

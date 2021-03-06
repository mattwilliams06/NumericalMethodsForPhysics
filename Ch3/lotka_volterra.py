import numpy as np
import matplotlib.pyplot as plt

from nm4p import rk4, rka

def lkderivs(s, t, param):
	'''
    Returns the RHS of the Lotka-Volterra ODEs
    Inputs
    ------
    s = current state vector [x, y]
    t = independent variable (time, not used here)
    param = parameter vector [a, b, c, d, e]

    Outputs
    -------
    derivs = derivatives of the state vector [dx/dt, dy/dt]
	'''
	# unpack state and parameter vectors
	x, y = s
	a, b, c, d, e = params

	derivs = np.empty(len(s))
	derivs[0] = (a - b*x - c*y) * x
	derivs[1] = (-d + e*x) * y

	return derivs

def main(pop_init, param):
	state = pop_init
	params = param
	tau = 0.1
	err = 1e-3

	nsteps = 10000
	time = 0.
	tplot = np.empty(nsteps)
	xplot = np.empty(nsteps)
	yplot = np.empty(nsteps)
	tauplot = np.empty(nsteps)

	for i in range(nsteps):
		x, y = state[0], state[1]
		tplot[i] = time
		xplot[i] = x
		yplot[i] = y
		tauplot[i] = tau
		[state, time, tau] = rka(state, time, tau, err, lkderivs, params)

	# compute the time-averaged values of x and y
	x_avg = np.sum(xplot) / len(xplot)
	y_avg = np.sum(yplot) / len(yplot)

	# calculate the steady-state values
	x_ss = params[3]/params[4]
	y_ss = (params[0]*params[4] - params[1]*params[3]) / (params[2]*params[4])

	a, b, c, d, e = params
	tau_max = np.max(tauplot[1:])
	tau_min = np.min(tauplot[1:])
	print(f'Adaptive step RK time steps: max = {tau_max}, min = {tau_min}')
	plt.figure(1)
	plt.plot(xplot, yplot, '-')
	plt.plot(xplot[0], yplot[0], 'g*', label='Initial populations')
	plt.xlabel('Prey population')
	plt.ylabel('Predator population')
	plt.title(f'Lotka-Volterra solution, a = {a}, b = {b}, c = {c}, d = {d}, e = {e}')
	plt.legend()

	plt.figure(2)
	plt.plot(tplot, xplot, 'g-', label='prey population')
	plt.plot(tplot, yplot, 'r-', label='predator population')
	plt.xlabel('time')
	plt.ylabel('population')
	plt.legend()

	print(f'Time-averaged values: prey = {x_avg}, predator = {y_avg}')
	print(f'Steady-state values: prey = {x_ss}, predator = {y_ss}')
	plt.show()
	plt.close()

if __name__ == '__main__':
	pop_init = np.array([100, 97])
	params = np.array([10, 1e-5, 0.1, 10, 0.1])
	x_ss = params[3]/params[4]
	y_ss = (params[1]*params[3] - params[0]*params[4]) / (params[2]*params[4])
	main(pop_init, params)









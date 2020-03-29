import numpy as np

def derivsHO(state, t, tau, params):
	''' Returns the RHS of the coupled harmonic oscillator differential equation.
	State vector input = [x1, x2, x3, v1, v2, v3]
	Return: [v1, v2, v3, a1, a2, a3] where ai is the acceleration of the
	ith mass in the system.
	'''
	x1, x2, x3, v1, v2, v3 = [state[i] for i in range(len(state))]
	k1, k2, k3, k4, L1, L2, L3, L4, Lw, m1, m2, m3 = [params[i] for i in range(len(params))]

	derivs = np.empty(len(state))
	derivs[3] = ((-k1 - k2)*x1 + k2*x2 + k1*L1 - k2*L2)/m1
	derivs[4] = (k2*x1 + (-k2 - k3)*x2 + k2*x3 + k2*L2 - k3*L3)/m2
	derivs[5] = (k3*x2 + (-k3 - k4)*x3 + k3*L3 + (Lw - L4)*k4)/m3
	derivs[0] = v1 + derivs[3] * tau
	derivs[1] = v2 + derivs[4] * tau
	derivs[2] = v3 + derivs[5] * tau

	return derivs

def rk4(x, t, tau, params):
	half_tau = 0.5 * tau
	F1 = derivsHO(x, t + tau, tau, params)
	xtemp = x + half_tau * F1
	F2 = derivsHO(xtemp, t + half_tau, tau, params)
	xtemp = x + half_tau * F2
	F3 = derivsHO(xtemp, t + half_tau, tau, params)
	xtemp = x + tau * F3
	F4 = derivsHO(xtemp, t + tau, tau, params)
	xout = tau/6. * (F1 + 2*F2 + 2*F3 + F4)

	return xout
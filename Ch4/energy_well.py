# A program to find the eigenvalues of a potential energy well

import numpy as np

def feig(E, V, a, h, m):
	''' Returns the potential energy function evaluation and the derivatives

	'''

	if m % 2 == 0:
		f = -np.sqrt(-E) + np.sqrt(E-V) * np.tan(a/h * np.sqrt(2*m*(E-V)))
		df = np.tan(np.sqrt(2)*(a/h)*np.sqrt(m*(E-V)))/(2*np.sqrt(E-V)) + \
		(a/h)*m*np.sqrt(E-V)*np.cos((a/h)*np.sqrt(2)*np.sqrt(m*(E-V)))**(-2)/(np.sqrt(2)*np.sqrt(m*(E-V))) + \
		1./(2*np.sqrt(-E))

	else:
		f = -np.sqrt(-E) - np.sqrt(E-V) * 1. / np.tan(a/h * np.sqrt(2*m*(E-V)))
		df = -(np.tan(np.sqrt(2)*(a/h)*np.sqrt(m*(E-V))))**(-1)/(2*np.sqrt(E-V)) + \
		(a/h)*m*np.sqrt(E-V)*np.sin((a/h)*np.sqrt(2)*np.sqrt(m*(E-V)))**(-2)/(np.sqrt(2)*np.sqrt(m*(E-V))) + \
		1./(2*np.sqrt(-E))

	return f, df

def main(E0):
	nsteps = 10
	# Initialize parameters
	V = -13.6                # eV
	a0 = 5.29177210903e-11   # m (Bohr radius)
	a = 20 * a0
	M = np.arange(1, 21)     # Energy state number
	h = 6.62607015e-34       # J-s (Planck's constant)
	E_soln = np.empty(len(M))
	for m in M:
		E = E0
		for i in range(nsteps):
			F, DF = feig(E, V, a, h, m)
			E = E - F/DF
		E_soln[i] = E

	return E_soln

if __name__ == '__main__':
	E = main(-10.)
	print(E)
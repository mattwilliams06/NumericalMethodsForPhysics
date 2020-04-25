import numpy as np
import matplotlib.pyplot as plt

'''Use Newton's method to solve for the positions of the mass for the V-spring problem '''

def fnewt(x, a):
	''' Function to return the force vector and the Jacobian of the force vector

	Inputs
	------
	x: State vector [x, y]
	a: Parameter vector [m g k1 k2 L1 L2 D]

	Outputs
	-------

	f: Force vector RHS
	D: Jacobian matrix D(i, j) = df(j)/dx(i)
	'''

	# Evaluate f(i)
	f = np.empty(2)
	m, g, k1, k2, L1, L2, d = a
	f[0] = k1*x[0] *(1 - L1/np.sqrt(x[0]**2 + x[1]**2)) + k2*(x[0] - d) * (1 - L2/np.sqrt((x[0]-d)**2 + x[1]**2))

	f[1] = k1*x[1] *(1 - L1/np.sqrt(x[0]**2 + x[1]**2)) + k2*x[1] * (1 - L2/np.sqrt((d-x[0])**2 + x[1]**2)) - m*g

	# Evaluate D(i, j)
	D = np.empty((2, 2))
	D[0, 0] = k2*(1 - L2/np.sqrt((x[0]-d)**2 + x[1]**2)) + k2*L2*(x[0]-d)**2/((x[0]-d)**2 + x[1]**2)**(3/2) \
	+ k1*L1*x[0]**2/(x[0]**2 + x[1]**2)**(3/2) + k1*(1 - L1/np.sqrt(x[0]**2 + x[1]**2))

	D[1, 0] = k2*L2*x[1]*(x[0]-d)/((x[0]-d)**2 + x[1]**2)**(3/2) + k1*L1*x[0]*x[1]/(x[0]**2 + x[1]**2)**(3/2)

	D[0, 1] = k1*L1*x[0]*x[1]/(x[0]**2+x[1]**2)**(3/2) - k2*L2*x[1]*(d-x[0])/((d-x[0])**2+x[1]**2)**(3/2)

	D[1, 1] = k2*L2*x[1]**2/((d-x[0])**2+x[1]**2)**(3/2) + k2*(1 - L2/np.sqrt((d-x[0])**2 + x[1]**2)) \
	+ k1*L1*x[1]**2/(x[0]**2 + x[1]**2)**(3/2) + k1*(1 - L1/np.sqrt(x[0]**2 + x[1]**2))

	return [f, D]

def main():
	x0 = np.array([2.5, 1.75]) # Initial guess
	x = np.copy(x0)
	k1 = 10    # N/m
	k2 = 20    # N/m
	L1 = 0.1   # m
	L2 = 0.1   # m
	D = 0.1    # m
	m = 0.1    # kg
	g = 9.81   # m/s^2
	a = [m, g, k1, k2, L1, L2, D]

	nsteps = 10

	xp = np.empty((len(x), nsteps+1))
	xp[:, 0] = np.copy(x[:])

	for i in range(nsteps):
		[f, D] = fnewt(x, a)
		dx = np.linalg.solve(D.T, f)
		x = x - dx
		xp[:, i+1] = np.copy(x[:])
	return xp

if __name__ == '__main__':
	xp = main()
	print(f'Final position: {xp[:, -1]}')
	plt.plot(range(len(xp[0])), xp[0, :], label='x-position')
	plt.plot(range(len(xp[1])), xp[1, :], label='y-position')
	plt.xlabel('iteration')
	plt.ylabel('position')
	plt.legend()
	plt.show()




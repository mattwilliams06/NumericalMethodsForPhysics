# A program to solve a system of the Lorenz nonlinear equations using Newton's method. 
# Equations are defined by function fnewt

import numpy as np
import matplotlib.pyplot as plt

def fnewt(x, a):
	''' Function used by the N-variable Newton's method

	Inputs
	------
	x       State vector [x, y, z]
	a       Parameters [r sigma b]
	Outputs
	-------
	f       Lorenz model RHS [dx/dt dy/dt dz/dt]
	D       Jacobian matrix, D(i, j) = df(j)/dx(i)

	'''

	# Evaluate f(i)
	f = np.empty(3)
	f[0] = a[1] * (x[1] - x[0])
	f[1] = x[0] * (a[0] - x[2]) - x[1]
	f[2] = x[0] * x[1] - a[2] * x[2]

	# Evaluate D(i, j)
	D = np.empty((3,3))
	D[0, 0] = -a[1]         # df(0)/dx(0)
	D[0, 1] = a[0] - x[2]   # df(1)/dx(0)
	D[0, 2] = x[1]          # df(2)/dx(0)
	D[1, 0] = a[1]          # df(0)/dx(1)
	D[1, 1] = -1.           # df(1)/dx(1)
	D[1, 2] = x[0]          # df(2)/dx(1)
	D[2, 0] = 0.            # df(0)/dx(2)
	D[2, 1] = -x[0]         # df(1)/dx(2)
	D[2, 2] = -a[2]         # df(2)/dx(2)

	return [f, D]

# Set initial guess and parameters
x0 = np.empty(3)
x_vec = ['x', 'y', 'z']
a_vec = ['r', 'sigma', 'beta']
a = np.empty(3)
for i, vec in enumerate(x_vec):
	x0[i] = float(input(f'Enter initial {vec}: '))
for i, vec in enumerate(a_vec):
	a[i] = float(input(f'Enter initial {vec}: '))
#x0 = np.array([5., 5., 5.])
x = np.copy(x0)
#a = np.array([28., 10., 8/3])
# Loop over desired number of steps
nsteps = 10
xp = np.empty((len(x), nsteps+1))
xp[:, 0] = np.copy(x[:])    # record initial guess for plotting

for i in range(1, nsteps+1):
	# Evaluate the function f and its Jacobian D
	[f, D] = fnewt(x, a)
	dx = np.linalg.solve(np.transpose(D), f)

	# Update the estimate for the root
	x = x - dx
	xp[:, i] = np.copy(x[:])

# Print final extimate for the root
print(f'After {nsteps} iterations, the root is ', x)

# Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xp[0, :], xp[1, :], xp[2, :], '*-')
ax.plot([x[0]], [x[1]], [x[2]], 'ro')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Steady state of the Lorenz model')
plt.show()





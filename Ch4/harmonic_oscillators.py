import numpy as np
import matplotlib.pyplot as plt
from nm4p import rk4

### Solve the coupled harmonic oscillator problem

# Spring constants
k1 = 1.
k2 = 1.
k3 = 1.
k4 = 0.

# Spring rest lengths
L1 = 2.
L2 = 2.
L3 = 1.
L4 = 1.

# Total length of system
Lw = 4.

K = np.array([[-k1 - k2, k2, 0.],
			   [k2, -k2 - k3, k3],
			   [0., k3, -k3 - k4]])

b = np.array([[-k1 * L1 + k2 * L2],
			  [-k2 * L2 + k3 * L3],
			  [-k3 * L3 - (Lw - L4) * k4]])

# Finding the rest lengths means setting F = 0

F = np.array([[0.], [0.], [0.]])

LHS = F + b
disp = np.linalg.solve(K, LHS)
print(disp)

# Initial system conditions
x1_0 = disp[0] + 0.5
x2_0 = disp[1] + 0.
x3_0 = disp[2] + 0.
v1_0 = 0.
v2_0 = 0.
v3_0 = 0.
m1 = 0.1
m2 = 0.1
m3 = 0.1

# Other system parameters
tau = 0.05
time = 0.
nsteps = 100
params = np.array([k1, k2, k3, k4, L1, L2, L3, L4, Lw, m1, m2, m3])
state = np.array([x1_0, x2_0, x3_0, v1_0, v2_0, v3_0])
xplot = np.empty((3, nsteps))
vplot = np.empty((3, nsteps))
tplot = np.empty(nsteps)

for i in range(nsteps):
	xplot[0, i] = state[0]
	xplot[1, i] = state[1]
	xplot[2, i] = state[2]
	vplot[0, i] = state[3]
	vplot[1, i] = state[4]
	vplot[2, i] = state[5]
	tplot[i] = time
	state = rk4(state, time, tau, params)
	time += tau

plt.figure(1)
plt.plot(tplot, xplot[0, :], 'b-', label='mass 1 position')
plt.plot(tplot, xplot[1, :], 'g--', label='mass 2 position')
plt.plot(tplot, xplot[2, :], 'r-.', label='mass 3 position')
plt.hlines(disp[0], tplot[0], tplot[-1], 'k', '--')
plt.hlines(disp[1], tplot[0], tplot[-1], 'k', '--')
plt.hlines(disp[2], tplot[0], tplot[-1], 'k', '--')
plt.xlabel('time (sec)')
plt.ylabel('mass position (m)')

plt.legend()

plt.figure(2)
plt.plot(tplot, vplot[0, :], 'b-', label='mass 1 velocity')
plt.plot(tplot, vplot[1, :], 'g--', label='mass 2 velocity')
plt.plot(tplot, vplot[2, :], 'r-.', label='mass 3 velocity')
plt.xlabel('time (sec)')
plt.ylabel('mass velocity (m/s)')
plt.legend()

plt.show()






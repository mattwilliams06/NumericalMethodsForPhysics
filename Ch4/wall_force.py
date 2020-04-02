import numpy as np
import matplotlib.pyplot as plt

### Solve for the force on the wall

# Spring constants
k1 = 1.
k2 = 1.
k3 = 1.
k4 = 1.

# Spring rest lengths
L1 = 3.
L2 = 2.
L3 = 1.
L4 = np.arange(0.5, 6, 0.1)

# Total length of system
Lw = 10.
Frw = np.empty(len(L4))
X = np.empty((len(L4), 3))
for i, L in enumerate(L4):
	K = np.array([[-k1 - k2, k2, 0.],
				   [k2, -k2 - k3, k3],
				   [0., k3, -k3 - k4]])

	b = np.array([[-k1 * L1 + k2 * L2],
				  [-k2 * L2 + k3 * L3],
				  [-k3 * L3 - (Lw - L) * k4]])

	# Finding the rest lengths means setting F = 0

	F = np.array([[0.], [0.], [0.]])

	LHS = F + b
	Z = np.linalg.solve(K, LHS)
	#print(f'Z shape: {Z.shape}')
	#print(f'X shape: {X.shape}')
	#print(f'X[i] shape: {X[i].shape}')
	X[i] = np.array(Z.reshape(3,))
	x1, x2, x3 = X[i, :]
	print(f'Rest positions:\n\tx_1: {x1}\n\tx_2: {x2}\n\tx_3: {x3}')

	# Force on the right wall:
	Frw[i] = -k4 * (Lw - x3 - L)
	# print(f'Force on wall: {Frw} N')
#L4_min = L4[np.abs(Frw)==np.min(np.abs(Frw))]
L4_min = L4[np.argmin(np.abs(Frw))]
print(f'L4_min: {L4_min}')
X3_min = X[np.where(np.isclose(L4, L4_min)), 2]
print(f'x3 min: {X3_min}')
Frw_min = np.min(np.abs(Frw))
print(f'Frw min: {Frw_min}')
Frw_calc = -k4 * (Lw - X3_min - L4_min)
print(f'Frw calc: {Frw_calc}')

plt.plot(L4, Frw)
plt.xlabel('Rest length of spring 4 (m)')
plt.ylabel('Force on the right wall (N)')
plt.title(f'Force on right wall vs L4 with L1 = {L1}, L2 = {L2}, L3 = {L3}, and Lw = {Lw}')
plt.show()

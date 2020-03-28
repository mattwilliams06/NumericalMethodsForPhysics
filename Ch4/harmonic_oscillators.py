import numpy as np
import matplotlib.pyplot as plt

### Solve the coupled harmonic oscillator problem

# Spring constants
k1 = 1
k2 = 1
k3 = 1
k4 = 0

# Spring rest lengths
L1 = 2
L2 = 2
L3 = 1
L4 = 1

# Total length of system
Lw = 4

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
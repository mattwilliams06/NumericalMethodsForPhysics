import numpy as np
import matplotlib.pyplot as plt

# Set up system parameters
k1 = 1# N/m
K2 = np.arange(.1, 1.5, .01)    # N/m
L = 1                           # m
X = np.empty((4, len(K2)))
F = np.zeros((4,1))
for idx, k2 in enumerate(K2):
	K = np.array([[k1 - k2, k1, k2, 0.], 
			  [k1, -2.*k1 - k2, k1, k2],
			  [k2, k1, -2*k1 - k2, k1],
			  [0., k2, k1, -k1 - k2]])

	b = np.array([[-k1*L - k2*L],
		          [(L - L)*k1 - k2*L],
		          [(L - L)*k1 + k2*L],
		          [k1*L - k2*L]]) 
	x = np.linalg.solve(K, F-b)
	X[:, idx] = x.reshape(4,)

ratio = k1/K2
sys_lens = np.sum(X, axis=0)

plt.plot(ratio, sys_lens, marker='^')
plt.plot(ratio[np.argmin(sys_lens)], sys_lens.min(), 'r*', label='minimum length')
#plt.vlines(ratio[np.argmin(sys_lens)], bot, sys_lens.min(), colors='red')
plt.xlabel('Ratio k1/k2')
plt.ylabel('System length (m)')
plt.legend()
plt.show()
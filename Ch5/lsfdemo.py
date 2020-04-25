# Demo program for running least-squares fit
import numpy as np
import matplotlib.pyplot as plt
from linreg import linreg
from pollsf import pollsf

# To use linear instead of quadratic data, set c[2] = 0
c = np.array([2.0, 0.5, .02])
N = 50
x = np.arange(1, N+1)
y = np.empty(N)
alpha = 2.
sigma = alpha * np.ones(N)   # Constant error bar
np.random.seed(42)
for i in range(N):
	r = alpha * np.random.normal()
	#print(r)
	y[i] = c[0] + c[1] * x[i] + c[2] * x[i]**2 + r

fit = int(input('Enter degree of polynomial (1 is linear): '))
fit += 1

if fit == 2:
	[a_fit, sig_a, yy, chisqr] = linreg(x, y, sigma)
else:
	[a_fit, sig_a, yy, chisqr] = pollsf(x, y, sigma, fit)

# Print out the fit parameters and their error bars:
print('Fit parameters: ')
for i in range(fit):
	print(f'a[{i}] = {a_fit[i]} +/- {sig_a[i]}')

# Graph the data, with error bars, and fitting function
plt.errorbar(x, y, sigma, None, 'o')
plt.plot(x, yy, '-')
plt.xlabel(r'$x_i$')
plt.ylabel(r'$y_i$ and $Y(x)$')
plt.title(f'$Chi^2$ = {chisqr:.3f}, N-M = {N-fit}')
plt.show()
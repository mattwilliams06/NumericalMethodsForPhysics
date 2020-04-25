import numpy as np
import matplotlib.pyplot as plt
from linreg import linreg
from pollsf import pollsf

def main():
	x = np.arange(1, 6)
	y = np.array([2470., 2510., 2410., 2350., 2240.])

	# Degree of curve fit for plotting
	deg = np.arange(1, 5)
	alpha = 1.0
	N = len(x)
	sigma = alpha * np.ones(N)

	yy = np.empty((len(deg), N))
	a_fit_all = {}
	y_pred = np.zeros(len(deg))

	for M in deg:
		if M == 1:
			[a_fit, sig_a, yy[M-1, :], chisqr] = linreg(x, y, sigma)
		else:
			[a_fit, sig_a, yy[M-1, :], chisqr] = pollsf(x, y, sigma, M+1)
		a_fit_all.update({M: a_fit})
		for i in range(M+1):
			y_pred[M-1] += a_fit[i] * 6**i
	
	print(f'y_pred: {y_pred}')
	print(f'a_fit: {a_fit_all}')
	plt.plot(x, y, 'g*', label='Data')
	for i in range(len(deg)):
		plt.plot(x, yy[i, :], label=f'LSF line of degree {i+1}')
		plt.plot(6, y_pred[i], '*', label=f'Day 6 prediction, degree {i+1}')
	plt.legend()
	plt.show()

	

if __name__ == '__main__':
	main()

def pollsf(x, y, sigma, M):
	import numpy as np
	''' Function to perform a linear regression
	Inputs
	------
	x      Independent variable
	y      Dependent variable
	sigma  Estimated error in y
	M      Number of parameters used to fit data

	Outputs
	-------
	a_fit  Fit parameters; a(1) = intercept, a(2) = slope
	sig_a  Estimated error in the parameters a
	yy     Curve to fit the data
	chisqr Chi squared statistic
	'''

	# Form the vector b and design matrix A
	# The design matrix contains N rows (the number of points) and M
	# columns (the number of parameters for the regression)
	N = len(x)
	b = np.empty(N)
	A = np.empty((N, M))
	for i in range(N):
		b[i] = y[i] / sigma[i]
		for j in range(M):
			A[i, j] = x[i]**j / sigma[i]  # A is a type of Vandermonde matrix, Yj(xi) = x**j

	# Compute the correlation matrix C
	C = np.linalg.inv(np.dot(A.T, A))

	# Compute coefficients a_fit
	a_fit = np.dot(C, np.dot(A.T, b))

	# Compute error bars
	sig_a = np.empty(M)
	for j in range(M):
		sig_a[j] = np.sqrt(C[j, j])

	# Evaluate
	yy = np.zeros(N)
	chisqr = 0.
	for i in range(N):
		for j in range(M):
			yy[i] += a_fit[j] * x[i] ** j
		chisqr += ((y[i] - yy[i]) / sigma[i])**2

	return [a_fit, sig_a, yy, chisqr]
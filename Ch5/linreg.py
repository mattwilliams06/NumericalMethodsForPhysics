def linreg(x, y, sigma):
	''' Function to perform a linear regression
	Inputs
	------
	x      Independent variable
	y      Dependent variable
	sigma  Estimated error in y

	Outputs
	-------
	a_fit  Fit parameters; a(1) = intercept, a(2) = slope
	sig_a  Estimated error in the parameters a
	yy     Curve to fit the data
	chisqr Chi squared statistic
	'''

	# Evaluate sums for the generalized solution
	import numpy as np

	s = 0.; sx = 0.; sy = 0.; sxx = 0.; sxy = 0.
	N = len(x)
	for i in range(N):
		sigmaTerm = sigma[i]**(-2)
		s += sigmaTerm
		sx += x[i] * sigmaTerm
		sy += y[i] * sigmaTerm
		sxx += x[i]**2 * sigmaTerm
		sxy += x[i] * y[i] * sigmaTerm

	a_fit = np.empty(2)
	denom = s*sxx - sx**2
	a_fit[0] = (sy * sxx - sx * sxy) / denom
	a_fit[1] = (s * sxy - sx * sy) / denom

	# Compute error bars for the intercept and slope
	sig_a = np.empty(2)
	sig_a[0] = np.sqrt(sxx / denom)
	sig_a[1] = np.sqrt(s / denom)

	# Evaluate curve fit at each point and compute chi^2
	yy = np.empty(len(x))
	chisqr = 0.

	for i in range(len(x)):
		yy[i] = a_fit[0] + a_fit[1] * x[i]
		chisqr += ((y[i] - yy[i]) / sigma[i])**2

	return [a_fit, sig_a, yy, chisqr]





def sprrk(s, t, param):
	import numpy as np
	''' Returns RHS of spring-mass system equations of motion

	Inputs
	------
	s       State vector [x(1), x(2), ... v(3)]
	t       Time (not used)
	param   Spring constant / mass

	Output
	------
	deriv   [dx(1)/dt, dx(2)/dt, ... dv(3), dt]
	'''

	deriv = np.empty(6)
	deriv[0] = s[3]
	deriv[1] = s[4]
	deriv[2] = s[5]

	param2 = -2. * param
	deriv[3] = param2*s[0] + param*s[1]
	deriv[4] = param2*s[1] + param*(s[0]+s[2])
	deriv[5] = param2*s[2] + param*s[1]

	return deriv
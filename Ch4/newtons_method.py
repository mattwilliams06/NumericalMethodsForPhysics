import numpy as np
import matplotlib.pyplot as plt

''' This is a Newton's method root-finding algorithm, which is guaranteed to converge
for functions where the root does not lie close to a location where the derivative
is equal to zero.
'''
h = 1e-3
print(f'h = {h:.3f}')
def func(x):
	y = (-x**6 + 2)
	return y

def dfunc(h, xold, xnew, func):
	dy = func(xnew) - func(xold)
	dydx = dy/h
	return dydx

def newton(x0, func, deriv, tol=1e-3):
	xold = x0
	xnew = x0 + h
	xplot = [xold]
	f = func(xold)
	yplot = [f]
	i = 0
	max_iter = 10
	print('Entering main loop')
	while abs(f) > tol:
		df = dfunc(h, xold, xnew, func)
		xnew = xold - f/df
		print(xnew)
		xplot.append(xnew)
		f = func(xnew)
		yplot.append(f)
		xold = xnew
		xnew = xold + h
		i += 1
		if i > max_iter:
			print('ERROR: Max iterations achieved')
			break
		if abs(f) <= tol:
			print('Convergence obtained')
			print(f'Root: x = {xold:.3f}')
			# return xold
	if xplot[0] > xplot[-1]:
		xgraph = np.linspace(xplot[-1] - 1, xplot[0] + 1, 1000)
	else:
		xgraph = np.linspace(xplot[0] + 1, xplot[-1] - 1, 1000)
	ygraph = func(xgraph)
	plt.plot(xgraph, ygraph)
	plt.hlines(0., xgraph[-1], xgraph[0])
	for i in range(len(xplot)-1):
		plt.plot([xplot[i], xplot[i+1]], [yplot[i], 0.], 'r--')
		plt.plot(xplot[-1], 0., 'g*')
	plt.grid(True)
	plt.show()
	print(xplot)


if __name__ == '__main__':
	newton(1, func, dfunc)
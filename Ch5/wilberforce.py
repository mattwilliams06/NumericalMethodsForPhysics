''' Program to solve the Wilberforce pendulum and determine its normal modes using FFT and power spectra'''

import numpy as np
import matplotlib.pyplot as plt
from nm4p import rk4

# parameters
m = 0.5            # mass (kg)
I = 1e-4           # moment of inertia (kg-m^2)
k = 5              # spring constant (N/m)
delta = 1e-3       # torsional spring constant (N-m)
eps = 1e-2         # force coupling between modes (N)
params = np.array([m, I, k, delta, eps])

x0 = np.array([0.01, 5.*np.pi/180])   # Initial displacement vector
v0 = np.array([0.0, 0.0])            # Initial velocity vector

state = np.array([x0[0], x0[1], v0[0], v0[1]])
time = 0.
tau = 0.1
nsteps = 256
tplot = np.empty(nsteps)
xplot = np.empty((nsteps, len(state)))


def wilber_derivs(s, t, params):
	''' Returns the derivative of the state vector '''
	m, I, k, delta, eps = params
	derivs = np.empty(len(s))
	derivs[0] = s[2]
	derivs[1] = s[3]
	derivs[2] = -k/m*s[0] - eps/(2*m)*s[1]
	derivs[3] = -delta/I*s[1] - eps/(2*I)*s[0]

	return derivs

for i in range(nsteps):
	state = rk4(state, time, tau, wilber_derivs, params)
	time += tau
	xplot[i, :] = np.copy(state)
	tplot[i] = time

plt.figure()
plt.plot(tplot, xplot[:, 0], '-', label='Longitudinal Displacement')
plt.xlabel('Time')
plt.ylabel('Longitudinal Displacement (m)')
plt.figure()
plt.plot(tplot, xplot[:, 1]*180/(np.pi), '-', label='Torsional Displacement')
plt.xlabel('Time')
plt.ylabel('Torsional Displacement (degrees)')

f = np.arange(nsteps)/(tau*nsteps)
x1 = xplot[:, 0]
x1fft = np.fft.fft(x1)
spect = np.empty(len(x1fft))
for i in range(len(x1fft)):
	spect[i] = np.abs(x1fft[i])**2

# Apply the Hanning window to the time series
x1w = np.empty(len(x1))
for i in range(len(x1)):
	window = 0.5 * (1 - np.cos(2*np.pi*i/nsteps)) # Hanning window
	x1w[i] = x1[i] * window

x1wfft = np.fft.fft(x1w)
spectw = np.empty(len(x1wfft))
for i in range(len(x1wfft)):
	spectw[i] = np.abs(x1wfft[i])**2

# Graph the power spectra for original and windowed data
plt.figure()
plt.semilogy(f[0:int((nsteps/2))], spect[0:int((nsteps/2))], '-')
plt.semilogy(f[0:int((nsteps/2))], spectw[0:int((nsteps/2))], '--')
plt.title('Power spectrum (dashed is windowed data')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show()
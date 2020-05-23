# Program to compute the power spectrum of a coupled spring-mass system
import numpy as np
import matplotlib.pyplot as plt
from nm4p import rk4
from sprrk import sprrk

# Set system parameters
x = np.array([0.3, 0., 0.])    # Initial positions (m)
v = np.array([0., 0., 0.])     # Initial velocities (m/s)
state = np.array([x[0], x[1], x[2], v[0], v[1], v[2]])
tau = 0.5                      # Time step (sec)
k_over_m = 1.                  # Spring constant divided by mass (N/(kg-m))

time = 0.
nsteps = 256
nprint = nsteps/8               # Number of steps between printing process
tplot = np.empty(nsteps)
xplot = np.empty((nsteps, 3))

### MAIN LOOP ###
for i in range(nsteps):
	# Use Runge-Kutta to find new mass displacements
	state = rk4(state, time, tau, sprrk, k_over_m)
	time += tau

	# Record positions for plotting
	tplot[i] = time
	xplot[i, :] = np.copy(state[:3])
	if i % nprint < 1:
		print(f'Finished {i} out of {nsteps} steps')

# Graph displacements
plt.figure()
plt.plot(tplot, xplot[:, 0], '-', label='Mass 1')
plt.plot(tplot, xplot[:, 1], '-', label='Mass 2')
plt.plot(tplot, xplot[:, 2], '-', label='Mass 3')
plt.title('Displacement of masses relative to rest position')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()

# Calculate the power spectrum of the time series for mass 1
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

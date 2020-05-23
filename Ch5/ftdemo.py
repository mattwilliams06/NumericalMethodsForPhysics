# ftdemo - Discrete Fourier transform program

import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize the sine wave series to be analyzed
N = 50
freq = .2
phase = 0.
tau = 1.
t = np.arange(N) * tau
y = np.empty(N)
for i in range(N):
	y[i] = np.sin(2 * np.pi *t[i] * freq + phase)

f = np.arange(N) / (N* tau)
# Compute the transform using either discrete direct summation or FFT
yt = np.zeros(N, dtype=complex)
method = 1    # Set 1 for direct summation, 2 for FFT

start_time = time.time()
if method == 1:
	twoPiN = -2 * np.pi * (1j) / N    # Exponentiated term in the Fourier transform
	for k in range(N):
		for i in range(N):
			exp_term = np.exp(twoPiN*k*i)
			yt[k] += y[i] * exp_term

else:
	yt = np.fft.fft(y)

stop_time = time.time()

Pk = yt  * np.conj(yt)

print(f'Elapsed time: {stop_time - start_time} seconds')
ft_freq = f[np.argmax(np.abs(yt))]
print(f'Extracted frequencies: {1 - ft_freq:.1f} Hz, {ft_freq} Hz')
# Graph
print(f'Shapes of t: {t.shape}, y: {y.shape}')
plt.subplot(1, 3, 1)
plt.plot(t, y)
plt.title('Original time series')
plt.xlabel('time')

plt.subplot(1, 3, 2)
plt.plot(f, np.real(yt), '-', f, np.imag(yt), '--')
plt.legend(['Real', 'Imaginary'])
plt.title('Fourier transform')
plt.xlabel('Frequency')

plt.subplot(1, 3, 3)
plt.plot(f, Pk)
plt.title('Power sprectrum')
plt.xlabel('Frequency')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
# Initialize parameters
## Resistances (ohms)
Ra = 1e-3
R1 = 100.
R2 = np.logspace(0, 5, 5)
R3 = np.array([1., 100., 1e4])
R4 = R3

## Voltage (volts)
E = 6.

currents = np.empty((len(R3), 3, len(R2)))

for i3, (r3, r4) in enumerate(zip(R3, R4)):
	for i2, r2 in enumerate(R2):
		res_mat = np.array([[-R1, r2, Ra], 
					[r3, r4, Ra+r3-r4], 
					[R1+r3, 0., r3]])
		volt_mat = np.array([0., 0., E])
		soln = np.linalg.solve(res_mat, volt_mat)
		currents[i3][:, i2] = np.linalg.solve(res_mat, volt_mat)
	plt.figure(i3+1)
	plt.loglog(R2, currents[i3][2, :])
	plt.xlabel('Variable resistor value (ohms)')
	plt.ylabel('Ammeter current (amps)')
	plt.title(f'Ammeter current vs R2 resistance for R3 = R4 = {r3} ohms')
plt.show()		

# The result tensor 'currents' contains a 3 3x5 matrices, with each 3x5 matrix computed
# over the ranfe of R2 for a given R3
for idx, r3 in enumerate(R3):
	print(f'R3 = {r3}')
	print(currents[idx])
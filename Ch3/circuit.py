import numpy as np
import matplotlib.pyplot as plt
# Initialize parameters
## Resistances (ohms)
R1 = 1.
R2 = 1.
R3 = 2.
R4 = 2.
R5 = 5.
## Voltage sources (volts)
E1 = 2.
E3 = 5.
E2 = [float(volts) for volts in range(2, 21)]

resistance_matrix = np.array([[R1+R2+R3, -R3], [-R3, R3+R4+R5]])
#print(resistance_matrix)


currents = np.empty((3, len(E2)))
for i, volt2 in enumerate(E2):
	voltage_matrix = np.array([[E1-volt2], [volt2-E3]])
	currents[0, i], currents[2, i] = np.linalg.solve(resistance_matrix, voltage_matrix)
	currents[1, i] = currents[0, i] - currents[2, i]

# Power of resistor 5
P5 = currents[2, :]**2 * R5

plt.plot(E2, P5)
plt.xlabel('Voltage (volts)')
plt.ylabel('Power (watts)')
plt.title('Power of Resistor 5')
plt.show()
print(E2[int(np.argwhere(currents[2]==0))])
print(currents[:, 6])


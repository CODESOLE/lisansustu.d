import matplotlib.pyplot as plt
import random
import numpy as np

plt.style.use("dark_background")

step = 100
time = 10

deltaTs = np.linspace(0, time, step)
deltaXs = np.full(step, 0.1)
zetas = np.empty(step)
acc = 0

for i in range(step):
    acc += random.choice([-1,1])
    zetas[i] = acc

fig, ax = plt.subplots()

ax.plot(deltaTs, deltaXs * zetas, marker='', label='dx=0.1')

deltaXs1 = np.sqrt(deltaTs)
ax.plot(deltaTs, deltaXs1 * zetas, marker='', label='dx=sqrt(T)')

ax.set_xlabel('delta times')
ax.set_ylabel('Ax x zetas')
ax.set_title('Discrete Brownian')
ax.legend(loc='upper right')
ax.grid(True)
plt.show()

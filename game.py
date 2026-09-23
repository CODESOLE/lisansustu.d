import numpy as np

N, p = 1_000_000, 0.5
played_round = np.arange(N)
for sim in range(N):
    nloss, nround = 0, 0
    while nloss < 2:
        nround += 1
        if np.random.uniform() < p: # success
            nloss = 0
        else:
            nloss += 1
    played_round[sim] = nround
print(np.average(played_round))

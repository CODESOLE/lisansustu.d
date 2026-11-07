import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("dark_background")
arr = "aatgcctattc"
nuc = np.frombuffer(arr.encode("ascii"), dtype="S1")

fig, axes = plt.subplots(3, 3)
axx = axes.flat
for i in range(9):
    unique, counts = np.unique(nuc, return_counts=True)
    probs = counts / counts.sum()
    next_nuc = np.random.choice(unique, size=1, replace=True, p=probs)
    print()
    nuc = np.append(nuc, next_nuc)
    sns.histplot(x=nuc.astype(str), stat="probability", discrete=True, ax=axx[i])

plt.show()

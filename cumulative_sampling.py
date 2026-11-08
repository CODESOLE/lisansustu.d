import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

plt.style.use("dark_background")
rng = np.random.default_rng()

# Generate random normally distributed data
data = rng.normal(loc=0, scale=1, size=1000)
# Create the KDE plot
sns.kdeplot(data, fill=True, color="blue", bw_adjust=1.0, label="Seaborn")
# Customize layout
plt.xlabel("Values")
plt.ylabel("Density")

# Data
data = rng.normal(0, 1, 500)
# Calculate the mathematical density
x_eval = np.linspace(min(data) - 1, max(data) + 1, 200)
y_eval = stats.gaussian_kde(data)(x_eval)
# Plot using pure Matplotlib
plt.plot(x_eval, y_eval, color="red", label="SciPy KDE")
plt.fill_between(x_eval, y_eval, alpha=0.3, color="red")
plt.title("KDE using SciPy and Matplotlib or Seaborn")
plt.legend()
plt.show()

arr = "aatgcctattc"
nuc = np.frombuffer(arr.encode("ascii"), dtype="S1")

fig, axes = plt.subplots(3, 3)
axx = axes.flat
for i in range(9):
    unique, counts = np.unique(nuc, return_counts=True)
    probs = counts / counts.sum()
    next_nuc = rng.choice(unique, size=1, replace=True, p=probs)
    print()
    nuc = np.append(nuc, next_nuc)
    sns.histplot(x=nuc.astype(str), stat="probability", discrete=True, ax=axx[i], kde=True)

plt.show()

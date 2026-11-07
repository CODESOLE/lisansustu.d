import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

rng = np.random.default_rng()

mu = 4
sigma = 2
sample = rng.normal(loc=4, scale=2)
prev = sp.stats.norm.pdf(sample, loc=mu, scale=sigma)
monte = [sample]
for i in range(1_000):
    sample = rng.normal(loc=4, scale=2)
    l = sp.stats.norm.pdf(sample, loc=mu, scale=sigma)
    if l < prev:
        cond = rng.choice([1, 0], replace=True, p=[l/prev, 1 - l/prev])
        if cond == 1:
            monte.append(sample)
        else:
            continue
    else:
        monte.append(sample)
    prev = l
mcmc = np.array(monte)
plt.hist(mcmc, bins=10)
plt.title('Histogram Example')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

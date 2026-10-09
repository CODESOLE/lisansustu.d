import numpy as np
import matplotlib.pyplot as plt

def likelihood(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (np.sqrt(2*np.pi) * sigma)

mu = 4
sigma = 2
sample = np.random.normal(loc=4, scale=2)
prev = likelihood(sample, mu, sigma)
monte = [sample]
for i in range(1_000):
    sample = np.random.normal(loc=4, scale=2)
    l = likelihood(sample, mu, sigma)
    if l < prev:
        cond = np.random.choice([1, 0], replace=True, p=[l/prev, 1 - l/prev])
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

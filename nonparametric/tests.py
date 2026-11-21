from statsmodels.stats.diagnostic import lilliefors
from scipy import stats
from numpy import np

# Uniform example: 6 TV programs (slide: χ²=4.6, p=0.47)
obs = [5, 8, 10, 12, 12, 13]
stat, p = stats.chisquare(obs)          # f_exp defaults to equal frequencies
# χ² = 4.60, p = 0.4667 → H0 not rejected ✓

# Binomial-fit example (defective packages, p estimated from data)
n, p_hat = 5, 0.14
pr = [stats.binom.pmf(x, n, p_hat) for x in range(5)]
pr = [pr[0], pr[1], pr[2], 1 - sum(pr[:3])]   # merge tail: expected < 5
exp = [280 * q for q in pr]                    # [131.7, 107.2, 34.9, 6.2]
res = stats.chisquare([157, 69, 35, 19], exp, ddof=1)   # ddof=1: one param estimated → df = 4-1-1 = 2
# χ² ≈ 45.2, p ≈ 1.5e-10 → H0 rejected ✓

data = [...]  # your raw measurements
# H0: data ~ Normal(mu=2.5, sigma=0.3)
stat, p = stats.kstest(data, 'norm', args=(2.5, 0.3))

stat, p = lilliefors(data, dist='norm')   # parameters estimated from data

res = stats.shapiro(data)
# res.statistic = W, res.pvalue
if res.pvalue > 0.05: print("H0 not rejected: normality OK")

stat, p = stats.ks_2samp(x, y)                  # two-sided
stat, p = stats.ks_2samp(x, y, alternative='less')   # one-sided

# Seed germination: 6 of 9 germinated, H0: π ≥ 0.80 (slide: p = 0.2618)
res = stats.binomtest(6, 9, 0.80, alternative='less')

# Public aid: 6 of 20, H0: π ≤ 0.19 (slide: p = 0.1643)
res = stats.binomtest(6, 20, 0.19, alternative='greater')

# Tea packaging: 10 of 10, H0: π ≤ 0.95 (slide: p = 0.5987)
res = stats.binomtest(10, 10, 0.95, alternative='greater')

print(res.pvalue)
res.proportion_ci(0.95)   # bonus: exact CI for π

d = np.array([32, 65, 27, 34, 66, 40, 36, 52])   # recovery times
d = d[d != 30]                                    # discard ties vs M0

k = (d > 30).sum();  n = len(d)                   # k=7 positive signs, n=8
res = stats.binomtest(k, n, 0.5, alternative='greater')   # H1: M > 30, alternative=less, two-sided
# p = 0.03516 < 0.05 → reject H0 ✓ (matches slide)


"""
Here's a decision guide for all 7 tests — when to reach for each one, plus a sample question per test.

## Quick decision map

```
Question about...                          → Test
────────────────────────────────────────────────────────────
Do category counts match a distribution?   → 1. Chi-Square GOF
Does data follow a KNOWN distribution?     → 2. KS One-Sample
Normality, parameters unknown?             → 3. Lilliefors
Is data normal (general check)?            → 4. Shapiro-Wilk
Do two groups share the same distribution? → 5. KS Two-Sample
Is a proportion above/below a target?      → 6. Binomial Test
Is the median equal to a value?            → 7. Sign Test
```

---

### 1. Chi-Square Goodness of Fit
**When:** Your data are **counts in categories** (nominal/categorical scale). You want to know if those proportions match a hypothesized distribution (uniform, binomial, Poisson, or any custom distribution). Use raw category frequencies, not measurements.

- ✔ *Sample question:* "A café owner recorded which of 5 menu items were ordered by 200 customers. Do customers order the 5 items **equally often**?"
- *(From your notes: do 6 TV programs have equal viewership? Is the defect count per package binomial with p=0.14?)*

### 2. Kolmogorov–Smirnov One-Sample
**When:** You have **continuous or ordinal measurements** and want to compare the whole empirical distribution against a **fully specified** distribution — parameters (μ, σ, λ...) are **known in advance** from theory, a standard, or a historical population, not estimated from your sample.

- ✔ *Sample question:* "Bottles must weigh 500 g on average with a known process σ = 8 g (set by the machine calibration). Did today's sample of 25 bottles come from that specified N(500, 8) process?"
- *(From your notes: do chick weights come from N(2.5, 0.3)? Do kidney weights come from N(85, 15)?)*

### 3. Lilliefors Test
**When:** Same goal as KS one-sample (normality check) but **μ and σ are unknown** and estimated from the data — this is the common real-world case. Plain KS would give p-values that are too large; Lilliefors corrects for the estimation.

- ✔ *Sample question:* "Before running a t-test on my 15 lab measurements, I must confirm normality — but I don't know the true mean or variance of the population. Is my data normal?"
- *(From your notes: chick weights again, but with mean/variance estimated from the sample.)*

### 4. Shapiro–Wilk
**When:** The standard, most **powerful normality test** for small-to-moderate samples (n ≤ ~50, works to a few thousand). Use it whenever the only question is "is this data normal?" — it generally beats KS/Lilliefors when you can choose. It's the default pre-check before parametric tests (t-test, ANOVA).

- ✔ *Sample question:* "I want to use a t-test on 20 patients' blood-pressure reductions. Is the data normal enough for that?"
- *(From your notes: W = 0.96, p = 0.31 on the chick data.)*

### 5. KS Two-Sample
**When:** You have **two independent groups** of continuous/ordinal data and ask whether their **entire distributions differ** — sensitive to differences in location, spread, *or shape* (not just the median). Nonparametric alternative when you don't want to assume normality; use raw measurements, not counts.

- ✔ *Sample question:* "Do delivery times differ between Courier A and Courier B — in any way (average, variability, or shape of the distribution)?"
- *(From your notes: basal metabolism of athletes vs. non-athletes; do boys' and girls' exam scores come from different distributions?)*

### 6. Binomial Test
**When:** Each observation is a **yes/no outcome** (success/failure, pass/fail, defective/OK) over a **fixed number of independent trials**, and you test whether the **success proportion** differs from a claimed value π₀. The exact test for small samples where the z-approximation for a proportion isn't valid.

- ✔ *Sample question:* "A supplier claims ≥90% of delivered bulbs pass quality control. Of 12 randomly inspected bulbs, 9 passed. Is the supplier's claim supported at α = 0.05?"
- *(From your notes: does seed germination rate fall below 0.80? Is the tea-packaging compliance rate above 95%?)*

### 7. Sign Test
**When:** You test a **hypothesis about the median** of a continuous, at-least-ordinal variable when the population is **not normal / sample is small** — the nonparametric replacement for the one-sample t-test (which targets the mean). Only the *signs* of (Xᵢ − M₀) matter, so it's very robust but less powerful than Wilcoxon signed-rank. Also used for paired data when you only trust the direction of differences.

- ✔ *Sample question:* "A mindfulness course claims to reduce stress. For 8 participants I know whether stress went up or down after the course (not by how much). Is the median stress change different from zero?"
- *(From your notes: is the median recovery time on drug A greater than 30 days? Is the median filling lifetime below 100 months?)*

---

## One-line summary

| # | Test | Data type | Tests | Parametric counterpart |
|---|------|-----------|-------|------------------------|
| 1 | Chi-Square GOF | category counts | fit to any distribution | — |
| 2 | KS 1-sample | continuous (params known) | fit to specified distribution | — |
| 3 | Lilliefors | continuous (params estimated) | normality | — |
| 4 | Shapiro–Wilk | continuous | normality (best default) | — |
| 5 | KS 2-sample | 2 independent continuous groups | equal distributions | t-test / F-test (partial) |
| 6 | Binomial | yes/no trials | proportion = π₀ | z-test for proportion |
| 7 | Sign test | continuous/ordinal | median = M₀ | one-sample t-test |

Want me to append this usage guide as a module docstring / decision-table comment in your `nonparametric_tests.py`?
"""

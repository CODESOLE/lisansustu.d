import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

plt.style.use("dark_background")
rng = np.random.default_rng(48)

x = np.arange(0, 100)
y = 2 * x + 3 + rng.normal(0, 40, size=x.shape)

def sse(df, m, b, lamda):
    pred = m * df.iloc[:, 0] + b
    return np.sum(np.square(df.iloc[:, 1].to_numpy() - pred)) + (lamda * m**2)

m = np.linspace(-50, 50, 100)
b = np.linspace(-50, 50, 100)
df = pd.DataFrame({"x": x, "y": y})

ms = np.empty_like(m)
for i, mi in enumerate(m):
    ms[i] = sse(df, mi, 0, lamda=0)
opt_m = m[np.argmin(ms)]

bs = np.empty_like(b)
for i, bi in enumerate(b):
    bs[i] = sse(df, opt_m, bi, lamda=0)
opt_b = b[np.argmin(bs)]

print(f"opt_m: {opt_m}")
print(f"opt_b: {opt_b}")

# ================================================================

ridge_reg_ms = np.empty_like(m)
for i, mi in enumerate(m):
    ridge_reg_ms[i] = sse(df, mi, 0, lamda=100)
ridge_reg_opt_m = m[np.argmin(ridge_reg_ms)]

ridge_reg_bs = np.empty_like(b)
for i, bi in enumerate(b):
    ridge_reg_bs[i] = sse(df, ridge_reg_opt_m, bi, lamda=100)
ridge_reg_opt_b = b[np.argmin(ridge_reg_bs)]

print(f"ridge_reg_opt_m: {ridge_reg_opt_m}")
print(f"ridge_reg_opt_b: {ridge_reg_opt_b}")

fig, axes = plt.subplots(2, 2, figsize=(13, 5))

sb.scatterplot(data=df, x="x", y="y", ax=axes[0, 0])
sb.lineplot(x=x, y=opt_m*x+opt_b, color="red", label="optimized", ax=axes[0, 0])
axes[0, 0].set_title("data vs fitted_line")

sb.lineplot(x=m, y=ms, ax=axes[0, 1], label="slope_vs_sse")
sb.lineplot(x=b, y=bs, ax=axes[0, 1], label="intercept_vs_sse")
axes[0, 1].set_title("slope & intercept vs SSE")

# ================================================================

sb.scatterplot(data=df, x="x", y="y", ax=axes[1, 0])
sb.lineplot(x=x, y=ridge_reg_opt_m*x+ridge_reg_opt_b, color="red", label="optimized", ax=axes[1, 0])
axes[1, 0].set_title("Data vs Ridge Regressed Fitted Line")

sb.lineplot(x=m, y=ridge_reg_ms, ax=axes[1, 1], label="ridge_regressed_slope_vs_sse")
sb.lineplot(x=b, y=ridge_reg_bs, ax=axes[1, 1], label="ridge_regressed_intercept_vs_sse")
axes[1, 1].set_title("Slope & Intercept vs SSE")

fig.suptitle("Ridge Regression")
plt.tight_layout()
plt.show()

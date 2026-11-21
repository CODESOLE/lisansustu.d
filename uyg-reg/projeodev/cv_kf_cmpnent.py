import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
from scipy import stats

df = pd.read_csv('tez/datasets/tecator/tecator.csv')
X = df[[f"absorbance_{i}" for i in range(1, 101)]]
y = df["fat"].astype(float)

# 2. Set up 10-fold Cross-Validation and component range
max_components = 35
k_values = np.arange(1, max_components + 1)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

mean_mse = []
se_mse = []

# 3. Evaluate PCR for each number of components k
for k in k_values:
  pcr_pipeline = make_pipeline(StandardScaler(), PCA(n_components=k), LinearRegression())
  scores = -cross_val_score(pcr_pipeline, X, y, cv=kf, scoring="neg_mean_squared_error")
  mean_mse.append(np.mean(scores))
  se_mse.append(np.std(scores, ddof=1) / np.sqrt(len(scores)))

mean_mse = np.array(mean_mse)
se_mse = np.array(se_mse)

# 4. Find optimal k using Minimum MSE and the 1-Standard-Error (1-SE) Rule
min_idx = np.argmin(mean_mse)
best_k_min = k_values[min_idx]
threshold_1se = mean_mse[min_idx] + se_mse[min_idx]
best_k_1se = k_values[np.where(mean_mse <= threshold_1se)[0][0]]

print(f"Optimal k (Minimum CV MSE): {best_k_min} (MSE = {mean_mse[min_idx]:.4f})")
print(f"Optimal k (1-SE Rule):      {best_k_1se} (MSE =" f" {mean_mse[best_k_1se - 1]:.4f})")

# 5. Plot the CV MSE Curve
plt.figure(figsize=(10, 6))
plt.errorbar(k_values, mean_mse, yerr=se_mse, fmt="-o", capsize=4, color="navy", ecolor="lightblue", label="10-Fold CV MSE ± 1 SE",)

# Highlight Minimum MSE and 1-SE threshold
plt.axvline(best_k_min, color="crimson", linestyle="--", label=f"Min MSE (k = {best_k_min})",)
plt.axvline(best_k_1se, color="darkgreen", linestyle=":", linewidth=2, label=f"1-SE Rule (k = {best_k_1se})",)
plt.axhline(threshold_1se, color="gray", linestyle="-.", alpha=0.7, label="1-SE Threshold",)

plt.title("PCR Cross-Validation MSE vs. Number of Principal Components (k)")
plt.xlabel("Number of Principal Components (k)")
plt.ylabel("Cross-Validated Mean Squared Error (MSE)")
plt.xticks(np.arange(0, max_components + 1, 2))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

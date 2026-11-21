import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
from scipy import stats

df = pd.read_csv('tez/datasets/tecator/tecator.csv')
X = df[[f"absorbance_{i}" for i in range(1, 101)]]
y = df["fat"].astype(float)

# df = pd.read_csv('tez/datasets/concrete/Concrete_Data_Sheet.csv')
# target_col = 'Concrete compressive strength(MPa megapascals)'
# X = df.drop(columns=[target_col])
# y = df[target_col].astype(float)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# ols = LinearRegression().fit(X_train, y_train)
# pcr = make_pipeline(StandardScaler(), PCA(n_components=10), LinearRegression()).fit(X_train, y_train)
# print("OLS Test MSE:", mean_squared_error(y_test, ols.predict(X_test)))
# print("PCR Test MSE:", mean_squared_error(y_test, pcr.predict(X_test)))

# 2. Preprocessing: Standardization is mandatory for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n, p = X_scaled.shape

# 3. Fit Full Multiple Linear Regression (MLR)
mlr = LinearRegression()
mlr.fit(X_scaled, y)
rss_mlr = np.sum((y - mlr.predict(X_scaled))**2)

# 4. Fit Principal Component Regression (PCR)
# Determine k by retaining components that explain >= 95% of variance
pca_full = PCA(n_components=30)
X_pca = pca_full.fit_transform(X_scaled)
k = X_pca.shape[1]

pcr = LinearRegression()
pcr.fit(X_pca, y)
rss_pcr = np.sum((y - pcr.predict(X_pca))**2)

print(f"Original features (p): {p}")
print(f"Retained components (k): {k}\n")

# ==========================================
# TEST 1: In-Sample F-Test for Nested Models
# ==========================================
df1 = p - k
df2 = n - p - 1
f_stat = ((rss_pcr - rss_mlr) / df1) / (rss_mlr / df2)
p_value_f = stats.f.sf(f_stat, df1, df2)

print("--- In-Sample F-Test ---")
print(f"F-statistic: {f_stat}")
print(f"p-value: {p_value_f}")
print("Conclusion: If p > 0.05, dimension reduction is statistically adequate.\n")

# ==========================================================
# TEST 2: Out-of-Sample Paired t-test on 5-Fold Cross-Validation
# ==========================================================
mse_mlr_folds = []
mse_pcr_folds = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mlr_score = -cross_val_score(mlr, X, y, cv=kf, scoring='neg_mean_squared_error')
pcr_score = -cross_val_score(pcr, X, y, cv=kf, scoring='neg_mean_squared_error')

print(f"mlr_score: {np.mean(mlr_score)} | pcr_score: {np.mean(pcr_score)}\n")

for train_idx, test_idx in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mlr.fit(X_train, y_train)
    pred_mlr = mlr.predict(X_test)
    mse_mlr_folds.append(mean_squared_error(y_test, pred_mlr))

    # PCR (Fit PCA only on training fold to prevent data leakage)
    pca_cv = PCA(n_components=k)
    X_train_pca = pca_cv.fit_transform(X_train)
    X_test_pca = pca_cv.transform(X_test)

    pcr.fit(X_train_pca, y_train)
    pred_pcr = pcr.predict(X_test_pca)
    mse_pcr_folds.append(mean_squared_error(y_test, pred_pcr))
    print(f"R^2(MLR): {r2_score(y_test, pred_mlr):.4f} | R^2(PCR): {r2_score(y_test, pred_pcr):.4f}")

# Paired t-test
t_stat, p_value_t = stats.ttest_rel(mse_mlr_folds, mse_pcr_folds)

print("--- Out-of-Sample Paired t-Test (5-Fold CV) ---")
print(f"Mean MSE MLR: {np.mean(mse_mlr_folds):.4f}")
print(f"Mean MSE PCR: {np.mean(mse_pcr_folds):.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value (two-sided): {p_value_t:.4f}")
print("Conclusion: If p > 0.05, the predictive drop from PCR is not statistically significant.")

print("mse_mlr_folds:\n\t", mse_mlr_folds)
print("mse_pcr_folds:\n\t", mse_pcr_folds)

# Bu çalışmada kullanılan spektral veriler, Tecator firmasına ait Infratec Food and Feed Analyzer cihazı ile 850–1050 nm dalga boyu aralığında kaydedilmiştir (Borggaard & Thodberg, 1992)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("tez/datasets/tecator/tecator.csv")
n = len(df)
# Convert byte-valued fields (often nominal/string attributes) to text
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.decode("utf-8")

X_raw = df.iloc[:, 0:100].to_numpy().astype(float)

fat = df["fat"].to_numpy()
moisture = df["moisture"].to_numpy()
protein = df["protein"].to_numpy()

X_c = X_raw - np.outer(np.ones(n), np.mean(X_raw, axis=0))
s_j = np.sqrt(np.diag(X_c.T @ X_c) * 1 / (n - 1))
D = np.diag(s_j)
X_s = X_c @ np.linalg.inv(D)

# pca = PCA(n_components=len(X[0, :]))
# pc = pca.fit_transform(X)

# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.cumsum())

# fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# ax[0].bar(range(len(X[0, :])), pca.explained_variance_ratio_.cumsum())
# ax[0].set_xticks(range(len(X[0, :])))
# ax[1].bar(range(len(X[0, :])), pca.explained_variance_ratio_)
# ax[1].set_xticks(range(len(X[0, :])))

plt.show()

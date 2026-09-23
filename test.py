import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.style.use("dark_background")
rng = np.random.default_rng(42)

x = np.arange(0, 100)
y = 2 * x + 3 + rng.normal(0, 40, size=x.shape)
z = np.column_stack(([x.T, y]))
Z = z.mean(axis=0)
h = z - Z
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z)

avg_x = z[:, 0].sum() / len(z[:, 0])
avg_y = z[:, 1].sum() / len(z[:, 1])
k = np.empty_like(z)
k[:, 0] = z[:, 0] - avg_x
k[:, 1] = z[:, 1] - avg_y

def project(a, onto_b):
    return abs(np.dot(a, onto_b)) / np.linalg.norm(onto_b)

m_values = np.arange(0, 180, 1)
total_var = np.empty(180)

for idx, m_value in enumerate(m_values):
    total_d = np.empty(len(k[:, 0]))
    unit_vec = np.array([1, np.tan(np.deg2rad(m_value)) * 1]) / np.linalg.norm([1, np.tan(np.deg2rad(m_value)) * 1])
    for i in range(len(k[:,0])):
        total_d[i] = project(k[i, :], unit_vec)
    total_var[idx] = total_d.sum()

argmax_m = m_values[np.argmax(total_var)]
print("m = ", argmax_m)

# plt.plot(m_values, total_var)
eigvecs = np.array([1, np.tan(np.deg2rad(argmax_m)) * 1]) / np.linalg.norm([1, np.tan(np.deg2rad(argmax_m)) * 1])
U, s, Vh = np.linalg.svd(h, full_matrices=False)
print("PC(man)  eigvec: ", eigvecs)
print("PC(SVD)  eigvec: ", Vh[0])
print("PC(PCA)  eigvec: ", pca.components_[0])
print("slope(SVD)", Vh[0, 1] / Vh[0, 0]) # each row corresponds to respective principal axis
print("slope(PCA)", pca.components_[0, 1] / pca.components_[0, 0]) # each row corresponds to respective principal axis
print("slope(manuel)", eigvecs[1] / eigvecs[0])
plt.arrow(0, 0, 100 *pca.components_[0, 0], 100 * pca.components_[0, 1], color="red")
plt.arrow(0, 0, 100 *pca.components_[1, 0], 100 * pca.components_[1, 1], color="red")
plt.arrow(0, 0, 100 * Vh[0, 0], 100 * Vh[0, 1], color="cyan")
plt.arrow(0, 0, 100 * Vh[1, 0], 100 * Vh[1, 1], color="cyan")
plt.scatter(k[:, 0], k[:, 1], label="data")
plt.grid(True)
plt.legend()
plt.show()

# print("z_pca.shape ", z_pca.shape)
# print("pca.components_.shape ", pca.components_.shape)
# print("pca.components_ ", pca.components_)
# print("pca.explained_variance_ ", pca.explained_variance_)
# print("pca.explained_variance_ratio_ ", pca.explained_variance_ratio_)
# plt.scatter(z[:, 0], z[:, 1], label="z")
# plt.scatter(h[:, 0], h[:, 1], label="h")
# L = U @ np.diag(s) @ Vh
# print("Vh ", Vh)
# plt.scatter(L[:, 0], L[:, 1], label="L") # L and h same
# plt.legend()
# plt.show()

# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# model = make_pipeline(StandardScaler(), PCA(n_components=2))
# X_pca = model.fit_transform(z)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.cumsum())
# plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker="o")
# plt.xlabel("Number of components")
# plt.ylabel("Cumulative explained variance")
# plt.show()

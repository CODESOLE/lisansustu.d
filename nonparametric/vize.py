import numpy as np
import scipy.stats as stats

# --- SORU 1 ---
x1 = np.array([2.5, 2.4, 3.1, 3.0, 2.3, 2.6, 1.5, 3.3, 3.1, 2.5, 2.8, 3.3, 3.2, 2.6, 3.4])
# a) Normallik Testi (Shapiro-Wilk)
shapiro_s1 = stats.shapiro(x1)
print(f"Soru 1a - Shapiro-Wilk Normallik Testi: p-değeri = {shapiro_s1.pvalue:.4f}")
# b) t Testi - Sol kuyruk (Tek Yönlü: < 3.5)
wilcoxon_s1 = stats.wilcoxon(x1 - 3.5, alternative="less")
print(f"Soru 1b - Wilcoxon Testi p-değeri = {wilcoxon_s1.pvalue:.4f}")

# --- SORU 2 ---
yagis = np.array([119.2, 110.7, 150.1, 201.5, 308.5])
wilcoxon_s2 = stats.wilcoxon(yagis - 118.2, alternative="greater")
print(f"Soru 2 - Wilcoxon Testi p-değeri = {wilcoxon_s2.pvalue:.4f}")

# --- SORU 3 ---
grup1 = np.array([39, 30, 35, 35, 30, 43, 48, 56, 40, 34, 32, 59, 43, 43, 52, 52, 44, 53, 52, 38])
grup2 = np.array([35, 25, 36, 33, 25, 34, 31, 39, 24, 38, 29, 39, 31, 31, 34, 29, 38, 29, 32, 24])
# Mann-Whitney U Testi (Tek Yönlü: grup1 > grup2)
mwu_s3 = stats.mannwhitneyu(grup1, grup2, alternative="greater")
print(f"Soru 3 - Mann-Whitney U Testi p-değeri = {mwu_s3.pvalue:.6f}")

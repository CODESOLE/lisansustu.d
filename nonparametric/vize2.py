import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================ SORU 1 ============================
print(" SORU 1: Kekik yağlı yemle beslenen civcivlerde E.coli (log10) ".center(100, "="))
x = np.array([2.5, 2.4, 3.1, 3.0, 2.3, 2.6, 1.5, 3.3, 3.1, 2.5, 2.8, 3.3, 3.2, 2.6, 3.4])
n, mu0 = len(x), 3.5
m = x.mean()
s = x.std(ddof=1)
se = s / np.sqrt(n)

fig, axes = plt.subplots(1, 2)
fig.suptitle("SORU-1-a")
axes[0].set_title("histogram")
axes[1].set_title("Box-Wisker Plot")
sns.histplot(x, discrete=True, ax=axes[0])
sns.boxplot(x, ax=axes[1])
plt.tight_layout()
plt.show()

print(f"n = {n},  toplam = {x.sum():.2f},  medyan = {np.median(x):.3f}")
print(f"xbar = {m:.4f},  s = {s:.4f},  s^2 = {s**2:.4f},  SE = {se:.4f}")

W, pW = stats.shapiro(x)
print("""
H0: Veriler Normal Dağılıma Sahiptir.
H1: Veriler Normal Dağılıma Sahip Değildir.
""")

print(f"[1a] Shapiro-Wilk: W = {W:.4f},  p = {pW:.4f}")
print(f"     0.05 anlamlılık oranı için; {pW:.4f} > 0.05 old. H0 reddedilemedi. Yani veriler normal dağılıma sahiptir.\n")
print(cizgi)
print("\nBir önceki şıkta Normallik sağlandığı için tek örneklem t-testi uygulayabiliriz. (Sol kuyruk)")
t1 = (m - mu0) / se
tcrit1 = stats.t.ppf(0.05, n - 1)          # sol kuyruk kritik deger
p1 = stats.t.cdf(t1, n - 1)                # sol kuyruk p degeri
print("kekik yağı ilavesi -> E.coli sayısı < 3.5 ?\n")
print("""
H0: Kekik yağı barsaktaki E.coli sayısında düşüşe sebeb olmaz. mu >= 3.5
H1: Kekik yağı barsaktaki E.coli sayısında düşüşe sebeb olur.  mu <  3.5
""")
print(f"[1b] t = ({m:.4f} - {mu0}) / {se:.4f} = {t1:.4f},  df = {n-1}")
print(f"     t_tablo(0.05; {n-1}, tek yonlu) = {tcrit1:.4f}")
print(f"     p (tek yönlü) = {p1:.6f}")
print(f"     {p1:.6f} << 0.05 (çok küçük) old. H0 güçlü bir şekilde reddedildi. Yani Kekik yağı barsaktaki E.coli sayısında düşüşe sebeb olur.")


# ============================ SORU 2 ============================
# örneklem sayısı az old. için (n=5) ve ortanca karşılaştırması olduğundan Wilcoxon İşaretli Sıra Testi
print("\n")
print(" SORU 2: Adana Aralık ayı yagıs (kg/m2) ".center(100, "="))

print("""
M0 = 118.2
H0: Yağış miktarında artış olmamıştır. mu  <= M0
H1: Yağış miktarında artış olmuştur.   mu  >  M0
""")

yagis = np.array([119.2, 110.7, 150.1, 201.5, 308.5])
wilcoxon = stats.wilcoxon(yagis - 118.2, alternative="greater")
print(f"Soru 2 - Wilcoxon Testi p-değeri = {wilcoxon.pvalue:.4f}")
print(f"""
%95 Güven Düzeyinde incelersek,
p-değeri: {wilcoxon.pvalue:.4f} > 0.05 old. H0 reddedilemez.
Yani Dolayısıyla gazete haberinin iddiası istatistiksel olarak anlamlı/doğru bir iddia değildir.
""")

# ============================ SORU 3 ============================

print("\n")
print(" SORU 3: Muz meyve eti agirligi - mantar kompost (g1) vs ciflik gubresi (g2) ".center(100, "="))

g1 = np.array([39, 30, 35, 35, 30, 43, 48, 56, 40, 34, 32, 59,
               43, 43, 52, 52, 44, 53, 52, 38])
g2 = np.array([35, 25, 36, 33, 25, 34, 31, 39, 24, 38, 29, 39,
               31, 31, 34, 29, 38, 29, 32, 24])
n1, n2 = len(g1), len(g2)
m1, m2 = g1.mean(), g2.mean()
s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
v1, v2 = s1**2, s2**2

print(f"g1: n={n1}, toplam={g1.sum():.0f}, xbar={m1:.4f}, s={s1:.4f}, s2={v1:.4f}")
print(f"g2: n={n2}, toplam={g2.sum():.0f}, xbar={m2:.4f}, s={s2:.4f}, s2={v2:.4f}")
print(f"Shapiro-Wilk g1: W={stats.shapiro(g1)[0]:.4f}, p={stats.shapiro(g1)[1]:.4f}")
print(f"Shapiro-Wilk g2: W={stats.shapiro(g2)[0]:.4f}, p={stats.shapiro(g2)[1]:.4f}")

F = v1 / v2
Fc = stats.f.ppf(0.975, n1 - 1, n2 - 1)    # iki yonlu %5 F kritik
print(f"Varyans esitligi (F testi): F = {F:.4f},  "
      f"F_tablo(0.025; {n1-1}; {n2-1}) = {Fc:.4f}")
lev = stats.levene(g1, g2)
print(f"Levene testi: W = {lev[0]:.4f},  p = {lev[1]:.4f}")

# Havuzlanmis (pooled) t
sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
se_p = np.sqrt(sp2 * (1 / n1 + 1 / n2))
tp = (m1 - m2) / se_p
df_p = n1 + n2 - 2
pp = stats.t.sf(tp, df_p)
tc_p = stats.t.ppf(0.99, df_p)             # tek yonlu %1 kritik
print(f"Havuzlanmis: sp2 = {sp2:.4f},  SE = {se_p:.4f}")
print(f"  t = ({m1:.4f} - {m2:.4f}) / {se_p:.4f} = {tp:.4f},  df = {df_p}")
print(f"  t_tablo(0.01; {df_p}, tek yonlu) = {tc_p:.4f},  p = {pp:.6f}")

# Welch t (varyanslar esit degilse)
se_w = np.sqrt(v1 / n1 + v2 / n2)
tw = (m1 - m2) / se_w
df_w = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) +
                                   (v2 / n2) ** 2 / (n2 - 1))
pw = stats.t.sf(tw, df_w)
tc_w = stats.t.ppf(0.99, df_w)
print(f"Welch: SE = {se_w:.4f},  t = {tw:.4f},  df = {df_w:.2f}")
print(f"  t_tablo(0.01; {df_w:.1f}, tek yonlu) ~ {tc_w:.4f},  p = {pw:.6f}")

rw = stats.ttest_ind(g1, g2, equal_var=False, alternative="greater")
rp = stats.ttest_ind(g1, g2, equal_var=True, alternative="greater")
print(f"SciPy dogrulama: pooled t={rp.statistic:.4f} p={rp.pvalue:.6f} | "
      f"Welch t={rw.statistic:.4f} p={rw.pvalue:.6f}")
sp = np.sqrt(sp2)
print(f"Cohen d (pooled) = {(m1 - m2)/sp:.4f}")
print(f"Ortalama fark = {m1 - m2:.4f} g,  goreli artis = {100*(m1-m2)/m2:.1f}%")

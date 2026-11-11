# -*- coding: utf-8 -*-
# modeller arasindaki fark tesaduf mu degil mi: friedman testi + nemenyi post-hoc (demsar, 2006)
# her test goruntusu icin her modelin dogru/yanlis bildigini karsilastiriyor
# degerlendir.py'nin kaydettigi *_dogru.npy dosyalarini kullaniyor
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

modeller = ["efficientnetb0", "mobilenetv3", "densenet121", "vit_small", "resnet50"]
dogru = pd.DataFrame({m: np.load(f"sonuclar/{m}_dogru.npy") for m in modeller})

print("dogruluklar:")
print(dogru.mean().sort_values(ascending=False).round(4))

stat, p = friedmanchisquare(*[dogru[m] for m in modeller])
print(f"\nfriedman: chi2 = {stat:.3f}, p = {p:.4g}")

nemenyi = sp.posthoc_nemenyi_friedman(dogru.values)
nemenyi.index = nemenyi.columns = modeller
print("\nnemenyi p degerleri:")
print(nemenyi.round(4))
nemenyi.to_csv("sonuclar/nemenyi.csv")

plt.figure(figsize=(7, 6))
plt.imshow(nemenyi.values, cmap="RdYlGn", vmin=0, vmax=1)
plt.xticks(range(5), modeller, rotation=45, ha="right"); plt.yticks(range(5), modeller)
for i in range(5):
    for j in range(5):
        plt.text(j, i, f"{nemenyi.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
plt.colorbar(label="p degeri (p < 0.05: anlamli fark)")
plt.title("Nemenyi post-hoc")
plt.tight_layout()
plt.savefig("sonuclar/nemenyi.png", dpi=120)

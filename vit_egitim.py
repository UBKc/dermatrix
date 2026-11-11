# -*- coding: utf-8 -*-
# vit_small, pytorch + timm ile (keras'ta hazir vit olmadigi icin ayri yazdim)
# diger modellerle ayni iki asamali mantik: once sadece siniflandirici, sonra son bloklar
# kullanim: python vit_egitim.py
import os
import numpy as np
import torch
import timm
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight

VERI = "veri"
BATCH = 32
cihaz = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

ortalama, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)   # timm vit_small icin
egitim_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(ortalama, std)])
test_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(ortalama, std)])

tum = datasets.ImageFolder(f"{VERI}/train", egitim_tf)
tum_dogrulama = datasets.ImageFolder(f"{VERI}/train", test_tf)
sira = np.random.RandomState(42).permutation(len(tum))
ayrim = int(len(tum) * 0.1)
egitim = Subset(tum, sira[ayrim:])
dogrulama = Subset(tum_dogrulama, sira[:ayrim])
egitim_y = np.array(tum.targets)[sira[ayrim:]]

agirlik = compute_class_weight("balanced", classes=np.unique(egitim_y), y=egitim_y)
kayip_fn = nn.CrossEntropyLoss(weight=torch.tensor(agirlik, dtype=torch.float32).to(cihaz), label_smoothing=0.1)

egitim_dl = DataLoader(egitim, BATCH, shuffle=True, num_workers=2)
dogrulama_dl = DataLoader(dogrulama, BATCH, num_workers=2)

model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=5).to(cihaz)


def dogruluk(dl):
    model.eval(); dogru = toplam = 0
    with torch.no_grad():
        for x, y in dl:
            dogru += (model(x.to(cihaz)).argmax(1).cpu() == y).sum().item(); toplam += len(y)
    return dogru / toplam


def egit(epoch, lr):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    en_iyi = 0
    for e in range(epoch):
        model.train()
        for x, y in egitim_dl:
            opt.zero_grad()
            kayip = kayip_fn(model(x.to(cihaz)), y.to(cihaz))
            kayip.backward(); opt.step()
        acc = dogruluk(dogrulama_dl)
        print(f"epoch {e + 1}: dogrulama dogrulugu {acc:.4f}")
        if acc > en_iyi:
            en_iyi = acc
            os.makedirs("modeller", exist_ok=True)
            torch.save(model.state_dict(), "modeller/vit_small_best.pt")


# 1. asama: sadece siniflandirici kafa
for ad, p in model.named_parameters():
    p.requires_grad = ad.startswith("head")
egit(30, 1e-3)

# 2. asama: son 4 transformer blogu da aciliyor
for ad, p in model.named_parameters():
    p.requires_grad = ad.startswith("head") or any(ad.startswith(f"blocks.{i}.") for i in range(8, 12))
egit(15, 1e-5)

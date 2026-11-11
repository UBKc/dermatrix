# -*- coding: utf-8 -*-
# egitilen modeli test kumesinde degerlendiriyor
# cikti: siniflandirma raporu, karmasiklik matrisi, egitim egrisi ve tahminler (istatistik.py icin)
# kullanim: python degerlendir.py --model efficientnetb0
import argparse
import json
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modeller import MODELLER, BOYUT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="efficientnetb0")
parser.add_argument("--veri", default="veri")
args = parser.parse_args()

_, on_isleme = MODELLER[args.model]
model = tf.keras.models.load_model(f"modeller/{args.model}_best.keras")

test = ImageDataGenerator(preprocessing_function=on_isleme).flow_from_directory(
    f"{args.veri}/test", target_size=(BOYUT, BOYUT), batch_size=32,
    class_mode="categorical", shuffle=False)
siniflar = list(test.class_indices)

olasilik = model.predict(test)
tahmin = olasilik.argmax(axis=1)
gercek = test.classes

os.makedirs("sonuclar", exist_ok=True)
rapor = classification_report(gercek, tahmin, target_names=siniflar, digits=4)
print(rapor)
with open(f"sonuclar/{args.model}_rapor.txt", "w") as f:
    f.write(rapor)

# her goruntunun dogru bilinip bilinmedigi, modeller arasi istatistiksel test icin
np.save(f"sonuclar/{args.model}_dogru.npy", (tahmin == gercek).astype(int))
np.save(f"sonuclar/{args.model}_olasilik.npy", olasilik)

ConfusionMatrixDisplay(confusion_matrix(gercek, tahmin), display_labels=siniflar).plot(cmap="Blues")
plt.title(f"{args.model} karmasiklik matrisi")
plt.tight_layout()
plt.savefig(f"sonuclar/{args.model}_karmasiklik.png", dpi=120)
plt.close()

gecmis_dosya = f"sonuclar/{args.model}_gecmis.json"
if os.path.exists(gecmis_dosya):
    gecmis = json.load(open(gecmis_dosya))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(gecmis["loss"], label="egitim"); ax[0].plot(gecmis["val_loss"], label="dogrulama")
    ax[0].set_title("kayip"); ax[0].legend()
    ax[1].plot(gecmis["accuracy"], label="egitim"); ax[1].plot(gecmis["val_accuracy"], label="dogrulama")
    ax[1].set_title("dogruluk"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"sonuclar/{args.model}_egitim_egrisi.png", dpi=120)

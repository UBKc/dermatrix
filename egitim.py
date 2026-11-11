# -*- coding: utf-8 -*-
# iki asamali egitim: once transfer learning (taban dondurulmus), sonra fine-tuning
# kullanim: python egitim.py --model efficientnetb0
import argparse
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from modeller import model_olustur, BOYUT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="efficientnetb0")
parser.add_argument("--veri", default="veri")
parser.add_argument("--asama1_epoch", type=int, default=30)
parser.add_argument("--asama2_epoch", type=int, default=15)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--acilacak_katman", type=int, default=30)
parser.add_argument("--agirliksiz", action="store_true", help="imagenet agirliklarini indirmeden dene (sadece test icin)")
args = parser.parse_args()

model, taban, on_isleme = model_olustur(args.model, None if args.agirliksiz else "imagenet")

# veri artirma sadece egitim goruntulerine, dogrulamaya sadece on isleme
# dogrulama kumesi egitim klasorunun %10'undan ayriliyor, test kumesine egitimde hic dokunulmuyor
egitim_gen = ImageDataGenerator(
    preprocessing_function=on_isleme,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.1,
)
dogrulama_gen = ImageDataGenerator(preprocessing_function=on_isleme, validation_split=0.1)

egitim = egitim_gen.flow_from_directory(
    f"{args.veri}/train", target_size=(BOYUT, BOYUT), batch_size=args.batch,
    class_mode="categorical", subset="training", seed=42)
dogrulama = dogrulama_gen.flow_from_directory(
    f"{args.veri}/train", target_size=(BOYUT, BOYUT), batch_size=args.batch,
    class_mode="categorical", subset="validation", shuffle=False, seed=42)
print("siniflar:", egitim.class_indices)

# siniflar dengesiz (mantar 6878, saglikli 1997 goruntu), az olan siniflarin hatasi daha agir sayiliyor
agirlik = compute_class_weight("balanced", classes=np.unique(egitim.classes), y=egitim.classes)
sinif_agirlik = dict(enumerate(agirlik))
print("sinif agirliklari:", {k: round(float(v), 2) for k, v in sinif_agirlik.items()})

kayip = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
os.makedirs("modeller", exist_ok=True)
kayit = f"modeller/{args.model}_best.keras"


def callbacklar():
    return [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ModelCheckpoint(kayit, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
    ]


# 1. asama: taban dondurulmus, lr 1e-3
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=kayip, metrics=["accuracy"])
gecmis1 = model.fit(egitim, validation_data=dogrulama, epochs=args.asama1_epoch,
                    class_weight=sinif_agirlik, callbacks=callbacklar())

# 2. asama: tabanin son 30 katmani aciliyor, lr 1e-5
# batchnorm katmanlari kapali kaliyor, kucuk batch'te istatistikleri bozulmasin diye
taban.trainable = True
for katman in taban.layers[:-args.acilacak_katman]:
    katman.trainable = False
for katman in taban.layers:
    if isinstance(katman, tf.keras.layers.BatchNormalization):
        katman.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=kayip, metrics=["accuracy"])
gecmis2 = model.fit(egitim, validation_data=dogrulama, epochs=args.asama2_epoch,
                    class_weight=sinif_agirlik, callbacks=callbacklar())

# egitim egrisi icin iki asamayi birlestirip kaydediyorum
gecmis = {k: gecmis1.history[k] + gecmis2.history[k] for k in gecmis1.history if k in gecmis2.history}
os.makedirs("sonuclar", exist_ok=True)
with open(f"sonuclar/{args.model}_gecmis.json", "w") as f:
    json.dump(gecmis, f)
print(f"en iyi model: {kayit}")

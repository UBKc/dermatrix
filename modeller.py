# -*- coding: utf-8 -*-
# karsilastirdigimiz 4 cnn modeli (vit ayri, pytorch ile: vit_egitim.py)
# hepsine ayni ust katmanlari ekliyorum ki karsilastirma adil olsun
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import applications as app

BOYUT = 224
SINIF_SAYISI = 5

# her modelin kendi on isleme fonksiyonu var, imagenet'te nasil egitildiyse oyle
MODELLER = {
    "efficientnetb0": (app.EfficientNetB0, app.efficientnet.preprocess_input),
    "resnet50":       (app.ResNet50, app.resnet50.preprocess_input),
    "mobilenetv3":    (app.MobileNetV3Large, app.mobilenet_v3.preprocess_input),
    "densenet121":    (app.DenseNet121, app.densenet.preprocess_input),
}


def model_olustur(ad, agirliklar="imagenet"):
    taban_sinif, on_isleme = MODELLER[ad]
    taban = taban_sinif(include_top=False, weights=agirliklar, input_shape=(BOYUT, BOYUT, 3))
    taban.trainable = False   # 1. asamada sadece ust katmanlar egitiliyor

    girdi = layers.Input((BOYUT, BOYUT, 3))
    x = taban(girdi, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    cikti = layers.Dense(SINIF_SAYISI, activation="softmax")(x)
    return Model(girdi, cikti), taban, on_isleme

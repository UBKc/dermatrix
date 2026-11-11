# -*- coding: utf-8 -*-
# mobil uygulamanin (node.js backend uzerinden) cagirdigi yapay zeka servisi
# POST /analiz  -> form-data: foto (dosya), cevaplar (istege bagli, json)
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from modeller import MODELLER, BOYUT
from semptom import fuzyon, SINIFLAR

app = Flask(__name__)
model = tf.keras.models.load_model("modeller/efficientnetb0_best.keras")
_, on_isleme = MODELLER["efficientnetb0"]


@app.post("/analiz")
def analiz():
    if "foto" not in request.files:
        return jsonify({"hata": "foto gonderilmedi"}), 400

    resim = Image.open(request.files["foto"].stream).convert("RGB").resize((BOYUT, BOYUT))
    girdi = on_isleme(np.array(resim, dtype="float32")[None])
    cnn = model.predict(girdi, verbose=0)[0].tolist()
    sonuc = {"cnn": dict(zip(SINIFLAR, cnn))}

    # anket cevaplanmissa gorsel sonuc semptomlarla birlestiriliyor
    if request.form.get("cevaplar"):
        try:
            cevaplar = json.loads(request.form["cevaplar"])
            nihai, acil = fuzyon(cnn, cevaplar)
        except (ValueError, json.JSONDecodeError) as e:
            return jsonify({"hata": str(e)}), 400
        sonuc["nihai"] = nihai
        sonuc["acil_uyari"] = acil
    else:
        nihai = sonuc["cnn"]

    sonuc["tahmin"] = max(nihai, key=nihai.get)
    return jsonify(sonuc)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

# -*- coding: utf-8 -*-
# grad-cam: modelin karar verirken goruntunun neresine baktigini gosteren isi haritasi
# kullanim: python gradcam.py foto.jpg
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from modeller import MODELLER, BOYUT

SINIFLAR = ["akne", "egzama", "mantar", "saglikli", "sedef"]


def gradcam(model, girdi, sinif=None):
    # modelin icindeki efficientnet tabanini bulup son konvolusyon ciktisini aliyorum
    taban = next(k for k in model.layers if isinstance(k, tf.keras.Model))
    son_conv = taban.get_layer("top_activation")
    taban_cikis = tf.keras.Model(taban.input, [son_conv.output, taban.output])

    with tf.GradientTape() as tape:
        conv, x = taban_cikis(girdi, training=False)
        tape.watch(conv)
        # tabandan sonraki katmanlari (pooling, dropout, dense) elle calistiriyorum
        for katman in model.layers[model.layers.index(taban) + 1:]:
            x = katman(x, training=False)
        if sinif is None:
            sinif = int(tf.argmax(x[0]))
        skor = x[:, sinif]

    gradyan = tape.gradient(skor, conv)[0]
    agirlik = tf.reduce_mean(gradyan, axis=(0, 1))          # her kanalin onemi
    harita = tf.nn.relu(tf.reduce_sum(conv[0] * agirlik, axis=-1))
    harita = harita / (tf.reduce_max(harita) + 1e-8)
    return harita.numpy(), sinif, float(x[0, sinif])


if __name__ == "__main__":
    model = tf.keras.models.load_model("modeller/efficientnetb0_best.keras")
    _, on_isleme = MODELLER["efficientnetb0"]

    resim = image.img_to_array(image.load_img(sys.argv[1], target_size=(BOYUT, BOYUT)))
    harita, sinif, guven = gradcam(model, on_isleme(resim[None].copy()))
    harita = tf.image.resize(harita[..., None], (BOYUT, BOYUT)).numpy()[..., 0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(resim.astype("uint8")); ax[0].set_title("orijinal"); ax[0].axis("off")
    ax[1].imshow(resim.astype("uint8")); ax[1].imshow(harita, cmap="jet", alpha=0.45)
    ax[1].set_title(f"{SINIFLAR[sinif]} (%{guven * 100:.1f})"); ax[1].axis("off")
    plt.tight_layout()
    plt.savefig("gradcam_sonuc.png", dpi=120)
    print(f"tahmin: {SINIFLAR[sinif]} (%{guven * 100:.1f}), gradcam_sonuc.png kaydedildi")

# -*- coding: utf-8 -*-
# 12 soruluk semptom anketi ve cnn ciktisiyla birlestirme (bayesian carpimsal fuzyon)
# carpanlar acil tip uzmani danismanliginda hazirlandi
# sinif sirasi her yerde ayni: akne, egzama, mantar, saglikli, sedef

SINIFLAR = ["akne", "egzama", "mantar", "saglikli", "sedef"]

CARPANLAR = {
    "kasinti": {
        "yok":      [1.0, 0.5, 0.7, 1.3, 0.7],
        "hafif":    [0.9, 1.2, 1.1, 0.5, 1.1],
        "siddetli": [0.6, 1.6, 1.3, 0.05, 1.2],
    },
    "yanma": {
        "yok":      [1.0, 0.8, 0.8, 1.2, 1.0],
        "hafif":    [0.9, 1.2, 1.3, 0.6, 0.9],
        "siddetli": [0.6, 1.5, 1.6, 0.15, 0.8],
    },
    "uyku": {
        "yok":    [1.0, 0.6, 1.0, 1.3, 1.0],
        "bazen":  [0.9, 1.3, 1.0, 0.6, 1.1],
        "sik":    [0.6, 2.0, 0.9, 0.05, 1.0],
    },
    "halka": {
        "yok":         [1.1, 1.0, 0.5, 1.2, 1.1],
        "evet":        [0.4, 0.5, 2.2, 0.2, 0.4],
        "emin_degil":  [1.0, 1.0, 1.2, 0.8, 1.0],
    },
    "kanama": {
        "yok":      [1.0, 0.9, 1.0, 1.1, 0.7],
        "catlama":  [0.8, 1.3, 0.7, 0.2, 1.5],
        "kanama":   [0.5, 0.8, 0.4, 0.02, 1.9],
    },
    "sure": {
        "lezyon_yok":   [0.6, 0.6, 0.6, 2.0, 0.6],
        "1_haftadan_az": [1.2, 0.8, 0.9, 1.0, 0.6],
        "1_4_hafta":    [1.0, 1.1, 1.2, 0.8, 1.0],
        "1_aydan_fazla": [0.7, 1.3, 1.1, 0.5, 1.5],
    },
    "baslangic": {
        "yok":         [0.6, 0.6, 0.6, 2.0, 0.6],
        "aniden":      [1.4, 1.0, 0.8, 0.9, 0.6],
        "kademeli":    [0.8, 1.0, 1.4, 0.9, 1.0],
        "tekrarlayan": [0.6, 1.4, 0.7, 0.2, 1.7],
    },
    "tetikleyici": {
        "yok":          [0.9, 0.7, 0.8, 1.5, 0.8],
        "stres":        [0.7, 1.2, 0.5, 0.6, 1.8],
        "kimyasal":     [0.5, 2.0, 0.6, 0.3, 0.7],
        "spor_terleme": [0.8, 0.8, 2.0, 0.4, 0.5],
        "bilmiyorum":   [1.1, 0.9, 1.0, 1.1, 1.0],
    },
    "tirnak": {
        "hayir": [1.1, 1.0, 0.9, 1.1, 0.9],
        "evet":  [0.2, 0.4, 1.7, 0.1, 1.6],
    },
    "eklem": {
        "hayir": [1.0, 1.0, 1.0, 1.1, 0.9],
        "evet":  [0.4, 0.6, 0.4, 0.2, 2.2],
    },
    "aile": {
        "yok":    [1.1, 0.9, 1.1, 1.1, 0.9],
        "sedef":  [0.3, 0.6, 0.3, 0.2, 2.5],
        "egzama": [0.5, 2.0, 0.5, 0.2, 0.6],
    },
    "yayilim": {
        "yok":           [1.0, 0.7, 0.9, 1.5, 0.6],
        "tek_bolge":     [1.2, 0.9, 1.1, 1.0, 0.8],
        "birden_fazla":  [0.8, 1.1, 1.0, 0.6, 1.3],
        "yaygin":        [0.6, 1.2, 0.7, 0.2, 1.5],
    },
}

ALFA = 0.20   # cnn ciktisini yumusatma orani


def fuzyon(cnn_olasiliklar, cevaplar):
    # 1) cnn ciktisini yumusatiyorum: hicbir sinif tam 0 olmasin,
    #    yoksa semptomlar ne derse desin o sinif geri gelemiyor
    #    yumusak = (1 - alfa) * cnn + alfa / 5
    yumusak = [(1 - ALFA) * p + ALFA / len(SINIFLAR) for p in cnn_olasiliklar]

    # 2) verilen her cevabin carpanlarini sinif sinif carpiyorum
    carpan = [1.0] * len(SINIFLAR)
    for soru, cevap in cevaplar.items():
        if soru not in CARPANLAR or cevap not in CARPANLAR[soru]:
            raise ValueError(f"gecersiz cevap: {soru}={cevap}")
        for i, c in enumerate(CARPANLAR[soru][cevap]):
            carpan[i] *= c

    # 3) yumusak olasilik x carpan, sonra toplami 1 olacak sekilde normalize
    skor = [y * c for y, c in zip(yumusak, carpan)]
    toplam = sum(skor)
    sonuc = [s / toplam for s in skor]

    # yaygin lezyonda sonuc ne olursa olsun doktora yonlendirme uyarisi
    acil_uyari = cevaplar.get("yayilim") == "yaygin"
    return dict(zip(SINIFLAR, sonuc)), acil_uyari

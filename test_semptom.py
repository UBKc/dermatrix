# tezdeki ornek hesaplamayla ayni sonucu verdigini kontrol ediyorum (EK 3)
# cnn: mantar 0.60, digerleri 0.10 / cevaplar: siddetli kasinti, halka var, spordan sonra artiyor
# beklenen: mantar yaklasik %95
from semptom import fuzyon


def test_mantar_ornegi():
    cnn = [0.10, 0.10, 0.60, 0.10, 0.10]
    cevap = {"kasinti": "siddetli", "halka": "evet", "tetikleyici": "spor_terleme"}
    sonuc, acil = fuzyon(cnn, cevap)
    assert abs(sonuc["mantar"] - 0.958) < 0.002
    assert sonuc["saglikli"] < 0.001
    assert acil is False


def test_cevapsiz_sonuc_cnn_sirasini_korur():
    cnn = [0.05, 0.70, 0.05, 0.05, 0.15]
    sonuc, _ = fuzyon(cnn, {})
    assert max(sonuc, key=sonuc.get) == "egzama"
    assert abs(sum(sonuc.values()) - 1) < 1e-9


def test_yaygin_lezyon_acil_uyari():
    _, acil = fuzyon([0.2] * 5, {"yayilim": "yaygin"})
    assert acil is True


if __name__ == "__main__":
    test_mantar_ornegi()
    test_cevapsiz_sonuc_cnn_sirasini_korur()
    test_yaygin_lezyon_acil_uyari()
    print("testler gecti")

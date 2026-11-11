# Dermatrix – Cilt Hastalığı Ön Değerlendirme

*Skin disease pre-assessment: 5 deep learning models compared on 20,213 images (EfficientNetB0 best, 97.87% accuracy), statistical validation with Friedman/Nemenyi, Grad-CAM, and a symptom questionnaire fused with the CNN output.*

Erzincan Binali Yıldırım Üniversitesi Bilgisayar Mühendisliği bitirme projemiz. Sena Ürenlioğlu ile birlikte geliştirdik, danışmanımız Doç. Dr. Volkan Kaya. Proje, TEKNOFEST 2025'te finale kalan [BioPredict – DeriTanı](https://github.com/UBKc/biopredict-deritani) projemizin devamı.

Fotoğraftan 5 sınıfı ayırt ediyor: **akne, egzama, mantar, sedef ve sağlıklı cilt**. Görsel sonucu kullanıcının cevapladığı 12 soruluk semptom anketiyle birleştiriyor. Amaç teşhis koymak değil, dermatoloğa erişimi zor olan kişiye ön değerlendirme yapıp doktora yönlendirmek.

Bu repoda projenin yapay zekâ kısmı var: model eğitimi, karşılaştırma, istatistiksel analiz, Grad-CAM, semptom füzyonu ve Flask servisi. Mobil uygulama (React Native) ve backend (Node.js, MongoDB) bu repoda yok.

> Projenin orijinal kodları elimde kalmadığı için bu repodaki kodu bitirme tezindeki yönteme ve ayarlara göre yeniden düzenleyerek yazdım. Aşağıdaki sonuçlar tezdeki deneylerden.

## Veri seti

Kaggle ve DermNet NZ'den derlenen, 224x224 boyutunda 20.213 görüntü:

| Sınıf | Eğitim | Test | Toplam |
|---|---|---|---|
| Akne | 1.701 | 539 | 2.240 |
| Egzama | 3.106 | 1.676 | 4.782 |
| Mantar | 3.654 | 3.224 | 6.878 |
| Sağlıklı | 1.633 | 364 | 1.997 |
| Sedef | 2.834 | 1.482 | 4.316 |
| **Toplam** | **12.928** | **7.285** | **20.213** |

"Sağlıklı" sınıfına sadece yüz değil el, kol, omuz ve tırnak görüntüleri de eklendi. Böylece model "yüz = sağlıklı" gibi yanlış bir kısayol öğrenmek yerine dokudaki bozukluğa bakmayı öğreniyor.

Sınıflar dengesiz olduğu için eğitimde `class_weight="balanced"` kullanıldı.

## Yöntem

Aynı veri ve aynı ayarlarla 5 model eğitildi: **EfficientNetB0, MobileNetV3, DenseNet121, ResNet50** (Keras) ve **ViT_Small** (PyTorch). Hepsine aynı üst katmanlar eklendi:

```
taban model -> GlobalAveragePooling -> Dropout(0.5) -> Dense(256, ReLU) -> Dense(5, Softmax)
```

Eğitim iki aşamalı:

1. **Transfer learning (30 epoch):** ImageNet ağırlıklı taban dondurulmuş, sadece üst katmanlar eğitiliyor, öğrenme hızı 1e-3.
2. **Fine-tuning (en fazla 15 epoch):** Tabanın son 30 katmanı açılıyor, öğrenme hızı 1e-5. EfficientNetB0'da erken durdurma 7. epoch civarında devreye girdi, toplam yaklaşık 37 epoch sürdü.

| Ayar | Değer |
|---|---|
| Batch | 32 |
| Optimizer | Adam |
| Kayıp | Categorical crossentropy + label smoothing (0.1) |
| Veri artırma | ±20° döndürme, %15 kaydırma, %15 yakınlaştırma, yatay çevirme |
| Erken durdurma | val_loss, patience 7 |
| ReduceLROnPlateau | factor 0.2, patience 3 |

Eğitim NVIDIA RTX 5060 Ti üzerinde yapıldı.

## Sonuçlar

### Model karşılaştırması (7.285 test görüntüsü)

| Model | Doğruluk | F1 |
|---|---|---|
| **EfficientNetB0** | **%97,87** | **0,970** (makro) |
| MobileNetV3 | %97,70 | 0,973 |
| DenseNet121 | %97,12 | 0,965 |
| ViT_Small | %96,95 | 0,961 |
| ResNet50 | %96,67 | 0,960 |

EfficientNetB0'ın değerleri aşağıdaki sınıf bazlı tablodan. Diğer modellerin değerleri tezdeki karşılaştırma tablosundan.

### EfficientNetB0 sınıf bazlı

| Sınıf | Kesinlik | Duyarlılık | F1 | Test görüntüsü |
|---|---|---|---|---|
| Akne | 0,976 | 0,976 | 0,976 | 539 |
| Egzama | 0,938 | 0,974 | 0,955 | 1.676 |
| Mantar | 0,999 | 0,993 | 0,996 | 3.224 |
| Sağlıklı | 0,939 | 0,984 | 0,961 | 364 |
| Sedef | 0,974 | 0,951 | 0,962 | 1.482 |
| **Makro ortalama** | **0,965** | **0,975** | **0,970** | 7.285 |

En çok karışan ikili **egzama ve sedef**. İkisinde de kızarıklık, pullanma ve kronik lezyon var, görüntüden ayırmak zor. Bu hata 5 modelin hepsinde görüldü. Mantar ise halka şeklindeki lezyonu sayesinde neredeyse hatasız ayrılıyor.

### Farklar tesadüf mü? (Friedman + Nemenyi)

Doğruluklar birbirine yakın olduğu için farkların gerçek olup olmadığını Friedman testi ve Nemenyi post-hoc analiziyle kontrol ettik:

| Karşılaştırma | p | Sonuç |
|---|---|---|
| EfficientNetB0 – ViT_Small | ≈ 0,0000 | anlamlı fark |
| EfficientNetB0 – ResNet50 | 0,0012 | anlamlı fark |
| EfficientNetB0 – MobileNetV3 | 0,5258 | fark yok |
| EfficientNetB0 – DenseNet121 | 0,4633 | fark yok |

Yani EfficientNetB0, ViT_Small ve ResNet50'den gerçekten daha iyi. MobileNetV3 ile arasında ise istatistiksel fark yok. MobileNetV3 çok daha hafif olduğu için telefonda çalıştırmak için iyi bir alternatif.

### Grad-CAM

Modelin karar verirken görüntünün neresine baktığını Grad-CAM ile kontrol ettik:
- Mantarda halka şeklindeki lezyonun kenarına odaklanıyor.
- Aknede sivilce bölgelerine odaklanıyor.
- Sedefi egzama sandığı örneklerde pullanma ve kızarıklığa bakıyor. Bu, iki hastalığın neden karıştığını görsel olarak da gösteriyor.

## Semptom anketi ile birleştirme

Egzama–sedef karışıklığını sadece görüntüyle çözmek zor. Doktor da teşhis koyarken sadece bakmıyor, hastaya soru soruyor. Bu yüzden görüntü analizinden sonra kullanıcıya 12 soru soruluyor (kaşıntı, halka şekli, tetikleyici, ailede sedef, eklem ağrısı...). Sorular ve çarpanlar bir acil tıp uzmanının danışmanlığında hazırlandı.

Her cevabın her sınıf için bir çarpanı var (`semptom.py`). Örnek: ailede sedef varsa sedef x2.5, halka şekli varsa mantar x2.2. CNN çıktısı bu çarpanlarla çarpılıp normalize ediliyor:

```
yumusak = 0.8 * cnn + 0.04                      # hicbir sinif tam 0 kalmasin
nihai   = normalize(yumusak * carpan_1 * carpan_2 * ...)
```

**Örnek:** CNN mantar için %60 diyor. Kullanıcı "şiddetli kaşıntı, halka şeklinde, spordan sonra artıyor" cevaplarını veriyor. Sonuç mantar için %95,8'e çıkıyor (`test_semptom.py` bu örneği test ediyor). Lezyonlar yaygınsa sonuç ne olursa olsun "dermatoloğa başvurun" uyarısı veriliyor.

Uzmanın test kümesinden seçtiği örnek vakalarda anketle birlikte doğruluk %97,87'den %98,5'e çıktı, yanlış pozitif oranı %5,2'den %3,2'ye düştü. En büyük iyileşme egzama–sedef ayrımında oldu. Bu sonuç tüm test kümesinde değil, sınırlı sayıda örnek vaka üzerinde ölçüldü.

## Dosyalar

| Dosya | Ne işe yarıyor |
|---|---|
| `modeller.py` | 4 CNN modelinin ortak yapısı |
| `egitim.py` | iki aşamalı eğitim (`--model efficientnetb0`, `resnet50`, `mobilenetv3`, `densenet121`) |
| `vit_egitim.py` | ViT_Small eğitimi (PyTorch + timm) |
| `degerlendir.py` | test kümesinde rapor, karmaşıklık matrisi, eğitim eğrisi |
| `istatistik.py` | Friedman testi + Nemenyi ısı haritası |
| `gradcam.py` | tek fotoğraf için Grad-CAM ısı haritası |
| `semptom.py` | 12 soru, çarpan matrisi ve füzyon |
| `api.py` | Flask servisi (port 5001), mobil uygulamanın backend'i bunu çağırıyor |

## Çalıştırma

Veri klasörü yapısı:

```
veri/
├── train/akne, egzama, mantar, saglikli, sedef
└── test/akne, egzama, mantar, saglikli, sedef
```

```
pip install -r requirements.txt
python egitim.py --model efficientnetb0
python degerlendir.py --model efficientnetb0
# diger modeller icin --model degistirilerek tekrarlanir, vit icin: python vit_egitim.py
python istatistik.py
python gradcam.py foto.jpg
python api.py
```

Kodu küçük bir deneme veri setiyle baştan sona çalıştırıp test ettim. Bu repodaki kodda doğrulama kümesi eğitim klasörünün %10'undan ayrılıyor, test kümesine eğitim sırasında hiç dokunulmuyor.

## Eksikler ve geliştirilebilecekler

- Eğitim/test ayrımında sınıfların test oranı farklı kaldı (sağlıklıda %18, mantarda %47). Tam stratified bir ayrımla tekrar denenmeli.
- Görüntüler iki farklı kaynaktan geldiği için aynı ya da çok benzer görüntülerin hem eğitimde hem testte bulunup bulunmadığı ayrıca kontrol edilmeli.
- Semptom anketinin katkısı daha fazla gerçek vakayla, dermatologlarla birlikte ölçülmeli.
- Farklı cilt tonlarından daha fazla örnek eklenmeli.
- Veri seti ve model ağırlıkları repoda yok.

## Lisans

MIT

# Proje Raporu: Kan Tahlili Olmadan Gizli Diyabeti Bulmak

Makine Öğrenmesi dersi dönem projesi · Yusuf Gündüz ve ekip arkadaşı

## 1. Çözülen problem

Türkiye'de diyabetlilerin **%45,5'i (4,4 milyon kişi)** hastalığından habersiz
(IDF Diyabet Atlası, 2024). Diyabet belirti vermeden ilerler ve tanı için kan tahlili gerekir.
Kimin tahlil yaptırması gerektiğine bugün ADA ve FINDRISC gibi 7-8 soruluk sabit puan
tablolarıyla karar veriliyor.

**Hedef:** Kan tahlili gerektirmeden, yalnızca sorularla, doktorun henüz tanı koymadığı ama
tahlili diyabet çıkan kişileri bu klinik testlerden daha iyi bulan bir makine öğrenmesi modeli.

## 2. Veri

| Veri seti | Kaynak | Boyut | Kullanım |
|---|---|---|---|
| CDC BRFSS 2015 | UCI Machine Learning Repository #891 | 253.680 kişi, 21 soru | Genel diyabet risk modeli |
| CDC NHANES 2007–2020 | CDC (90 dosya, 6 dönem) | 29.399 tanısız yetişkin | Gizli diyabet modeli, klinik karşılaştırma |
| IDF Diyabet Atlası 2024 | diabetesatlas.org | Türkiye ülke profili | Türkiye kalibrasyonu |

Kişi bazlı, kan tahlili içeren açık bir Türkiye verisi bulunamadı (TÜİK mikro verisi başvuruyla
ve TÜİK binasında kullanılıyor; WHO arşivinde Türkiye için yalnızca tütün anketleri var).

## 3. Yöntem

1. **Veri hazırlama:** tekrar eden satırlar, eksik ve aykırı değerler, ölçekleme; %14 diyabet
   oranındaki dengesizlik için class_weight ve SMOTE karşılaştırması.
2. **Model yarışı (BRFSS):** lojistik regresyon, KNN, SVM, karar ağacı, Random Forest, AdaBoost,
   XGBoost, Keras sinir ağı; GridSearchCV + 5-katlı Stratified K-Fold.
3. **Gizli diyabet modeli (NHANES):** hedef kan tahliliyle tanımlandı (HbA1c ≥ 6,5 veya açlık
   şekeri ≥ 126, doktor tanısı yok). Zamansal doğrulama: 2007–2016 eğitim, 2017–2020 test.
4. **Klinik karşılaştırma:** ADA Risk Testi ve FINDRISC aynı kişilerde hesaplandı; 2000 tekrarlı
   bootstrap güven aralıkları ve aynı yanlış alarm sayısında yakalanan hasta karşılaştırması.
5. **Kalite kontrolleri:** veri sızıntısı tespiti, 6 geliştirme adayının önceden belirlenmiş kurala
   göre elenmesi, 13 alt grupta adalet analizi, gereksiz sosyoekonomik soruların ablasyonla çıkarılması.
6. **Türkiye kalibrasyonu:** olasılıklar IDF 2024 Türkiye verisine göre önsel olasılık kaydırmasıyla
   (Saerens ve ark., 2002) yeniden ölçeklendi. Tanısı olmayan yetişkinlerde gizli diyabet oranı
   Türkiye'de %8,1, ABD eğitim verisinde %5,3. Dönüşüm monoton olduğu için AUC ve kararlar değişmez.

## 4. Sonuçlar

| Ölçüt | Model | FINDRISC | ADA |
|---|---|---|---|
| ROC-AUC (2017–20 test) | **0,809** | 0,769 | 0,751 |
| %95 güven aralığı | 0,790–0,830 | 0,746–0,792 | 0,729–0,772 |
| Aynı yanlış alarmla yakalanan (389 hastadan) | **336** / **242** | 218 | 322 |

- Model ADA'yı ve FINDRISC'i anlamlı farkla geçiyor (fark güven aralıkları sıfırı içermiyor).
- 13 alt grubun hepsinde model iki klinik testten de iyi.
- BRFSS'te en iyi model XGBoost (AUC 0,818); ilk altı model arasındaki fark 0,01'in altında.
- Daha fazla veri (20 bin → 160 bin kişi) ve daha uzun eğitim (300 → 3000 ağaç) başarıyı artırmadı.
- **Ana bulgu:** Fark algoritmadan değil sorulardan geliyor. ADA'nın 7 sorusunu ML ile puanlamak
  +0,009 AUC kazandırırken daha fazla soru eklemek +0,059 kazandırdı.

## 5. Ürün

- **Web uygulaması** (FastAPI + HTML/CSS/JS): risk değerlendirmesi, SHAP açıklaması, model ile
  FINDRISC ve ADA'nın yan yana karşılaştırması, model raporu; açık/koyu tema, mobil uyumlu.
- **Otomatik testler:** `tests/test_api.py`, 12 test (girdi doğrulama, klinik puanlar, kalibrasyon).

## 6. Sınırlılıklar

- Modeller ABD verisiyle eğitildi; Türkiye'ye yalnızca taban oran kalibrasyonu yapıldı, kişi bazlı
  Türkiye verisiyle doğrulanmadı.
- Yanıtlar kişisel beyan; gizli diyabet tanımı çoğunlukla tek bir HbA1c ölçümüne dayanıyor.
- 65 yaş üstü, erkekler ve siyahi katılımcılarda başarı daha düşük.
- NHANES örnekleme ağırlıkları kullanılmadı.
- Bu bir tarama aracıdır, tanı koymaz.

## 7. Sonraki adımlar

TÜİK Türkiye Sağlık Araştırması mikro verisine başvuru ile Türkiye'de doğrulama; aile hekimliği
ortamında ileriye dönük deneme.

## Kaynaklar

- CDC BRFSS 2015, UCI Machine Learning Repository #891
- CDC NHANES 2007–2020
- IDF Diyabet Atlası, Türkiye ülke profili (2024): https://diabetesatlas.org/data-by-location/country/trkiye/
- Bang H. ve ark. (2009). ADA Diabetes Risk Test. Annals of Internal Medicine.
- Lindström J., Tuomilehto J. (2003). FINDRISC. Diabetes Care.
- Saerens M. ve ark. (2002). Adjusting the outputs of a classifier to new a priori probabilities. Neural Computation.
- Xie Z. ve ark. (2019). Preventing Chronic Disease.

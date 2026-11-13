<div align="center">

# DiyabetAI: Makine Öğrenmesiyle Diyabet Riski Tahmini

**Basit sağlık anketi sorularından diyabet riskini tahmin eder. CDC BRFSS'in 253.680 kişilik anketiyle 8 model eğitilip karşılaştırıldı; ikinci aşamada kan tahlili olmadan teşhis edilmemiş diyabet aranıyor.**

[![Canlı demo](https://img.shields.io/badge/Canl%C4%B1_demo-diyabet--risk--analizi.netlify.app-00c7b7?style=for-the-badge)](https://diyabet-risk-analizi.netlify.app)

![Python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-f7931e?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189fdd?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-a%C3%A7%C4%B1klanabilirlik-6f42c1?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Lisans](https://img.shields.io/badge/Lisans-MIT-green?style=flat-square)

Türkçe · [English](README.md)

</div>

> **Tıbbi uyarı.** DiyabetAI, eğitim ve araştırma amaçlı bir üniversite makine öğrenmesi projesidir.
> **Tıbbi cihaz değildir** ve teşhis koymaz. Modeller Amerika Birleşik Devletleri anket verisiyle eğitilmiştir.
> Diyabetten endişeleniyorsanız bir hekime başvurun.

## Ne yapar?

Diyabet genellikle fark edilmez, çünkü tanı için kan tahlili gerekir. Bu proje, **yalnızca anket soruları**
ile ne kadar ileri gidilebileceğini ve sonucun bugün hekimlerin kullandığı klinik tarama araçlarıyla nasıl
karşılaştığını inceler.

- **1. aşama: genel risk modeli (CDC BRFSS 2015).** 253.680 kişi ve 21 soru üzerinde 8 model eğitilip karşılaştırılır.
- **2. aşama: teşhis edilmemiş diyabet (CDC NHANES 2007-2020).** Kan tahlilini hiç görmeyen bir model, tahlili
  diyabet çıkan ama henüz hekimin tanı koymadığı kişileri bulmaya çalışır ve ADA Risk Testi ile FINDRISC ile yarışır.
- **Web uygulaması** modeli SHAP açıklamalarıyla sunar; her tahminde hangi cevapların riski artırdığını veya azalttığını gösterir.

## Sonuçlar

Tüm sayılar [`backend/reports/`](backend/reports) içindeki JSON dosyalarından gelir ve aşağıdaki betiklerle yeniden üretilebilir.

**1. aşama: BRFSS 2015 (ROC-AUC, `model_sonuclari.json`)**

| Model | ROC-AUC |
|---|---|
| XGBoost (gradyan artırma) | 0,8176 |
| Yapay sinir ağı (Keras MLP) | 0,8171 |
| Random Forest | 0,8156 |
| AdaBoost | 0,8141 |
| SVM (RBF çekirdek) | 0,8121 |
| Lojistik Regresyon | 0,8097 |
| Karar Ağacı | 0,8062 |
| KNN | 0,8014 |

Gelir, eğitim, sağlık sigortası ve "parasızlıktan doktora gidememe" soruları son soru setinden çıkarıldı (17 soru
kaldı). Diyabetle ilişkililer ama etkileri zaten kilo, tansiyon ve genel sağlık cevaplarına yansıyor;
`model_sonuclari.json` içindeki ablasyon, AUC'ye yalnızca yaklaşık 0,001 katkı yaptıklarını gösteriyor.

**2. aşama: NHANES, 2007-2016 ile eğitildi, hiç görülmemiş 2017-2020 verisinde test edildi (n = 6.650)**

| Yöntem | ROC-AUC |
|---|---|
| **DiyabetAI (lojistik regresyon)** | **0,809** |
| FINDRISC | 0,769 |
| ADA Risk Testi | 0,751 |

Altı geliştirme adayı (XGBoost, spline, HbA1c regresyonu, topluluk modeli, beslenme verisi ve iteratif doldurma)
sıkı bir kabul kuralıyla (+0,003 AUC ve 5 katın 4'ünde üstünlük) denendi. Hiçbiri geçemedi, bu yüzden uygulama
modeli kalibre edilmiş lojistik regresyon olarak kaldı.

## Özellikler

- GridSearchCV, K-Fold çapraz doğrulama ve Keras sinir ağıyla 8 modelin karşılaştırması
- SHAP açıklamaları, K-Means ve hiyerarşik kümeleme, PCA ve t-SNE görselleştirmeleri
- FastAPI servisi: `/api/tahmin`, `/api/sonuclar`, `/api/meta`
- Derleme gerektirmeyen Türkçe arayüz (HTML, CSS, JavaScript)
- **Tarayıcıda çalışan statik sürüm** (`frontend/statik.js`): XGBoost ağaçları, TreeSHAP, kalibre lojistik regresyon,
  ADA ve FINDRISC skorları JavaScript'e aktarıldı, böylece site statik barındırmada çalışır.
  `tests/test_statik.py`, Python API'siyle aynı sonucu verdiğini doğrular
- Otomatik testler (`tests/`) ve CI

## Proje yapısı

```
backend/
  ml/        veri indirme, keşifsel analiz, model eğitimi, SHAP ve kümeleme, NHANES hattı
  api/       FastAPI uygulaması
  models/    eğitilmiş modeller (küçük olanlar dahil)
  reports/   JSON sonuçlar ve grafikler
frontend/    web uygulaması (index.html, app.js, statik.js, styles.css)
tests/       API ve statik sürüm testleri
```

## Hızlı başlangıç

```bash
git clone https://github.com/yusufgunduz-dev/diyabet-risk-ai.git
cd diyabet-risk-ai
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
```

Web uygulamasını çalıştırın:

```bash
.venv/Scripts/python -m uvicorn backend.api.main:app --port 8000
# http://localhost:8000 adresini açın
```

Modelleri baştan eğitmek için (veri setleri indirilir, depoda saklanmaz):

```bash
.venv/Scripts/python -m backend.ml.veri_indir        # UCI'dan BRFSS
.venv/Scripts/python -m backend.ml.kesif_analizi     # keşifsel analiz
.venv/Scripts/python -m backend.ml.model_egitimi     # 8 model
.venv/Scripts/python -m backend.ml.aciklama_kumeleme # SHAP ve kümeleme
.venv/Scripts/python -m backend.ml.nhanes_indir      # 84 NHANES dosyası, yaklaşık 170 MB
.venv/Scripts/python -m backend.ml.nhanes_hazirla
.venv/Scripts/python -m backend.ml.nhanes_model
.venv/Scripts/python -m backend.ml.nhanes_uygulama
```

Testleri çalıştırın:

```bash
.venv/Scripts/python -m pytest tests -q
```

## Veri kaynakları

| Veri seti | Kaynak |
|---|---|
| CDC BRFSS 2015 (Diabetes Health Indicators) | [UCI Machine Learning Repository #891](https://archive.ics.uci.edu/dataset/891) |
| CDC NHANES 2007-2020 | [CDC National Center for Health Statistics](https://www.cdc.gov/nchs/nhanes/) |

Veriler bu depoda dağıtılmaz. Kullanmadan önce her kaynağın koşullarını okuyun.
Tam proje raporu için [RAPOR.md](RAPOR.md) dosyasına bakın.

## Katkı ve güvenlik

[CONTRIBUTING.md](CONTRIBUTING.md) ve [SECURITY.md](SECURITY.md) dosyalarına bakın.

## Lisans

[MIT](LICENSE) © 2026 Yusuf Gündüz. Makine Öğrenmesi dersi dönem projesi olarak geliştirilmiştir.

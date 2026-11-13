"""NHANES 4. adım: Web uygulamasında kullanılacak "gizli diyabet" modelini hazırlar.

Araştırma modelinden (nhanes_model.py) farkları:
  - ABD'ye özgü sorular çıkarıldı (etnik köken, ABD yoksulluk sınırına göre gelir) ve
    yalnızca içenlere sorulan alkol sorusu çıkarıldı. Böylece Türkiye'deki bir kullanıcı da
    tüm soruları cevaplayabilir.
  - Ölçüm soruları (bel çevresi, tansiyon) isteğe bağlıdır; boş bırakılırsa medyanla doldurulur.
  - Olasılıklar izotonik regresyonla kalibre edilir (5-katlı CalibratedClassifierCV).
  - Uyarı eşiği, ADA'nın "5 puan ve üzeri" kuralıyla aynı yanlış alarm oranını verecek şekilde
    eğitim verisinde seçilir; böylece iki test adil biçimde karşılaştırılır.
  - Türkiye kalibrasyonu: olasılıklar, Türkiye'deki gizli diyabet yaygınlığına göre önsel olasılık
    kaydırmasıyla (Saerens ve ark., 2002) yeniden ölçeklenir. Sıralama ve kararlar değişmez.

Çalıştırma (nhanes_hazirla'dan sonra):
    .venv/Scripts/python -m backend.ml.nhanes_uygulama
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.ml.nhanes_hazirla import CIKTI, NHANES_OZELLIKLER
from backend.ml.ozellikler import GRAFIK_DIR, MODEL_DIR, RAPOR_DIR, RASTGELE

HEDEF = "tani_konmamis_diyabet"
TEST_DONEMI = "2017-20"
CIKARILAN = ["etnik", "gelir_orani", "alkol_gunluk", "bel_boy_orani"]
UYGULAMA_OZELLIKLERI = [o for o in NHANES_OZELLIKLER if o not in CIKARILAN] + ["bel_boy_orani"]

# IDF Diyabet Atlası, Türkiye ülke profili, 2024 (20-79 yaş, bin kişi)
# https://diabetesatlas.org/data-by-location/country/trkiye/
TR_KAYNAK = "IDF Diyabet Atlası, Türkiye, 2024"
TR_YETISKIN = 58_964.3
TR_DIYABETLI = 9_603.1
TR_TANISIZ = 4_369.4      # diyabetlilerin %45,5'i
# Tanısı olmayan yetişkinler içinde gizli diyabet oranı (modelin hedef kitlesiyle aynı tanım)
TR_GIZLI_ORAN = TR_TANISIZ / (TR_YETISKIN - (TR_DIYABETLI - TR_TANISIZ))


def odds(p):
    return p / (1 - p)


def turkiyeye_kaydir(p, kaynak_oran, hedef_oran=TR_GIZLI_ORAN):
    """Önsel olasılık kaydırma: modelin olabilirlik oranları korunur, taban oran değiştirilir."""
    k = odds(hedef_oran) / odds(kaynak_oran)
    p = np.asarray(p, dtype=float)
    return p * k / (p * k + (1 - p))

EH = [{"deger": 0, "etiket": "Hayır"}, {"deger": 1, "etiket": "Evet"}]
# Frontend formu için soru tanımları (bmi ve bel_boy_orani tarayıcıda/API'de hesaplanır)
SORULAR = [
    {"ad": "erkek", "grup": "Temel bilgiler", "soru": "Cinsiyetiniz", "tip": "secim", "secenekler": [{"deger": 0, "etiket": "Kadın"}, {"deger": 1, "etiket": "Erkek"}]},
    {"ad": "yas", "grup": "Temel bilgiler", "soru": "Yaşınız", "tip": "sayi", "min": 20, "max": 80, "varsayilan": 45},
    {"ad": "boy", "grup": "Temel bilgiler", "soru": "Boy (cm)", "tip": "sayi", "min": 130, "max": 210, "varsayilan": 170},
    {"ad": "kilo", "grup": "Temel bilgiler", "soru": "Kilo (kg)", "tip": "sayi", "min": 35, "max": 220, "varsayilan": 75},
    {"ad": "egitim", "grup": "Temel bilgiler", "soru": "Eğitim düzeyiniz", "tip": "secim", "secenekler": [
        {"deger": 1, "etiket": "İlkokul / ortaokul"}, {"deger": 2, "etiket": "Lise terk"}, {"deger": 3, "etiket": "Lise mezunu"},
        {"deger": 4, "etiket": "Ön lisans / üniversite terk"}, {"deger": 5, "etiket": "Üniversite mezunu"}]},
    {"ad": "aile_diyabet", "grup": "Sağlık geçmişi", "soru": "Anne, baba veya kardeşlerinizde diyabet var mı?", "tip": "secim", "secenekler": EH},
    {"ad": "prediyabet_tani", "grup": "Sağlık geçmişi", "soru": "Doktor size hiç şekerinizin yüksek veya sınırda olduğunu söyledi mi?", "tip": "secim", "secenekler": EH},
    {"ad": "gebelik_diyabeti", "grup": "Sağlık geçmişi", "soru": "Hamileliğinizde gebelik şekeri çıktı mı?", "tip": "secim", "secenekler": EH, "sadece_kadin": True},
    {"ad": "tansiyon_tani", "grup": "Sağlık geçmişi", "soru": "Doktor size yüksek tansiyon tanısı koydu mu?", "tip": "secim", "secenekler": EH},
    {"ad": "tansiyon_ilac", "grup": "Sağlık geçmişi", "soru": "Tansiyon ilacı kullanıyor musunuz?", "tip": "secim", "secenekler": EH, "kosullu": True},
    {"ad": "kolesterol_tani", "grup": "Sağlık geçmişi", "soru": "Doktor size yüksek kolesterol dedi mi?", "tip": "secim", "secenekler": EH},
    {"ad": "kalp_hastaligi", "grup": "Sağlık geçmişi", "soru": "Kalp hastalığı veya kalp krizi geçirdiniz mi?", "tip": "secim", "secenekler": EH},
    {"ad": "inme", "grup": "Sağlık geçmişi", "soru": "Hiç inme (felç) geçirdiniz mi?", "tip": "secim", "secenekler": EH},
    {"ad": "genel_saglik", "grup": "Sağlık geçmişi", "soru": "Genel sağlığınızı nasıl değerlendirirsiniz?", "tip": "secim", "secenekler": [
        {"deger": 1, "etiket": "Mükemmel"}, {"deger": 2, "etiket": "Çok iyi"}, {"deger": 3, "etiket": "İyi"}, {"deger": 4, "etiket": "Orta"}, {"deger": 5, "etiket": "Kötü"}]},
    {"ad": "agir_aktivite", "grup": "Yaşam tarzı", "soru": "Haftada en az 10 dakika sizi nefes nefese bırakan ağır aktivite (iş veya spor) yapar mısınız?", "tip": "secim", "secenekler": EH},
    {"ad": "orta_aktivite", "grup": "Yaşam tarzı", "soru": "Tempolu yürüyüş gibi orta düzeyde aktivite yapar mısınız?", "tip": "secim", "secenekler": EH},
    {"ad": "oturma_dk", "grup": "Yaşam tarzı", "soru": "Günde kaç saat oturarak geçiriyorsunuz?", "tip": "sayi", "min": 0, "max": 20, "varsayilan": 6, "carpan": 60, "birim": "saat"},
    {"ad": "uyku_saat", "grup": "Yaşam tarzı", "soru": "Günde ortalama kaç saat uyuyorsunuz?", "tip": "sayi", "min": 3, "max": 14, "varsayilan": 7, "birim": "saat"},
    {"ad": "sigara_hic", "grup": "Yaşam tarzı", "soru": "Hayatınız boyunca en az 100 sigara içtiniz mi?", "tip": "secim", "secenekler": EH},
    {"ad": "sigara_simdi", "grup": "Yaşam tarzı", "soru": "Halen sigara içiyor musunuz?", "tip": "secim", "secenekler": EH, "kosullu": True},
    {"ad": "bel", "grup": "Ölçümler (isteğe bağlı)", "soru": "Bel çevresi (cm, göbek hizasından)", "tip": "sayi", "min": 50, "max": 180, "istege_bagli": True},
    {"ad": "sistolik", "grup": "Ölçümler (isteğe bağlı)", "soru": "Büyük tansiyon (ör. 120)", "tip": "sayi", "min": 70, "max": 230, "istege_bagli": True},
    {"ad": "diastolik", "grup": "Ölçümler (isteğe bağlı)", "soru": "Küçük tansiyon (ör. 80)", "tip": "sayi", "min": 30, "max": 140, "istege_bagli": True},
]


FINDRISC_ESIK = 15  # standart "yüksek risk" kategorisi (15-20 puan)
ETNIK_AD = {1: "Meksika kökenli", 2: "Diğer Hispanik", 3: "Beyaz", 4: "Siyah", 5: "Diğer / çok ırklı"}


def esit_alarmda_yakalanan(skor, y, yanlis_alarm):
    """Verilen yanlış alarm sayısına izin verildiğinde skorun yakaladığı hasta sayısı."""
    esik = np.sort(skor[y == 0])[::-1][yanlis_alarm - 1]
    return int(((skor >= esik) & (y == 1)).sum())


def bootstrap(y, skorlar, n=2000):
    """AUC'ler ve modelin rakiplere göre farkı için %95 güven aralıkları."""
    rs = np.random.RandomState(RASTGELE)
    kayit = {a: [] for a in skorlar}
    for _ in range(n):
        i = rs.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        for a, s in skorlar.items():
            kayit[a].append(roc_auc_score(y[i], s[i]))
    k = {a: np.array(v) for a, v in kayit.items()}
    ara = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {**{a: ara(v) for a, v in k.items()},
            "model_eksi_ada": ara(k["model"] - k["ada"]), "model_eksi_findrisc": ara(k["model"] - k["findrisc"])}


def adalet_analizi(test, y, skorlar):
    """Model ve klinik testlerin alt gruplardaki başarısı: herkes için eşit derecede iyi mi?"""
    gruplar = {
        "Cinsiyet": test["erkek"].map({0: "Kadın", 1: "Erkek"}),
        "Yaş": pd.cut(test["yas"], [0, 44.99, 64.99, 200], labels=["20-44", "45-64", "65+"]).astype(str),
        "Gelir": pd.cut(test["gelir_orani"], [-1, 1.3, 3.5, 99], labels=["Düşük", "Orta", "Yüksek"]).astype(str),
        "Etnik köken": test["etnik"].map(ETNIK_AD),
    }
    satirlar = []
    for boyut, etiket in gruplar.items():
        for deger in sorted(etiket.dropna().unique()):
            if deger == "nan":
                continue
            m = (etiket == deger).values
            if y[m].sum() < 20:
                continue
            satirlar.append({"boyut": boyut, "grup": deger, "n": int(m.sum()), "hasta": int(y[m].sum()),
                             **{a: float(roc_auc_score(y[m], s[m])) for a, s in skorlar.items()}})

    tablo = pd.DataFrame(satirlar)
    fig, ax = plt.subplots(figsize=(10, 5))
    konum = np.arange(len(tablo))
    for k, (a, renk) in enumerate([("Yapay zekâ", "#1f4e79"), ("FINDRISC", "#7f8c99"), ("ADA", "#c5ccd3")]):
        ax.bar(konum + (k - 1) * 0.27, tablo[a], 0.27, label=a, color=renk)
    ax.set_xticks(konum, [f"{r.grup}\n(n={r.hasta})" for r in tablo.itertuples()], fontsize=8)
    sinir = 0
    for boyut in tablo["boyut"].unique():
        say = (tablo["boyut"] == boyut).sum()
        ax.text(sinir + (say - 1) / 2, 0.93, boyut, ha="center", fontsize=9, fontweight="bold")
        sinir += say
        if sinir < len(tablo):
            ax.axvline(sinir - 0.5, color="#ccc", lw=0.8)
    ax.set_ylim(0.5, 0.97)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Adalet analizi: alt gruplarda başarı (n = gruptaki gizli hasta sayısı)")
    ax.legend(loc="lower right", ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(GRAFIK_DIR / "34_adalet_analizi.png", dpi=150)
    plt.close()
    return satirlar


def model_kur():
    return Pipeline([
        ("doldur", SimpleImputer(strategy="median")),
        ("olcek", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])


def main():
    df = pd.read_csv(CIKTI, index_col=0)
    egitim, test = df[df["donem"] != TEST_DONEMI], df[df["donem"] == TEST_DONEMI]
    Xe, ye = egitim[UYGULAMA_OZELLIKLERI], egitim[HEDEF].values
    Xt, yt = test[UYGULAMA_OZELLIKLERI], test[HEDEF].values
    kfold = StratifiedKFold(5, shuffle=True, random_state=RASTGELE)

    arama = GridSearchCV(model_kur(), {"model__C": [0.003, 0.01, 0.03, 0.1, 1]}, scoring="roc_auc", cv=kfold, n_jobs=-1).fit(Xe, ye)
    ham = arama.best_estimator_
    kalibre = CalibratedClassifierCV(model_kur().set_params(**arama.best_params_), method="isotonic", cv=kfold).fit(Xe, ye)

    # ADA'nın eğitim verisindeki özgüllüğü ile aynı özgüllüğü veren eşik
    oof = cross_val_predict(CalibratedClassifierCV(model_kur().set_params(**arama.best_params_), method="isotonic", cv=3),
                            Xe, ye, cv=kfold, method="predict_proba")[:, 1]
    ada_ozgulluk = float((egitim["ada_puani"] < 5)[ye == 0].mean())
    esik = float(np.quantile(oof[ye == 0], ada_ozgulluk))

    pt = kalibre.predict_proba(Xt)[:, 1]
    ada_t = test["ada_puani"].values
    fin_t = test["findrisc_puani"].values
    # Test setinde ADA ile birebir aynı yanlış alarm sayısında modelin yakaladığı hasta sayısı
    ada_yanlis = int(((ada_t >= 5) & (yt == 0)).sum())
    esit_yakalanan = esit_alarmda_yakalanan(pt, yt, ada_yanlis)
    fin_yanlis = int(((fin_t >= FINDRISC_ESIK) & (yt == 0)).sum())
    bs = bootstrap(yt, {"model": pt, "ada": ada_t, "findrisc": fin_t})
    adalet = adalet_analizi(test, yt, {"Yapay zekâ": pt, "ADA": ada_t, "FINDRISC": fin_t})
    sonuc = {
        "model_auc": float(roc_auc_score(yt, pt)),
        "ada_auc": float(roc_auc_score(yt, ada_t)),
        "findrisc_auc": float(roc_auc_score(yt, fin_t)),
        "guven_araliklari": bs,
        "findrisc_esik": FINDRISC_ESIK,
        "findrisc_yakalanan": int(((fin_t >= FINDRISC_ESIK) & (yt == 1)).sum()),
        "findrisc_yanlis_alarm": fin_yanlis,
        "model_findrisc_alarminda_yakalanan": esit_alarmda_yakalanan(pt, yt, fin_yanlis),
        "adalet": adalet,
        "cv_auc": float(arama.best_score_),
        "C": arama.best_params_["model__C"],
        "esik": esik,
        "ada_ozgulluk_egitim": ada_ozgulluk,
        "test_n": int(len(yt)),
        "test_hasta": int(yt.sum()),
        "ada_yakalanan": int(((ada_t >= 5) & (yt == 1)).sum()),
        "ada_yanlis_alarm": ada_yanlis,
        "model_esit_alarmda_yakalanan": esit_yakalanan,
        "model_yakalanan": int(((pt >= esik) & (yt == 1)).sum()),
        "model_yanlis_alarm": int(((pt >= esik) & (yt == 0)).sum()),
        "taban_oran": float(ye.mean()),
        "turkiye": {
            "kaynak": TR_KAYNAK,
            "gizli_oran": TR_GIZLI_ORAN,
            "tanisiz_payi": TR_TANISIZ / TR_DIYABETLI,
            "tanisiz_kisi_bin": TR_TANISIZ,
            "katsayi": odds(TR_GIZLI_ORAN) / odds(float(ye.mean())),
            "esik": float(turkiyeye_kaydir(esik, float(ye.mean()))),
            # Monoton dönüşüm olduğu için AUC aynı kalmalı (kontrol)
            "test_auc": float(roc_auc_score(yt, turkiyeye_kaydir(pt, float(ye.mean())))),
        },
    }
    print(json.dumps(sonuc, indent=2, ensure_ascii=False))

    joblib.dump({"kalibre": kalibre, "ham": ham, "ozellikler": UYGULAMA_OZELLIKLERI, "esik": esik, "sonuc": sonuc},
                MODEL_DIR / "nhanes_uygulama.joblib")
    with open(RAPOR_DIR / "nhanes_uygulama.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, indent=2, ensure_ascii=False)
    print("Kaydedildi: backend/models/nhanes_uygulama.joblib")


if __name__ == "__main__":
    main()

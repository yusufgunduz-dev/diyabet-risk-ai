"""2. adım: Keşifsel veri analizi (EDA) - Hafta 2.

Veri kalitesini kontrol eder (eksik / tekrar eden / aykırı değerler) ve sunum
için grafikleri backend/reports/figures altına kaydeder.

Çalıştırma:
    .venv/Scripts/python -m backend.ml.kesif_analizi
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend.ml.ozellikler import (
    GRAFIK_DIR, HEDEF, RAPOR_DIR, TURKCE_ETIKET, VERI_DOSYASI, YAS_GRUPLARI,
)

sns.set_theme(style="whitegrid", font_scale=1.05)
RENKLER = ["#4C72B0", "#DD5555"]


def kaydet(ad):
    plt.tight_layout()
    plt.savefig(GRAFIK_DIR / ad, dpi=150)
    plt.close()


def main():
    GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(VERI_DOSYASI)

    # --- Veri kalitesi ---
    q1, q3 = df["BMI"].quantile([0.25, 0.75])
    iqr = q3 - q1
    bmi_aykiri = int(((df["BMI"] < q1 - 1.5 * iqr) | (df["BMI"] > q3 + 1.5 * iqr)).sum())
    ozet = {
        "satir": int(len(df)),
        "ozellik": int(df.shape[1] - 1),
        "eksik_deger": int(df.isna().sum().sum()),
        "tekrar_eden_satir": int(df.duplicated().sum()),
        "diyabet_orani": float(df[HEDEF].mean()),
        "bmi_aykiri_iqr": bmi_aykiri,
        "bmi_min": float(df["BMI"].min()),
        "bmi_max": float(df["BMI"].max()),
    }
    print(json.dumps(ozet, indent=2, ensure_ascii=False))

    # 1) Sınıf dağılımı
    plt.figure(figsize=(6, 4))
    sayim = df[HEDEF].value_counts().sort_index()
    ax = sns.barplot(x=["Diyabet yok", "Diyabet / Prediyabet"], y=sayim.values, palette=RENKLER, hue=["0", "1"], legend=False)
    for i, v in enumerate(sayim.values):
        ax.text(i, v, f"{v:,}\n(%{100 * v / len(df):.1f})", ha="center", va="bottom")
    plt.title("Sınıf dağılımı: veri dengesiz")
    plt.ylabel("Kişi sayısı")
    plt.ylim(0, sayim.max() * 1.18)
    kaydet("01_sinif_dagilimi.png")

    # 2) BMI dağılımı (sınıfa göre)
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df[df["BMI"] < 60], x="BMI", hue=HEDEF, common_norm=False, fill=True, palette=RENKLER, alpha=0.4)
    plt.title("Vücut kitle indeksi dağılımı")
    plt.legend(title="", labels=["Diyabet / Prediyabet", "Diyabet yok"])
    kaydet("02_bmi_dagilimi.png")

    # 3) Yaş grubuna göre diyabet oranı
    plt.figure(figsize=(9, 4))
    oran = df.groupby("Age")[HEDEF].mean() * 100
    sns.barplot(x=[YAS_GRUPLARI[int(a) - 1] for a in oran.index], y=oran.values, color="#DD5555")
    plt.title("Yaş grubuna göre diyabet oranı (%)")
    plt.xlabel("Yaş grubu")
    plt.ylabel("Diyabet oranı (%)")
    plt.xticks(rotation=45)
    kaydet("03_yas_diyabet.png")

    # 4) Risk faktörlerinin etkisi
    faktorler = ["HighBP", "HighChol", "HeartDiseaseorAttack", "DiffWalk", "Stroke", "PhysActivity", "Smoker", "HvyAlcoholConsump"]
    satirlar = []
    for f in faktorler:
        for deger, ad in [(0, "Hayır"), (1, "Evet")]:
            satirlar.append({"Faktör": TURKCE_ETIKET[f], "Durum": ad, "Oran": df.loc[df[f] == deger, HEDEF].mean() * 100})
    plt.figure(figsize=(10, 4.5))
    sns.barplot(data=pd.DataFrame(satirlar), x="Faktör", y="Oran", hue="Durum", palette=["#8C9BB0", "#DD5555"])
    plt.title("Risk faktörü var / yok iken diyabet oranı (%)")
    plt.ylabel("Diyabet oranı (%)")
    plt.xlabel("")
    plt.xticks(rotation=20)
    kaydet("04_risk_faktorleri.png")

    # 5) Genel sağlık algısı
    plt.figure(figsize=(7, 4))
    oran = df.groupby("GenHlth")[HEDEF].mean() * 100
    sns.barplot(x=["Mükemmel", "Çok iyi", "İyi", "Orta", "Kötü"], y=oran.values, color="#DD5555")
    plt.title("Genel sağlık algısına göre diyabet oranı (%)")
    plt.ylabel("Diyabet oranı (%)")
    kaydet("05_genel_saglik.png")

    # 6) Korelasyon ısı haritası
    plt.figure(figsize=(12, 10))
    kor = df.rename(columns={**TURKCE_ETIKET, HEDEF: "DİYABET"}).corr()
    sns.heatmap(kor, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
    plt.title("Korelasyon matrisi")
    kaydet("06_korelasyon.png")

    # 7) Hedefle korelasyon sıralaması
    plt.figure(figsize=(7, 7))
    hk = df.corr()[HEDEF].drop(HEDEF).sort_values()
    sns.barplot(x=hk.values, y=[TURKCE_ETIKET[i] for i in hk.index], palette=["#DD5555" if v > 0 else "#4C72B0" for v in hk.values], hue=hk.index, legend=False)
    plt.title("Diyabetle korelasyon")
    plt.xlabel("Pearson korelasyonu")
    kaydet("07_hedef_korelasyon.png")

    # 8) BMI kutu grafiği (aykırı değerler)
    plt.figure(figsize=(7, 3))
    sns.boxplot(x=df["BMI"], color="#4C72B0")
    plt.title(f"BMI kutu grafiği - IQR yöntemine göre {bmi_aykiri:,} aykırı değer")
    kaydet("08_bmi_aykiri.png")

    RAPOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAPOR_DIR / "veri_ozeti.json", "w", encoding="utf-8") as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False)
    print(f"Grafikler: {GRAFIK_DIR}")


if __name__ == "__main__":
    main()

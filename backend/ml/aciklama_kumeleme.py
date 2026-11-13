"""4. adım: Açıklanabilirlik (SHAP), kümeleme ve boyut indirgeme - Hafta 9-10.

  - SHAP: modelin hangi özelliğe ne kadar önem verdiğini gösterir.
  - K-Means: dirsek + siluet yöntemleriyle k seçer, kişileri risk profillerine ayırır.
  - Hiyerarşik kümeleme: dendrogram ve K-Means ile siluet karşılaştırması.
  - PCA ve t-SNE: 17 boyutlu veriyi 2 boyutta görselleştirir.

Çalıştırma (model_egitimi'nden sonra):
    .venv/Scripts/python -m backend.ml.aciklama_kumeleme
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from backend.ml.ozellikler import (
    GRAFIK_DIR, MODEL_DIR, OZELLIK_ADLARI, RAPOR_DIR, RASTGELE, TURKCE_ETIKET,
)

sns.set_theme(style="whitegrid", font_scale=1.05)
KUME_ORNEK = 20_000


def kaydet(ad):
    plt.tight_layout()
    plt.savefig(GRAFIK_DIR / ad, dpi=150, bbox_inches="tight")
    plt.close()


def shap_analizi(model, X_test):
    print("SHAP analizi...")
    ornek = X_test.sample(3000, random_state=RASTGELE)
    aciklayici = shap.TreeExplainer(model)
    degerler = aciklayici(ornek)
    degerler.feature_names = [TURKCE_ETIKET[o] for o in OZELLIK_ADLARI]

    plt.figure()
    shap.plots.beeswarm(degerler, max_display=15, show=False)
    plt.title("SHAP: her özelliğin tahmine etkisi")
    kaydet("20_shap_ozet.png")

    plt.figure()
    shap.plots.bar(degerler, max_display=15, show=False)
    plt.title("SHAP: ortalama etki büyüklüğü")
    kaydet("21_shap_bar.png")

    ortalama = pd.Series(np.abs(degerler.values).mean(axis=0), index=OZELLIK_ADLARI).sort_values(ascending=False)
    return [{"ozellik": o, "etiket": TURKCE_ETIKET[o], "ortalama_shap": float(v)} for o, v in ortalama.items()]


def kume_adi(profil, genel):
    """Kümenin en belirgin özelliklerinden okunabilir bir isim üretir."""
    parcalar = []
    if profil["Age"] >= genel["Age"] + 1.5:
        parcalar.append("ileri yaş")
    elif profil["Age"] <= genel["Age"] - 1.5:
        parcalar.append("genç")
    if profil["GenHlth"] >= genel["GenHlth"] + 0.6:
        parcalar.append("sağlığı kötü")
    elif profil["GenHlth"] <= genel["GenHlth"] - 0.4:
        parcalar.append("sağlığı iyi")
    if profil["HighBP"] >= genel["HighBP"] + 0.15:
        parcalar.append("tansiyonlu")
    if profil["BMI"] >= genel["BMI"] + 2:
        parcalar.append("kilolu")
    if profil["PhysActivity"] <= genel["PhysActivity"] - 0.15:
        parcalar.append("hareketsiz")
    if profil["DiffWalk"] >= genel["DiffWalk"] + 0.2:
        parcalar.append("yürüme güçlüğü")
    if not parcalar:
        return "Ortalama profil"
    ad = ", ".join(parcalar[:3])
    ilk = {"i": "İ", "ı": "I"}.get(ad[0], ad[0].upper())  # Türkçe büyük harf
    return ilk + ad[1:]


def kumeleme(X, y):
    print("Kümeleme...")
    Xk = X.sample(KUME_ORNEK, random_state=RASTGELE)
    yk = y.loc[Xk.index]
    olcek = StandardScaler().fit(Xk)
    Z = olcek.transform(Xk)
    sil_idx = np.random.RandomState(RASTGELE).choice(len(Z), 5000, replace=False)

    k_degerleri = list(range(2, 9))
    atalet, siluet = [], []
    for k in k_degerleri:
        km = KMeans(n_clusters=k, n_init=10, random_state=RASTGELE).fit(Z)
        atalet.append(float(km.inertia_))
        siluet.append(float(silhouette_score(Z[sil_idx], km.labels_[sil_idx])))
    fig, eksen = plt.subplots(1, 2, figsize=(11, 4))
    eksen[0].plot(k_degerleri, atalet, "o-")
    eksen[0].set_title("Dirsek yöntemi (inertia)")
    eksen[0].set_xlabel("Küme sayısı (k)")
    eksen[1].plot(k_degerleri, siluet, "o-", color="#DD5555")
    eksen[1].set_title("Siluet skoru")
    eksen[1].set_xlabel("Küme sayısı (k)")
    kaydet("22_kmeans_k_secimi.png")

    # Siluet skorları bu veride düşük ve birbirine yakın; yorumlanabilirlik için k=4.
    K = 4
    km = KMeans(n_clusters=K, n_init=10, random_state=RASTGELE).fit(Z)
    etiket = km.labels_
    hiy = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Z[sil_idx])
    karsilastirma = {
        "kmeans_siluet": float(silhouette_score(Z[sil_idx], etiket[sil_idx])),
        "hiyerarsik_siluet": float(silhouette_score(Z[sil_idx], hiy.labels_)),
    }

    genel = Xk.mean()
    profiller = []
    for c in range(K):
        maske = etiket == c
        p = Xk[maske].mean()
        profiller.append({
            "kume": c, "ad": kume_adi(p, genel), "kisi_orani": float(maske.mean()), "diyabet_orani": float(yk[maske].mean()),
            "ortalama": {o: float(p[o]) for o in ["BMI", "Age", "GenHlth", "HighBP", "HighChol", "PhysActivity", "DiffWalk", "HeartDiseaseorAttack"]},
        })
    profiller.sort(key=lambda p: p["diyabet_orani"])
    for p in profiller:
        print(f"  Küme {p['kume']}: {p['ad']:40s} kişi %{100 * p['kisi_orani']:.0f}  diyabet %{100 * p['diyabet_orani']:.1f}")

    plt.figure(figsize=(8, 4))
    sns.barplot(x=[f"{p['ad']}\n(%{100 * p['kisi_orani']:.0f} kişi)" for p in profiller], y=[100 * p["diyabet_orani"] for p in profiller],
                palette="Reds", hue=[p["ad"] for p in profiller], legend=False)
    plt.axhline(100 * y.mean(), ls="--", color="gray", label="Genel ortalama")
    plt.ylabel("Diyabet oranı (%)")
    plt.title("K-Means risk profilleri")
    plt.legend()
    plt.xticks(fontsize=8)
    kaydet("23_kume_profilleri.png")

    # PCA
    pca = PCA(n_components=2, random_state=RASTGELE).fit(Z)
    P = pca.transform(Z)
    pca_tam = PCA(random_state=RASTGELE).fit(Z)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(OZELLIK_ADLARI) + 1), np.cumsum(pca_tam.explained_variance_ratio_) * 100, "o-")
    plt.axhline(90, ls="--", color="gray")
    plt.xlabel("Bileşen sayısı")
    plt.ylabel("Açıklanan varyans (%)")
    plt.title("PCA: kümülatif açıklanan varyans")
    kaydet("24_pca_varyans.png")

    goster = np.random.RandomState(RASTGELE).choice(len(P), 6000, replace=False)
    ad_haritasi = {p["kume"]: p["ad"] for p in profiller}
    fig, eksen = plt.subplots(1, 2, figsize=(13, 5))
    for c in range(K):
        m = etiket[goster] == c
        eksen[0].scatter(P[goster][m, 0], P[goster][m, 1], s=4, alpha=0.5, label=ad_haritasi[c])
    eksen[0].set_title("PCA - K-Means kümeleri")
    eksen[0].legend(markerscale=4, fontsize=8)
    renk = yk.values[goster]
    eksen[1].scatter(P[goster][renk == 0, 0], P[goster][renk == 0, 1], s=4, alpha=0.3, c="#4C72B0", label="Diyabet yok")
    eksen[1].scatter(P[goster][renk == 1, 0], P[goster][renk == 1, 1], s=4, alpha=0.6, c="#DD5555", label="Diyabet var")
    eksen[1].set_title("PCA - gerçek sınıflar")
    eksen[1].legend(markerscale=4)
    for e in eksen:
        e.set_xlabel(f"PC1 (%{100 * pca.explained_variance_ratio_[0]:.1f})")
        e.set_ylabel(f"PC2 (%{100 * pca.explained_variance_ratio_[1]:.1f})")
    kaydet("25_pca_2d.png")

    # t-SNE
    print("t-SNE (3000 örnek)...")
    ts_idx = goster[:3000]
    T = TSNE(n_components=2, perplexity=40, random_state=RASTGELE, init="pca").fit_transform(Z[ts_idx])
    fig, eksen = plt.subplots(1, 2, figsize=(13, 5))
    for c in range(K):
        m = etiket[ts_idx] == c
        eksen[0].scatter(T[m, 0], T[m, 1], s=5, alpha=0.6, label=ad_haritasi[c])
    eksen[0].set_title("t-SNE - K-Means kümeleri")
    eksen[0].legend(markerscale=4, fontsize=8)
    renk = yk.values[ts_idx]
    eksen[1].scatter(T[renk == 0, 0], T[renk == 0, 1], s=5, alpha=0.3, c="#4C72B0", label="Diyabet yok")
    eksen[1].scatter(T[renk == 1, 0], T[renk == 1, 1], s=5, alpha=0.7, c="#DD5555", label="Diyabet var")
    eksen[1].set_title("t-SNE - gerçek sınıflar")
    eksen[1].legend(markerscale=4)
    kaydet("26_tsne_2d.png")

    # Dendrogram
    plt.figure(figsize=(10, 4))
    dendrogram(linkage(Z[sil_idx[:400]], method="ward"), no_labels=True, color_threshold=None)
    plt.title("Hiyerarşik kümeleme dendrogramı (400 kişilik örnek, Ward)")
    plt.ylabel("Uzaklık")
    kaydet("27_dendrogram.png")

    return {
        "k_secimi": {"k": k_degerleri, "inertia": atalet, "siluet": siluet, "secilen_k": K},
        "siluet_karsilastirma": karsilastirma,
        "profiller": profiller,
        "pca_ilk2_varyans": [float(v) for v in pca.explained_variance_ratio_],
        "pca_90_icin_bilesen": int(np.searchsorted(np.cumsum(pca_tam.explained_variance_ratio_), 0.90) + 1),
    }


def main():
    paket = joblib.load(MODEL_DIR / "diyabet_modeli.joblib")
    veri = joblib.load(MODEL_DIR / "bolunmus_veri.joblib")
    sonuc = {"shap_global": shap_analizi(paket["model"], veri["X_test"]), "kumeleme": kumeleme(veri["X_egitim"], veri["y_egitim"])}
    with open(RAPOR_DIR / "aciklama_kumeleme.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, indent=2, ensure_ascii=False)
    print("Bitti.")


if __name__ == "__main__":
    main()

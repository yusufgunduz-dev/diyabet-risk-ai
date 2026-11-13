"""NHANES 3. adım: Teşhis edilmemiş diyabeti tahlilsiz bulmak ve ADA Risk Testi ile yarış.

Tasarım
  - Zamansal doğrulama: 2007-2016 verisiyle eğitilir, hiç görülmemiş 2017-2020 verisinde test edilir.
  - Ablasyon: Hangi bilgi ne kadar katkı sağlıyor?
        ADA testi (kural tabanlı puan)
        -> ADA'nın 7 sorusu + Lojistik Regresyon
        -> Tüm anket soruları
        -> Anket + kan gerektirmeyen ölçümler (bel çevresi, tansiyon)
  - Adil karşılaştırma: Model, ADA'nın "5 puan ve üzeri" kuralıyla AYNI yanlış alarm oranında
    (özgüllükte) kaç hastayı daha fazla yakalıyor?
  - Belirsizlik: 2000 tekrarlı bootstrap ile %95 güven aralıkları.

Çalıştırma:
    .venv/Scripts/python -m backend.ml.nhanes_model
"""

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.ml.nhanes_hazirla import ADA_OZELLIKLER, CIKTI, NHANES_OZELLIKLER
from backend.ml.ozellikler import GRAFIK_DIR, MODEL_DIR, RAPOR_DIR, RASTGELE

sns.set_theme(style="whitegrid", font_scale=1.05)
TEST_DONEMI = "2017-20"
OLCUMLER = ["bel", "bel_boy_orani", "sistolik", "diastolik"]
ANKET = [o for o in NHANES_OZELLIKLER if o not in OLCUMLER]
GOREVLER = {"tani_konmamis_diyabet": "Teşhis edilmemiş diyabet", "disglisemi": "Disglisemi (prediyabet + teşhis edilmemiş diyabet)"}
ETIKET = {**NHANES_OZELLIKLER, "aktif": "Fiziksel olarak aktif"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def kaydet(ad):
    plt.tight_layout()
    plt.savefig(GRAFIK_DIR / ad, dpi=150, bbox_inches="tight")
    plt.close()


def hazirla(df, sutunlar):
    X = df[sutunlar].copy()
    if "etnik" in X:
        X = pd.get_dummies(X, columns=["etnik"], prefix="etnik", dtype=float)
    return X


def bootstrap_auc(y, skorlar, n=2000):
    """Her skor için AUC güven aralığı + ilk skorun ADA'ya (son skor) göre farkı."""
    rs = np.random.RandomState(RASTGELE)
    y = np.asarray(y)
    sonuc = {ad: [] for ad in skorlar}
    for _ in range(n):
        i = rs.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        for ad, s in skorlar.items():
            sonuc[ad].append(roc_auc_score(y[i], np.asarray(s)[i]))
    return {ad: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), np.array(v)) for ad, v in sonuc.items()}


def esdeger_ozgullukte(y, skor, hedef_ozgulluk):
    """Verilen özgüllüğü (1 - yanlış alarm oranı) sağlayan en yüksek duyarlılık."""
    fpr, tpr, esik = roc_curve(y, skor)
    uygun = fpr <= 1 - hedef_ozgulluk
    i = np.where(uygun)[0][-1]
    return float(tpr[i]), float(1 - fpr[i])


def modeller(oran):
    imp = lambda m: Pipeline([("doldur", SimpleImputer(strategy="median")), ("olcek", StandardScaler()), ("model", m)])
    return {
        "Lojistik Regresyon": (imp(LogisticRegression(max_iter=3000, class_weight="balanced")), {"model__C": [0.01, 0.1, 1]}),
        "Random Forest": (Pipeline([("doldur", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=RASTGELE))]),
                          {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [10, 40]}),
        "HistGradientBoosting": (HistGradientBoostingClassifier(class_weight="balanced", random_state=RASTGELE),
                                 {"max_depth": [3, 5], "learning_rate": [0.03, 0.08], "max_iter": [300]}),
        "XGBoost": (XGBClassifier(tree_method="hist", scale_pos_weight=oran, eval_metric="logloss", n_jobs=-1, random_state=RASTGELE),
                    {"max_depth": [2, 3, 4], "n_estimators": [300, 600], "learning_rate": [0.02, 0.05], "subsample": [0.8], "colsample_bytree": [0.8]}),
        "Yapay Sinir Ağı (MLP)": (imp(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=True, max_iter=500, random_state=RASTGELE)),
                                  {"model__alpha": [1e-3, 1e-2, 1e-1]}),
    }


def gorev_calistir(df, hedef):
    log(f"=== Görev: {GOREVLER[hedef]} ===")
    egitim, test = df[df["donem"] != TEST_DONEMI], df[df["donem"] == TEST_DONEMI]
    ye, yt = egitim[hedef].values, test[hedef].values
    log(f"Eğitim {len(egitim):,} (pozitif {ye.sum():,}) | Test {len(test):,} (pozitif {yt.sum():,})")
    kfold = StratifiedKFold(5, shuffle=True, random_state=RASTGELE)
    oran = float((ye == 0).sum() / (ye == 1).sum())
    skorlar, cv, parametre = {}, {}, {}

    # 1) ADA testi: eğitim yok, sadece puan
    skorlar["ADA Risk Testi (doktor testi)"] = test["ada_puani"].values

    # 2) Ablasyon: aynı model (XGBoost) farklı bilgi setleriyle
    setler = {
        "ADA'nın 7 sorusu + ML": ADA_OZELLIKLER,
        "Tüm anket soruları + ML": ANKET,
        "Anket + ölçümler + ML": list(NHANES_OZELLIKLER),
        # Sağlamlık kontrolü: önceki kan şekeri bilgisini hiç kullanmayan model
        "Şeker geçmişi sorusu olmadan": [o for o in NHANES_OZELLIKLER if o != "prediyabet_tani"],
    }
    for ad, sutun in setler.items():
        Xe, Xt = hazirla(egitim, sutun), hazirla(test, sutun)
        xgb, izgara = modeller(oran)["XGBoost"]
        arama = GridSearchCV(xgb, izgara, scoring="roc_auc", cv=kfold, n_jobs=-1).fit(Xe, ye)
        skorlar[ad] = arama.predict_proba(Xt)[:, 1]
        cv[ad] = float(arama.best_score_)
        log(f"  [ablasyon] {ad:28s} CV AUC={arama.best_score_:.4f}  TEST AUC={roc_auc_score(yt, skorlar[ad]):.4f}")

    # 3) Model yarışı: tüm bilgilerle 5 model
    Xe, Xt = hazirla(egitim, list(NHANES_OZELLIKLER)), hazirla(test, list(NHANES_OZELLIKLER))
    yaris, en_iyi = {}, None
    for ad, (m, izgara) in modeller(oran).items():
        arama = GridSearchCV(m, izgara, scoring="roc_auc", cv=kfold, n_jobs=-1).fit(Xe, ye)
        s = arama.predict_proba(Xt)[:, 1]
        yaris[ad] = {"cv_auc": float(arama.best_score_), "test_auc": float(roc_auc_score(yt, s)), "test_pr_auc": float(average_precision_score(yt, s))}
        parametre[ad] = {k: str(v) for k, v in arama.best_params_.items()}
        log(f"  [yarış] {ad:24s} CV AUC={arama.best_score_:.4f}  TEST AUC={yaris[ad]['test_auc']:.4f}")
        if en_iyi is None or arama.best_score_ > en_iyi[1]:
            en_iyi = (ad, arama.best_score_, arama.best_estimator_, s)
    log(f"  En iyi (CV'ye göre): {en_iyi[0]}")
    skorlar["En iyi model: " + en_iyi[0]] = en_iyi[3]

    # 4) Güven aralıkları ve ADA ile adil karşılaştırma
    ada = skorlar["ADA Risk Testi (doktor testi)"]
    ada_pozitif = ada >= 5
    ada_duyarlilik = float(ada_pozitif[yt == 1].mean())
    ada_ozgulluk = float((~ada_pozitif)[yt == 0].mean())
    bs = bootstrap_auc(yt, skorlar)
    ada_bs = bs["ADA Risk Testi (doktor testi)"][2]
    ozet = {}
    for ad, s in skorlar.items():
        alt, ust, dagilim = bs[ad]
        d = {"auc": float(roc_auc_score(yt, s)), "auc_alt": alt, "auc_ust": ust, "pr_auc": float(average_precision_score(yt, s))}
        if not ad.startswith("ADA Risk"):
            n = min(len(dagilim), len(ada_bs))
            fark = dagilim[:n] - ada_bs[:n]
            d["ada_fark"] = float(d["auc"] - roc_auc_score(yt, ada))
            d["ada_fark_alt"], d["ada_fark_ust"] = float(np.percentile(fark, 2.5)), float(np.percentile(fark, 97.5))
            d["ada_ozgullugunde_duyarlilik"], _ = esdeger_ozgullukte(yt, s, ada_ozgulluk)
        ozet[ad] = d
    en_iyi_ad = "En iyi model: " + en_iyi[0]
    yakalanan_ada = int(ada_pozitif[yt == 1].sum())
    yakalanan_model = int(round(ozet[en_iyi_ad]["ada_ozgullugunde_duyarlilik"] * yt.sum()))
    log(f"  ADA (>=5): duyarlılık {ada_duyarlilik:.3f}, özgüllük {ada_ozgulluk:.3f} -> {yakalanan_ada}/{yt.sum()} hasta")
    log(f"  Model aynı özgüllükte: duyarlılık {ozet[en_iyi_ad]['ada_ozgullugunde_duyarlilik']:.3f} -> {yakalanan_model}/{yt.sum()} hasta")

    # --- Grafikler ---
    kisa = {"tani_konmamis_diyabet": "tani", "disglisemi": "disglisemi"}[hedef]
    plt.figure(figsize=(7, 6.5))
    for ad in ["ADA Risk Testi (doktor testi)", "ADA'nın 7 sorusu + ML", "Anket + ölçümler + ML", en_iyi_ad]:
        fpr, tpr, _ = roc_curve(yt, skorlar[ad])
        plt.plot(fpr, tpr, lw=2.5 if ad == en_iyi_ad else 1.5, ls="--" if ad.startswith("ADA Risk") else "-",
                 label=f"{ad} (AUC={ozet[ad]['auc']:.3f})")
    plt.plot(1 - ada_ozgulluk, ada_duyarlilik, "ko", ms=8)
    plt.annotate("ADA kuralı (puan ≥ 5)", (1 - ada_ozgulluk, ada_duyarlilik), xytext=(10, -18), textcoords="offset points")
    plt.plot([0, 1], [0, 1], color="gray", alpha=0.4)
    plt.xlabel("Yanlış alarm oranı (1 - özgüllük)")
    plt.ylabel("Yakalanan hasta oranı (duyarlılık)")
    plt.title(f"{GOREVLER[hedef]}\nYapay zekâ vs ADA Risk Testi (test: {TEST_DONEMI})")
    plt.legend(fontsize=8, loc="lower right")
    kaydet(f"30_nhanes_{kisa}_roc.png")

    adlar = list(ozet)
    plt.figure(figsize=(9, 4.5))
    auc = [ozet[a]["auc"] for a in adlar]
    hata = [[ozet[a]["auc"] - ozet[a]["auc_alt"] for a in adlar], [ozet[a]["auc_ust"] - ozet[a]["auc"] for a in adlar]]
    renk = ["#8C9BB0"] + ["#4C72B0"] * (len(adlar) - 2) + ["#DD5555"]
    plt.barh(adlar, auc, xerr=hata, color=renk, capsize=4)
    for i, a in enumerate(adlar):
        plt.text(ozet[a]["auc_ust"] + 0.008, i, f"{ozet[a]['auc']:.3f}", va="center", fontweight="bold")
    plt.xlim(0.5, 1)
    plt.gca().invert_yaxis()
    plt.xlabel("Test ROC-AUC (%95 güven aralığı)")
    plt.title(f"{GOREVLER[hedef]}: bilgi eklendikçe başarı")
    kaydet(f"31_nhanes_{kisa}_ablasyon.png")

    return {
        "egitim_n": int(len(egitim)), "egitim_pozitif": int(ye.sum()), "test_n": int(len(test)), "test_pozitif": int(yt.sum()),
        "ozet": ozet, "cv_auc": cv, "yaris": yaris, "parametreler": parametre, "en_iyi_model": en_iyi[0],
        "ada_kural": {"duyarlilik": ada_duyarlilik, "ozgulluk": ada_ozgulluk, "yakalanan": yakalanan_ada},
        "model_ayni_ozgullukte_yakalanan": yakalanan_model,
    }, en_iyi[2], Xt


def shap_grafigi(model, X, hedef):
    try:
        ic = model.named_steps["model"] if hasattr(model, "named_steps") else model
        if not hasattr(ic, "get_booster") and not isinstance(ic, (HistGradientBoostingClassifier, RandomForestClassifier)):
            return
        aciklayici = shap.TreeExplainer(ic)
        ornek = X.sample(min(3000, len(X)), random_state=RASTGELE)
        if hasattr(model, "named_steps"):
            ornek = pd.DataFrame(model[:-1].transform(ornek), columns=ornek.columns)
        d = aciklayici(ornek)
        if d.values.ndim == 3:
            d = d[..., 1]
        d.feature_names = [ETIKET.get(c, c.replace("etnik_", "Etnik köken ")) for c in ornek.columns]
        plt.figure()
        shap.plots.beeswarm(d, max_display=14, show=False)
        plt.title(f"SHAP: {GOREVLER[hedef]}")
        kaydet(f"32_nhanes_{'tani' if hedef == 'tani_konmamis_diyabet' else 'disglisemi'}_shap.png")
    except Exception as hata:
        log(f"  SHAP atlandı: {hata}")


def main():
    df = pd.read_csv(CIKTI, index_col=0)
    rapor = {}
    for hedef in GOREVLER:
        sonuc, model, Xt = gorev_calistir(df, hedef)
        rapor[hedef] = sonuc
        shap_grafigi(model, Xt, hedef)
        joblib.dump({"model": model, "ozellikler": list(Xt.columns)}, MODEL_DIR / f"nhanes_{hedef}.joblib")
    with open(RAPOR_DIR / "nhanes_sonuclari.json", "w", encoding="utf-8") as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False)
    log("Bitti.")


if __name__ == "__main__":
    main()

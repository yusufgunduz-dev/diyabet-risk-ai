"""NHANES 5. adım: Geliştirme adaylarını eleme yoluyla seçer.

Her aday, mevcut uygulama modeliyle (lojistik regresyon, medyanla doldurma) AYNI 5 katta
karşılaştırılır. Yalnızca eğitim dönemleri (2007-2016) kullanılır; test seti (2017-20)
bu aşamada hiç görülmez.

Kabul kuralı: ortalama AUC kazancı >= 0.003 VE 5 katın en az 4'ünde temel modelden iyi.

Adaylar
  A. XGBoost (doğrusal olmayan ilişkiler)
  B. Spline özellikleri + lojistik regresyon (yaş, BMI gibi değişkenlerde eğrisel etki)
  C. HbA1c regresyonu: "diyabet var/yok" yerine kan şekeri değerini tahmin edip skor olarak kullanmak
  D. Topluluk: lojistik regresyon + XGBoost + HbA1c regresyonu (sıra ortalaması)
  E. Beslenme verisi (24 saatlik kayıt) eklemek
  F. İteratif doldurma (eksik değeri diğer özelliklerden tahmin etmek)

Çalıştırma:
    .venv/Scripts/python -m backend.ml.nhanes_gelistirme
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from backend.ml.nhanes_hazirla import CIKTI, DIYET_OZELLIKLER
from backend.ml.nhanes_uygulama import HEDEF, TEST_DONEMI, UYGULAMA_OZELLIKLERI
from backend.ml.ozellikler import GRAFIK_DIR, RAPOR_DIR, RASTGELE

KAZANC_ESIGI = 0.003
SUREKLI = ["yas", "bmi", "bel", "boy", "kilo", "sistolik", "diastolik", "oturma_dk", "uyku_saat", "bel_boy_orani"]


def lr(imputer=None, C=0.1):
    return Pipeline([("doldur", imputer or SimpleImputer(strategy="median")), ("olcek", StandardScaler()),
                     ("model", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])


def spline_lr(sutunlar):
    surekli = [s for s in SUREKLI if s in sutunlar]
    diger = [s for s in sutunlar if s not in surekli]
    donustur = ColumnTransformer([("spline", SplineTransformer(n_knots=4, degree=3), surekli), ("gec", "passthrough", diger)])
    return Pipeline([("doldur", SimpleImputer(strategy="median")), ("sutun", _Sutunlu(sutunlar, donustur)),
                     ("olcek", StandardScaler()), ("model", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))])


class _Sutunlu:
    """SimpleImputer numpy döndürdüğü için ColumnTransformer'a sütun adlarını geri verir."""

    def __init__(self, sutunlar, ic):
        self.sutunlar, self.ic = sutunlar, ic

    def fit(self, X, y=None):
        self.ic.fit(pd.DataFrame(X, columns=self.sutunlar))
        return self

    def transform(self, X):
        return self.ic.transform(pd.DataFrame(X, columns=self.sutunlar))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {"sutunlar": self.sutunlar, "ic": self.ic}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


def xgb(oran):
    return XGBClassifier(tree_method="hist", max_depth=3, n_estimators=600, learning_rate=0.02, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=oran, eval_metric="logloss", n_jobs=-1, random_state=RASTGELE)


def hba1c_regresor():
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=40, random_state=RASTGELE)


def sira(s):
    return rankdata(s) / len(s)


def main():
    df = pd.read_csv(CIKTI, index_col=0)
    egitim = df[df["donem"] != TEST_DONEMI]
    temel = UYGULAMA_OZELLIKLERI
    diyetli = temel + list(DIYET_OZELLIKLER)
    X, y, hba1c = egitim, egitim[HEDEF].values, egitim["hba1c"].values
    oran = float((y == 0).sum() / (y == 1).sum())

    kfold = StratifiedKFold(5, shuffle=True, random_state=RASTGELE)
    adlar = ["Temel (mevcut model)", "A. XGBoost", "B. Spline + LR", "C. HbA1c regresyonu",
             "D. Topluluk (LR+XGB+HbA1c)", "E. Beslenme verisi", "F. İteratif doldurma"]
    auc = {a: [] for a in adlar}

    for k, (i, j) in enumerate(kfold.split(X, y), 1):
        Xe, Xv, ye, yv = X.iloc[i], X.iloc[j], y[i], y[j]
        s = {}
        s["Temel (mevcut model)"] = lr().fit(Xe[temel], ye).predict_proba(Xv[temel])[:, 1]
        s["A. XGBoost"] = xgb(oran).fit(Xe[temel], ye).predict_proba(Xv[temel])[:, 1]
        s["B. Spline + LR"] = spline_lr(temel).fit(Xe[temel], ye).predict_proba(Xv[temel])[:, 1]
        s["C. HbA1c regresyonu"] = hba1c_regresor().fit(Xe[temel], np.log(hba1c[i])).predict(Xv[temel])
        s["D. Topluluk (LR+XGB+HbA1c)"] = (sira(s["Temel (mevcut model)"]) + sira(s["A. XGBoost"]) + sira(s["C. HbA1c regresyonu"])) / 3
        s["E. Beslenme verisi"] = lr().fit(Xe[diyetli], ye).predict_proba(Xv[diyetli])[:, 1]
        s["F. İteratif doldurma"] = lr(IterativeImputer(max_iter=10, random_state=RASTGELE)).fit(Xe[temel], ye).predict_proba(Xv[temel])[:, 1]
        for a in adlar:
            auc[a].append(float(roc_auc_score(yv, s[a])))
        print(f"Kat {k}: " + "  ".join(f"{a.split(' ')[0]}={auc[a][-1]:.4f}" for a in adlar), flush=True)

    taban = np.array(auc["Temel (mevcut model)"])
    karar = {}
    for a in adlar[1:]:
        fark = np.array(auc[a]) - taban
        kabul = bool(fark.mean() >= KAZANC_ESIGI and (fark > 0).sum() >= 4)
        karar[a] = {"ortalama_auc": float(np.mean(auc[a])), "kazanc": float(fark.mean()), "kazanc_kat": fark.tolist(),
                    "kazandigi_kat": int((fark > 0).sum()), "karar": "DAHİL" if kabul else "ELENDİ"}
        print(f"{a:30s} kazanç {fark.mean():+.4f}  ({(fark > 0).sum()}/5 kat)  -> {karar[a]['karar']}")

    # Grafik: her adayın katlardaki kazancı
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for n, a in enumerate(adlar[1:]):
        f = np.array(karar[a]["kazanc_kat"])
        renk = "#1f4e79" if karar[a]["karar"] == "DAHİL" else "#9aa3ad"
        ax.barh(n, f.mean(), color=renk, height=0.55)
        ax.scatter(f, [n] * 5, color="#222", s=12, zorder=3)
        ax.text(max(f.max(), f.mean()) + 0.001, n, karar[a]["karar"], va="center", fontsize=9, color=renk, fontweight="bold")
    ax.axvline(0, color="#222", lw=0.8)
    ax.axvline(KAZANC_ESIGI, color="#b22222", lw=0.8, ls="--")
    ax.set_yticks(range(len(adlar) - 1), adlar[1:])
    ax.invert_yaxis()
    ax.set_xlabel("Temel modele göre AUC kazancı (çubuk: ortalama, nokta: her kat)")
    ax.set_title("Geliştirme adaylarının elenmesi (5-katlı CV, yalnızca 2007-16)")
    plt.tight_layout()
    plt.savefig(GRAFIK_DIR / "33_gelistirme_eleme.png", dpi=150)
    plt.close()

    with open(RAPOR_DIR / "nhanes_gelistirme.json", "w", encoding="utf-8") as f:
        json.dump({"kural": f"ortalama kazanç >= {KAZANC_ESIGI} ve en az 4/5 katta üstün",
                   "temel_auc": float(taban.mean()), "temel_kat": taban.tolist(), "adaylar": karar}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

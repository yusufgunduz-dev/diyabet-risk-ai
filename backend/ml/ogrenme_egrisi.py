"""5. adım: Öğrenme eğrisi - "Daha fazla veri / daha uzun eğitim başarıyı artırır mı?"

  1. Veri miktarı: XGBoost'u 500 kişiden 160 bin kişiye kadar artan verilerle eğitir.
  2. Eğitim süresi: ağaç sayısını (iterasyon) 50'den 3000'e kadar artırır.
Her durumda aynı test setinde ROC-AUC ölçülür.

Çalıştırma (model_egitimi'nden sonra):
    .venv/Scripts/python -m backend.ml.ogrenme_egrisi
"""

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from backend.ml.ozellikler import GRAFIK_DIR, MODEL_DIR, RAPOR_DIR, RASTGELE


def model(n_agac=300):
    return XGBClassifier(tree_method="hist", max_depth=3, learning_rate=0.03, n_estimators=n_agac,
                         subsample=0.8, eval_metric="logloss", n_jobs=-1, random_state=RASTGELE)


def main():
    v = joblib.load(MODEL_DIR / "bolunmus_veri.joblib")
    Xe, ye, Xt, yt = v["X_egitim"], v["y_egitim"], v["X_test"], v["y_test"]

    print("1) Veri miktarı deneyi")
    boyutlar = [500, 1000, 2000, 5000, 10000, 20000, 40000, 80000, len(Xe)]
    veri_auc = []
    for n in boyutlar:
        Xa, ya = (Xe, ye) if n >= len(Xe) else train_test_split(Xe, ye, train_size=n, stratify=ye, random_state=RASTGELE)[::2]
        auc = roc_auc_score(yt, model().fit(Xa, ya).predict_proba(Xt)[:, 1])
        veri_auc.append(float(auc))
        print(f"  {n:>7,} kişi -> AUC {auc:.4f}")

    print("2) Eğitim uzunluğu deneyi (ağaç sayısı)")
    agaclar = [50, 100, 200, 300, 600, 1000, 2000, 3000]
    sure_auc, sureler = [], []
    for n in agaclar:
        t0 = time.time()
        auc = roc_auc_score(yt, model(n).fit(Xe, ye).predict_proba(Xt)[:, 1])
        sure_auc.append(float(auc))
        sureler.append(round(time.time() - t0, 1))
        print(f"  {n:>5} ağaç -> AUC {auc:.4f} ({sureler[-1]} sn)")

    fig, eksen = plt.subplots(1, 2, figsize=(12, 4.5))
    eksen[0].plot(boyutlar, veri_auc, "o-", color="#4C72B0")
    eksen[0].set_xscale("log")
    eksen[0].set_xlabel("Eğitimdeki kişi sayısı (log ölçek)")
    eksen[0].set_ylabel("Test ROC-AUC")
    eksen[0].set_title("Veri miktarı arttıkça başarı")
    eksen[1].plot(agaclar, sure_auc, "o-", color="#DD5555")
    eksen[1].set_xscale("log")
    eksen[1].set_xlabel("Ağaç sayısı / eğitim uzunluğu (log ölçek)")
    eksen[1].set_title("Eğitim uzadıkça başarı")
    for e in eksen:
        e.set_ylim(min(veri_auc + sure_auc) - 0.01, max(veri_auc + sure_auc) + 0.01)
    kaydet = GRAFIK_DIR / "19_ogrenme_egrisi.png"
    plt.tight_layout()
    plt.savefig(kaydet, dpi=150)
    plt.close()

    with open(RAPOR_DIR / "ogrenme_egrisi.json", "w", encoding="utf-8") as f:
        json.dump({"boyut": boyutlar, "veri_auc": veri_auc, "agac": agaclar, "agac_auc": sure_auc, "agac_sure_sn": sureler}, f, indent=2)
    print(f"Kaydedildi: {kaydet}")


if __name__ == "__main__":
    main()

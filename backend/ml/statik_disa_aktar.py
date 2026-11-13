"""Statik site (Netlify) için modelleri ve raporları JSON'a aktarır, site klasörünü üretir.

Netlify Python çalıştırmadığı için tahminler tarayıcıda yapılır (frontend/statik.js):
  - BRFSS: XGBoost ağaçları + izotonik kalibrasyon + TreeSHAP
  - NHANES: 5 katlı kalibre lojistik regresyon + katkılar + ADA/FINDRISC + Türkiye kalibrasyonu
Aynı sonucu verdiği tests/test_statik.py ile doğrulanır.

Çalıştırma (proje kök dizininden):
    .venv/Scripts/python -m backend.ml.statik_disa_aktar
Çıktı: netlify_site/  (bu klasör Netlify'a sürüklenip bırakılır)
"""

import json
import shutil

import joblib
import numpy as np

from backend.api.main import sonuclar
from backend.ml.nhanes_hazirla import NHANES_OZELLIKLER
from backend.ml.nhanes_uygulama import SORULAR
from backend.ml.ozellikler import BACKEND_DIR, GRAFIK_DIR, MODEL_DIR, OZELLIKLER

FRONTEND = BACKEND_DIR.parent / "frontend"
SITE = BACKEND_DIR.parent / "netlify_site"
FRONTEND_DOSYALARI = ["index.html", "styles.css", "app.js", "gizli.js", "statik.js", "qr.html"]


def izotonik(m):
    return {"x": m.X_thresholds_.tolist(), "y": m.y_thresholds_.tolist()}


def boru_hatti(pipe):
    """SimpleImputer(median) -> StandardScaler -> LogisticRegression parametreleri."""
    return {
        "medyan": pipe.named_steps["doldur"].statistics_.tolist(),
        "ortalama": pipe.named_steps["olcek"].mean_.tolist(),
        "olcek": pipe.named_steps["olcek"].scale_.tolist(),
        "katsayi": pipe.named_steps["model"].coef_[0].tolist(),
        "sabit": float(pipe.named_steps["model"].intercept_[0]),
    }


def brfss_aktar():
    paket = joblib.load(MODEL_DIR / "diyabet_modeli.joblib")
    booster = paket["model"].get_booster()
    ayar = json.loads(booster.save_config())["learner"]
    assert ayar["objective"]["name"] == "binary:logistic"
    taban = float(ayar["learner_model_param"]["base_score"].strip("[]"))
    agaclar = [json.loads(a) for a in booster.get_dump(dump_format="json", with_stats=True)]
    return {
        "ozellikler": paket["ozellikler"],
        "taban_marj": float(np.log(taban / (1 - taban))),
        "agaclar": agaclar,
        "kalibrator": izotonik(paket["kalibrator"]),
        "esik": paket["esik"],
        "taban_oran": paket["taban_oran"],
        "meta": OZELLIKLER,
    }


def nhanes_aktar():
    paket = joblib.load(MODEL_DIR / "nhanes_uygulama.joblib")
    katlar = [{"boru": boru_hatti(c.estimator), "kalibrator": izotonik(c.calibrators[0])}
              for c in paket["kalibre"].calibrated_classifiers_]
    return {
        "ozellikler": paket["ozellikler"],
        "etiketler": {o: NHANES_OZELLIKLER.get(o, o) for o in paket["ozellikler"]},
        "katlar": katlar,
        "ham": boru_hatti(paket["ham"]),
        "esik": paket["esik"],
        "sonuc": paket["sonuc"],
        "sorular": SORULAR,
    }


def main():
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "model").mkdir(parents=True)
    for ad in FRONTEND_DOSYALARI:
        shutil.copy(FRONTEND / ad, SITE / ad)
    shutil.copytree(GRAFIK_DIR, SITE / "figures")

    yaz = lambda ad, veri: (SITE / "model" / ad).write_text(json.dumps(veri, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    yaz("brfss.json", brfss_aktar())
    yaz("nhanes.json", nhanes_aktar())
    yaz("sonuclar.json", sonuclar())

    # Statik modda tahmin katmanı app.js'ten önce yüklenir
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert '<script src="app.js' in html
    html = html.replace('<script src="app.js', '<script src="statik.js"></script>\n  <script src="app.js', 1)
    (SITE / "index.html").write_text(html, encoding="utf-8")

    boyut = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file()) / 1e6
    print(f"Site hazır: {SITE} ({boyut:.1f} MB)")


if __name__ == "__main__":
    main()

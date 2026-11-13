"""Diyabet Risk API'si (FastAPI).

Eğitilmiş XGBoost modelini yükler ve frontend'e şu uç noktaları sunar:
    GET  /api/saglik     -> sunucu ve model durumu
    GET  /api/meta       -> form için soru / seçenek tanımları
    GET  /api/sonuclar   -> model karşılaştırma, kümeleme ve veri özeti raporları
    POST /api/tahmin     -> risk olasılığı + SHAP açıklaması + "ne yapabilirim" senaryoları
    GET  /api/gizli/meta   -> NHANES "gizli diyabet" formu ve test seti karşılaştırması
    POST /api/gizli/tahmin -> NHANES modeli ve ADA Risk Testi yan yana
Ayrıca /figures altında grafikleri, / altında frontend'i yayınlar.

Çalıştırma (proje kök dizininden):
    .venv/Scripts/python -m uvicorn backend.api.main:app --port 8000
"""

import json

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.ml.nhanes_hazirla import NHANES_OZELLIKLER
from backend.ml.nhanes_uygulama import SORULAR as GIZLI_SORULAR, turkiyeye_kaydir
from backend.ml.ozellikler import (
    BACKEND_DIR, GRAFIK_DIR, MODEL_DIR, OZELLIKLER, OZELLIK_ADLARI, RAPOR_DIR,
)

FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
OZELLIK = {o["ad"]: o for o in OZELLIKLER}

paket = joblib.load(MODEL_DIR / "diyabet_modeli.joblib")
model, kalibrator, ESIK, TABAN = paket["model"], paket["kalibrator"], paket["esik"], paket["taban_oran"]
aciklayici = shap.TreeExplainer(model)


def rapor_oku(ad):
    yol = RAPOR_DIR / ad
    return json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else None


app = FastAPI(title="Diyabet Risk API", version="1.0")


class TahminIstegi(BaseModel):
    degerler: dict[str, float]


def dogrula(degerler):
    eksik = [a for a in OZELLIK_ADLARI if a not in degerler]
    if eksik:
        raise HTTPException(422, f"Eksik alanlar: {', '.join(eksik)}")
    for ad in OZELLIK_ADLARI:
        o, v = OZELLIK[ad], degerler[ad]
        if o["tip"] == "sayi" and not o["min"] <= v <= o["max"]:
            raise HTTPException(422, f"{o['etiket']} {o['min']}-{o['max']} arasında olmalı")
        if o["tip"] == "secim" and v not in {s["deger"] for s in o["secenekler"]}:
            raise HTTPException(422, f"{o['etiket']} için geçersiz değer: {v}")
    return pd.DataFrame([[degerler[a] for a in OZELLIK_ADLARI]], columns=OZELLIK_ADLARI)


def olasilik(X):
    return kalibrator.predict(model.predict_proba(X)[:, 1])


def deger_metni(ad, v):
    o = OZELLIK[ad]
    if o["tip"] == "sayi":
        return f"{v:g}"
    return next(s["etiket"] for s in o["secenekler"] if s["deger"] == v)


def seviye(p):
    """Kullanıcıya gösterilen risk etiketi (mutlak olasılığa göre).

    Sınıflandırma eşiği (ESIK) ayrı tutulur: o, tarama amacıyla recall'u yüksek
    tutmak için düşük seçildi ve "doktora görünün" önerisini belirler.
    """
    if p >= 0.40:
        return "Çok yüksek"
    if p >= 0.25:
        return "Yüksek"
    if p >= 0.10:
        return "Orta"
    return "Düşük"


# Kişinin değiştirebileceği alışkanlıklar: (özellik, hedef değer, açıklama)
SENARYOLAR = [
    ("BMI", lambda v: min(v, 24.9), "Kilonuzu normal aralığa (BMI < 25) indirirseniz"),
    ("PhysActivity", lambda v: 1, "Düzenli fiziksel aktivite yaparsanız"),
    ("HvyAlcoholConsump", lambda v: 0, "Yoğun alkol tüketimini bırakırsanız"),
    ("Veggies", lambda v: 1, "Her gün sebze yerseniz"),
    ("Fruits", lambda v: 1, "Her gün meyve yerseniz"),
]


@app.get("/api/saglik")
def saglik():
    return {"durum": "ok", "model": "XGBoost + izotonik kalibrasyon", "esik": ESIK, "taban_oran": TABAN}


@app.get("/api/meta")
def meta():
    return {"ozellikler": OZELLIKLER, "esik": ESIK, "taban_oran": TABAN}


@app.get("/api/sonuclar")
def sonuclar():
    return {
        "veri": rapor_oku("veri_ozeti.json"),
        "model": rapor_oku("model_sonuclari.json"),
        "aciklama": rapor_oku("aciklama_kumeleme.json"),
        "ogrenme_egrisi": rapor_oku("ogrenme_egrisi.json"),
        "nhanes_veri": rapor_oku("nhanes_veri_ozeti.json"),
        "nhanes": rapor_oku("nhanes_uygulama.json"),
        "gelistirme": rapor_oku("nhanes_gelistirme.json"),
        "grafikler": sorted(p.name for p in GRAFIK_DIR.glob("*.png")),
    }


@app.post("/api/tahmin")
def tahmin(istek: TahminIstegi):
    X = dogrula(istek.degerler)
    p = float(olasilik(X)[0])

    sv = aciklayici(X).values[0]
    faktorler = sorted(
        ({"ozellik": a, "etiket": OZELLIK[a]["etiket"], "deger": deger_metni(a, X.iloc[0][a]), "etki": float(s)}
         for a, s in zip(OZELLIK_ADLARI, sv)),
        key=lambda f: abs(f["etki"]), reverse=True,
    )

    senaryolar = []
    for ad, hedef, metin in SENARYOLAR:
        mevcut = X.iloc[0][ad]
        yeni = hedef(mevcut)
        if yeni == mevcut:
            continue
        X2 = X.copy()
        X2[ad] = yeni
        p2 = float(olasilik(X2)[0])
        if p2 < p - 0.005:
            senaryolar.append({"metin": metin, "yeni_olasilik": p2, "fark": p2 - p})
    senaryolar.sort(key=lambda s: s["fark"])

    return {
        "olasilik": p,
        "seviye": seviye(p),
        "tarama_pozitif": p >= ESIK,
        "esik": ESIK,
        "taban_oran": TABAN,
        "kat": p / TABAN,
        "faktorler": faktorler,
        "senaryolar": senaryolar,
    }


# ---------------- NHANES: gizli (teşhis edilmemiş) diyabet ----------------

gizli = joblib.load(MODEL_DIR / "nhanes_uygulama.joblib")
GIZLI_SORU = {s["ad"]: s for s in GIZLI_SORULAR}


def ada_detay(d):
    """ADA Diyabet Risk Testi puanının madde madde dökümü (Bang ve ark. 2009)."""
    yas = d["yas"]
    bmi = d["bmi"]
    maddeler = [
        ("Yaş", 0 if yas < 40 else 1 if yas < 50 else 2 if yas < 60 else 3, f"{yas:g} yaş"),
        ("Erkek olmak", int(d["erkek"] == 1), "Erkek" if d["erkek"] == 1 else "Kadın"),
        ("Gebelik diyabeti", int(d["erkek"] == 0 and d.get("gebelik_diyabeti") == 1), ""),
        ("Ailede diyabet", int(d.get("aile_diyabet") == 1), ""),
        ("Yüksek tansiyon", int(d.get("tansiyon_tani") == 1 or d.get("tansiyon_ilac") == 1), ""),
        ("Fiziksel olarak aktif değil", int(d.get("agir_aktivite") != 1 and d.get("orta_aktivite") != 1), ""),
        ("Vücut kitle indeksi", 0 if bmi < 25 else 1 if bmi < 30 else 2 if bmi < 40 else 3, f"BMI {bmi:.1f}"),
    ]
    return [{"madde": m, "puan": p, "not": n} for m, p, n in maddeler]


def findrisc_detay(d):
    """FINDRISC puanı (NHANES uyarlaması, bkz. nhanes_hazirla.findrisc_puani)."""
    yas, bmi, bel = d["yas"], d["bmi"], d.get("bel")
    if bel is None:
        bel_puan, bel_not = 0, "ölçülmedi"
    else:
        sinir = (94, 102) if d["erkek"] == 1 else (80, 88)
        bel_puan = 0 if bel < sinir[0] else 3 if bel <= sinir[1] else 4
        bel_not = f"{bel:g} cm"
    aktif = d.get("agir_aktivite") == 1 or d.get("orta_aktivite") == 1
    maddeler = [
        ("Yaş", 0 if yas < 45 else 2 if yas < 55 else 3 if yas < 65 else 4, f"{yas:g} yaş"),
        ("Vücut kitle indeksi", 0 if bmi < 25 else 1 if bmi <= 30 else 3, f"BMI {bmi:.1f}"),
        ("Bel çevresi", bel_puan, bel_not),
        ("Fiziksel olarak aktif değil", 0 if aktif else 2, ""),
        ("Tansiyon ilacı", 2 if d.get("tansiyon_ilac") == 1 else 0, ""),
        ("Daha önce şekeri yüksek bulunmuş", 5 if d.get("prediyabet_tani") == 1 else 0, ""),
        ("Ailede diyabet", 5 if d.get("aile_diyabet") == 1 else 0, ""),
    ]
    return [{"madde": m, "puan": p, "not": n} for m, p, n in maddeler]


class GizliIstek(BaseModel):
    degerler: dict[str, float | None]


@app.get("/api/gizli/meta")
def gizli_meta():
    return {"sorular": GIZLI_SORULAR, "sonuc": gizli["sonuc"]}


@app.post("/api/gizli/tahmin")
def gizli_tahmin(istek: GizliIstek):
    d = dict(istek.degerler)
    for s in GIZLI_SORULAR:
        v = d.get(s["ad"])
        if v is None:
            if s.get("istege_bagli") or s.get("sadece_kadin") or s.get("kosullu"):
                d[s["ad"]] = None
                continue
            raise HTTPException(422, f"Eksik alan: {s['soru']}")
        if s["tip"] == "sayi" and not s["min"] <= v <= s["max"]:
            raise HTTPException(422, f"{s['soru']}: {s['min']}-{s['max']} arasında olmalı")
        if s.get("carpan"):
            d[s["ad"]] = v * s["carpan"]
    if d["erkek"] == 1:
        d["gebelik_diyabeti"] = 0
    if d.get("tansiyon_tani") != 1:
        d["tansiyon_ilac"] = 0
    if d.get("sigara_hic") != 1:
        d["sigara_simdi"] = 0
    d["bmi"] = d["kilo"] / (d["boy"] / 100) ** 2
    d["bel_boy_orani"] = d["bel"] / d["boy"] if d.get("bel") else None

    X = pd.DataFrame([[d.get(o) for o in gizli["ozellikler"]]], columns=gizli["ozellikler"]).astype(float)
    p = float(gizli["kalibre"].predict_proba(X)[0, 1])

    # Lojistik regresyonda katkı = katsayı x standartlaştırılmış değer (ortalama kişiye göre log-odds)
    ham = gizli["ham"]
    z = ham[:-1].transform(X)[0]
    katki = ham.named_steps["model"].coef_[0] * z
    faktorler = sorted(
        ({"ozellik": o, "etiket": NHANES_OZELLIKLER.get(o, o), "etki": float(k), "eksik": d.get(o) is None}
         for o, k in zip(gizli["ozellikler"], katki)),
        key=lambda f: abs(f["etki"]), reverse=True,
    )

    ada = ada_detay(d)
    ada_puan = sum(m["puan"] for m in ada)
    fin = findrisc_detay(d)
    fin_puan = sum(m["puan"] for m in fin)
    return {
        "model": {"olasilik": p, "uyari": p >= gizli["esik"], "esik": gizli["esik"], "kat": p / gizli["sonuc"]["taban_oran"], "faktorler": faktorler[:8],
                  # Türkiye'deki gizli diyabet yaygınlığına göre kaydırılmış olasılık (karar aynı kalır)
                  "olasilik_tr": float(turkiyeye_kaydir(p, gizli["sonuc"]["taban_oran"])), "taban_tr": gizli["sonuc"]["turkiye"]["gizli_oran"]},
        "ada": {"puan": ada_puan, "uyari": ada_puan >= 5, "maddeler": ada},
        "findrisc": {"puan": fin_puan, "uyari": fin_puan >= gizli["sonuc"]["findrisc_esik"], "maddeler": fin},
        "bmi": d["bmi"],
        "taban_oran": gizli["sonuc"]["taban_oran"],
    }


app.mount("/figures", StaticFiles(directory=GRAFIK_DIR), name="figures")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

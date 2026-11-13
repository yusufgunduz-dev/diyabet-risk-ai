"""API ve klinik puanlama testleri.

Çalıştırma (proje kök dizininden):
    .venv/Scripts/python -m pytest -q
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.ml.nhanes_uygulama import TR_GIZLI_ORAN, turkiyeye_kaydir

istemci = TestClient(app)

# Sunumdaki "testlerin ayrıştığı profil"
ORNEK_KISI = {
    "erkek": 0, "yas": 47, "boy": 160, "kilo": 70, "egitim": 2, "aile_diyabet": 0, "prediyabet_tani": 1,
    "gebelik_diyabeti": 0, "tansiyon_tani": 0, "tansiyon_ilac": None, "kolesterol_tani": 1, "kalp_hastaligi": 0,
    "inme": 0, "genel_saglik": 4, "agir_aktivite": 0, "orta_aktivite": 1, "oturma_dk": 8, "uyku_saat": 6,
    "sigara_hic": 0, "sigara_simdi": None, "bel": 102, "sistolik": 138, "diastolik": 84,
}


def brfss_ornek():
    meta = istemci.get("/api/meta").json()
    return {o["ad"]: (o["min"] if o["tip"] == "sayi" else o["secenekler"][0]["deger"]) for o in meta["ozellikler"]} | {"BMI": 27}


def test_saglik():
    r = istemci.get("/api/saglik")
    assert r.status_code == 200 and r.json()["durum"] == "ok"


def test_meta_17_soru():
    assert len(istemci.get("/api/meta").json()["ozellikler"]) == 17


def test_brfss_tahmin_gecerli_olasilik():
    r = istemci.post("/api/tahmin", json={"degerler": brfss_ornek()})
    assert r.status_code == 200
    s = r.json()
    assert 0 <= s["olasilik"] <= 1
    assert len(s["faktorler"]) == 17
    assert s["seviye"] in {"Düşük", "Orta", "Yüksek", "Çok yüksek"}


@pytest.mark.parametrize("alan, deger", [("BMI", 200), ("Age", 99), ("HighBP", 5)])
def test_brfss_gecersiz_girdi_reddedilir(alan, deger):
    r = istemci.post("/api/tahmin", json={"degerler": brfss_ornek() | {alan: deger}})
    assert r.status_code == 422


def test_brfss_eksik_alan_reddedilir():
    d = brfss_ornek()
    d.pop("HighBP")
    assert istemci.post("/api/tahmin", json={"degerler": d}).status_code == 422


def test_klinik_puanlar_ornek_kisi():
    s = istemci.post("/api/gizli/tahmin", json={"degerler": ORNEK_KISI}).json()
    assert s["ada"]["puan"] == 2 and not s["ada"]["uyari"]
    assert s["findrisc"]["puan"] == 12 and not s["findrisc"]["uyari"]
    assert s["model"]["uyari"]  # klinik testlerin kaçırdığı profil


def test_istege_bagli_olcumler_bos_birakilabilir():
    kisi = ORNEK_KISI | {"bel": None, "sistolik": None, "diastolik": None}
    r = istemci.post("/api/gizli/tahmin", json={"degerler": kisi})
    assert r.status_code == 200
    assert 0 <= r.json()["model"]["olasilik"] <= 1


def test_zorunlu_klinik_alan_eksikse_hata():
    kisi = ORNEK_KISI | {"yas": None}
    assert istemci.post("/api/gizli/tahmin", json={"degerler": kisi}).status_code == 422


def test_turkiye_kalibrasyonu_siralamayi_korur():
    p = np.linspace(0.001, 0.99, 200)
    q = turkiyeye_kaydir(p, 0.053)
    assert np.all(np.diff(q) > 0)                   # monoton: sıralama ve AUC değişmez
    assert np.allclose(turkiyeye_kaydir(p, TR_GIZLI_ORAN), p)  # aynı oranda dönüşüm yok
    assert np.all(q > p)                            # Türkiye'de taban oran daha yüksek


def test_api_turkiye_olasiligi_dondurur():
    m = istemci.post("/api/gizli/tahmin", json={"degerler": ORNEK_KISI}).json()["model"]
    assert m["olasilik_tr"] > m["olasilik"]
    assert m["taban_tr"] == pytest.approx(TR_GIZLI_ORAN)

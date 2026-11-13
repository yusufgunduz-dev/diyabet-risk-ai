"""Statik sitenin (tarayıcıda çalışan modeller) Python API'siyle aynı sonucu verdiğini doğrular.

Önce site üretilmiş olmalı:  .venv/Scripts/python -m backend.ml.statik_disa_aktar
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.ml.nhanes_uygulama import SORULAR
from backend.ml.ozellikler import OZELLIKLER

KOK = Path(__file__).resolve().parents[1]
SITE = KOK / "netlify_site"
istemci = TestClient(app)
rs = np.random.RandomState(0)

pytestmark = pytest.mark.skipif(not (SITE / "model" / "brfss.json").exists() or shutil.which("node") is None,
                                reason="Statik site üretilmemiş veya Node.js yok")


def rastgele_brfss():
    d = {}
    for o in OZELLIKLER:
        if o["tip"] == "sayi":
            d[o["ad"]] = int(rs.randint(o["min"], min(o["max"], 60) + 1))
        else:
            d[o["ad"]] = rs.choice([s["deger"] for s in o["secenekler"]]).item()
    return d


def rastgele_nhanes():
    d = {}
    for s in SORULAR:
        if s["tip"] == "sayi":
            d[s["ad"]] = None if s.get("istege_bagli") and rs.rand() < 0.3 else int(rs.randint(s["min"], s["max"] + 1))
        else:
            d[s["ad"]] = rs.choice([o["deger"] for o in s["secenekler"]]).item()
    return d


@pytest.fixture(scope="module")
def karsilastirma(tmp_path_factory):
    girdiler = {"brfss": [rastgele_brfss() for _ in range(30)], "nhanes": [rastgele_nhanes() for _ in range(30)]}
    dosya = tmp_path_factory.mktemp("statik") / "girdiler.json"
    dosya.write_text(json.dumps(girdiler), encoding="utf-8")
    cikti = subprocess.run(["node", str(KOK / "tests" / "statik_calistir.mjs"), str(SITE), str(dosya)],
                           capture_output=True, text=True, encoding="utf-8", check=True).stdout
    js = json.loads(cikti)
    py = {"brfss": [istemci.post("/api/tahmin", json={"degerler": d}).json() for d in girdiler["brfss"]],
          "nhanes": [istemci.post("/api/gizli/tahmin", json={"degerler": d}).json() for d in girdiler["nhanes"]]}
    return js, py


def test_brfss_olasilik_ayni(karsilastirma):
    js, py = karsilastirma
    for a, b in zip(js["brfss"], py["brfss"]):
        assert a["olasilik"] == pytest.approx(b["olasilik"], abs=1e-6)
        assert a["seviye"] == b["seviye"] and a["tarama_pozitif"] == b["tarama_pozitif"]


def test_brfss_shap_ayni(karsilastirma):
    js, py = karsilastirma
    for a, b in zip(js["brfss"], py["brfss"]):
        ja = {f["ozellik"]: f["etki"] for f in a["faktorler"]}
        pa = {f["ozellik"]: f["etki"] for f in b["faktorler"]}
        for k in pa:
            assert ja[k] == pytest.approx(pa[k], abs=1e-4)
        assert [f["deger"] for f in a["faktorler"][:5]] == [f["deger"] for f in b["faktorler"][:5]]


def test_brfss_senaryolar_ayni(karsilastirma):
    js, py = karsilastirma
    for a, b in zip(js["brfss"], py["brfss"]):
        assert [s["metin"] for s in a["senaryolar"]] == [s["metin"] for s in b["senaryolar"]]


def test_nhanes_ayni(karsilastirma):
    js, py = karsilastirma
    for a, b in zip(js["nhanes"], py["nhanes"]):
        assert a["model"]["olasilik"] == pytest.approx(b["model"]["olasilik"], abs=1e-6)
        assert a["model"]["olasilik_tr"] == pytest.approx(b["model"]["olasilik_tr"], abs=1e-6)
        assert a["model"]["uyari"] == b["model"]["uyari"]
        assert [f["ozellik"] for f in a["model"]["faktorler"]] == [f["ozellik"] for f in b["model"]["faktorler"]]
        assert a["ada"] == b["ada"] and a["findrisc"]["puan"] == b["findrisc"]["puan"]

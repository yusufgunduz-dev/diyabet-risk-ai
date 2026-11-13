"""NHANES 2. adım: 6 dönemi birleştirir, hedefleri ve özellikleri oluşturur.

Hedefler (kan tahliline göre, model bu değerleri GÖRMEZ):
  - tani_konmamis_diyabet: Doktor "diyabetsin" dememiş ama HbA1c >= 6.5 veya açlık şekeri >= 126.
  - disglisemi: Doktor "diyabetsin" dememiş ama HbA1c >= 5.7 veya açlık şekeri >= 100
    (prediyabet + teşhis edilmemiş diyabet).
Modelin girdileri yalnızca anket soruları ve kan gerektirmeyen ölçümlerdir
(boy, kilo, bel çevresi, tansiyon).

Ayrıca karşılaştırma için Amerikan Diyabet Derneği (ADA) Risk Testi puanını hesaplar
(Bang ve ark., 2009, Annals of Internal Medicine).

Çalıştırma:
    .venv/Scripts/python -m backend.ml.nhanes_hazirla
"""

import json

import numpy as np
import pandas as pd

from backend.ml.nhanes_indir import DONEMLER, NHANES_DIR, dosya_adi
from backend.ml.ozellikler import BACKEND_DIR, RAPOR_DIR

CIKTI = BACKEND_DIR / "data" / "nhanes_hazir.csv"
DONEM_ADI = {2007: "2007-08", 2009: "2009-10", 2011: "2011-12", 2013: "2013-14", 2015: "2015-16", 2017: "2017-20"}

# Model girdileri: ad -> Türkçe açıklama
NHANES_OZELLIKLER = {
    "yas": "Yaş",
    "erkek": "Cinsiyet (erkek)",
    "etnik": "Etnik köken",
    "gelir_orani": "Gelir / yoksulluk sınırı oranı",
    "egitim": "Eğitim düzeyi",
    "bmi": "Vücut kitle indeksi",
    "bel": "Bel çevresi (cm)",
    "boy": "Boy (cm)",
    "kilo": "Kilo (kg)",
    "bel_boy_orani": "Bel / boy oranı",
    "sistolik": "Büyük tansiyon (ölçüm)",
    "diastolik": "Küçük tansiyon (ölçüm)",
    "aile_diyabet": "Ailede diyabet",
    "tansiyon_tani": "Yüksek tansiyon tanısı",
    "tansiyon_ilac": "Tansiyon ilacı",
    "kolesterol_tani": "Yüksek kolesterol tanısı",
    "kalp_hastaligi": "Kalp hastalığı / kalp krizi",
    "inme": "İnme",
    "gebelik_diyabeti": "Gebelik diyabeti öyküsü",
    "prediyabet_tani": "Daha önce şekeri yüksek / sınırda denmiş",
    "agir_aktivite": "Ağır fiziksel aktivite",
    "orta_aktivite": "Orta fiziksel aktivite",
    "oturma_dk": "Günlük oturma süresi (dk)",
    "sigara_hic": "Sigara geçmişi",
    "sigara_simdi": "Halen sigara içiyor",
    "alkol_gunluk": "Günlük alkol (kadeh)",
    "genel_saglik": "Genel sağlık algısı (1-5)",
    "uyku_saat": "Uyku süresi (saat)",
}

ADA_OZELLIKLER = ["yas", "erkek", "gebelik_diyabeti", "aile_diyabet", "tansiyon_tani", "aktif", "bmi"]

# 24 saatlik beslenme kaydından (yalnızca güvenilir kayıtlar) türetilen özellikler
DIYET_OZELLIKLER = {
    "kalori": "Günlük kalori (kcal)",
    "karbonhidrat_orani": "Kalorinin karbonhidrattan gelen payı",
    "seker_g": "Günlük şeker (g)",
    "lif_g": "Günlük lif (g)",
    "doymus_yag_orani": "Kalorinin doymuş yağdan gelen payı",
    "protein_orani": "Kalorinin proteinden gelen payı",
}


def oku(tablo, yil, ek):
    yol = NHANES_DIR / f"{dosya_adi(tablo, ek)}.xpt"
    return pd.read_sas(yol, format="xport", encoding="latin1").set_index("SEQN")


def evet_hayir(s):
    """NHANES kodlaması: 1=Evet, 2=Hayır, 7/9=Reddetti/Bilmiyor."""
    return s.map({1: 1.0, 2: 0.0})


def gecerli(s, ust):
    """'Bilmiyor / reddetti' kodlarını (777, 999 vb.) eksik yapar."""
    return s.where(s <= ust)


def donem_hazirla(yil, ek):
    t = {ad: oku(ad, yil, ek) for ad in ["DEMO", "BMX", "BPX", "GHB", "GLU", "DIQ", "MCQ", "BPQ", "PAQ", "SMQ", "ALQ", "HUQ", "SLQ", "RHQ", "DR1TOT"]}
    d = pd.DataFrame(index=t["DEMO"].index)
    demo = t["DEMO"]
    d["donem"] = DONEM_ADI[yil]
    d["yas"] = demo["RIDAGEYR"]
    d["erkek"] = (demo["RIAGENDR"] == 1).astype(float)
    d["etnik"] = demo["RIDRETH1"]
    d["gelir_orani"] = demo["INDFMPIR"]
    d["egitim"] = gecerli(demo["DMDEDUC2"], 5)
    d["gebe"] = demo["RIDEXPRG"] == 1

    bmx = t["BMX"].reindex(d.index)
    d["bmi"], d["bel"], d["boy"], d["kilo"] = bmx["BMXBMI"], bmx["BMXWAIST"], bmx["BMXHT"], bmx["BMXWT"]
    d["bel_boy_orani"] = d["bel"] / d["boy"]

    bp = t["BPX"].reindex(d.index)
    sis = [c for c in bp.columns if c.startswith(("BPXSY", "BPXOSY"))]
    dia = [c for c in bp.columns if c.startswith(("BPXDI", "BPXODI"))]
    d["sistolik"] = bp[sis].mean(axis=1)
    d["diastolik"] = bp[dia].replace(0, np.nan).mean(axis=1)

    diq = t["DIQ"].reindex(d.index)
    d["diyabet_tanili"] = diq["DIQ010"] == 1
    # "Sınırda diyabet" (DIQ010=3) diyenlere DIQ160 sorulmaz; boş bırakılırsa model bu boşluğu
    # gizli ipucu olarak öğrenir. Bu yüzden ikisini tek açık soruda birleştiriyoruz
    # (FINDRISC'teki "daha önce kan şekeriniz yüksek bulundu mu?" sorusunun karşılığı).
    d["prediyabet_tani"] = evet_hayir(diq["DIQ160"]).mask(diq["DIQ010"] == 3, 1.0)

    mcq = t["MCQ"].reindex(d.index)
    d["aile_diyabet"] = evet_hayir(mcq["MCQ300C"])
    d["kalp_hastaligi"] = (evet_hayir(mcq["MCQ160C"]).fillna(0) + evet_hayir(mcq["MCQ160E"]).fillna(0) + evet_hayir(mcq["MCQ160B"]).fillna(0)).clip(0, 1)
    d["inme"] = evet_hayir(mcq["MCQ160F"])

    bpq = t["BPQ"].reindex(d.index)
    d["tansiyon_tani"] = evet_hayir(bpq["BPQ020"])
    d["tansiyon_ilac"] = evet_hayir(bpq["BPQ040A"]).where(d["tansiyon_tani"] == 1, 0.0)
    d["kolesterol_tani"] = evet_hayir(bpq["BPQ080"])

    rhq = t["RHQ"].reindex(d.index)
    d["gebelik_diyabeti"] = evet_hayir(rhq["RHQ162"]).where(d["erkek"] == 0, 0.0).fillna(0.0)

    paq = t["PAQ"].reindex(d.index)
    d["agir_aktivite"] = ((paq["PAQ605"] == 1) | (paq["PAQ650"] == 1)).astype(float)
    d["orta_aktivite"] = ((paq["PAQ620"] == 1) | (paq["PAQ665"] == 1)).astype(float)
    d["aktif"] = ((d["agir_aktivite"] + d["orta_aktivite"]) > 0).astype(float)
    d["oturma_dk"] = gecerli(paq["PAD680"], 1440)

    smq = t["SMQ"].reindex(d.index)
    d["sigara_hic"] = evet_hayir(smq["SMQ020"])
    d["sigara_simdi"] = smq["SMQ040"].map({1: 1.0, 2: 1.0, 3: 0.0}).where(d["sigara_hic"] == 1, 0.0)

    d["alkol_gunluk"] = gecerli(t["ALQ"].reindex(d.index)["ALQ130"], 50)
    d["genel_saglik"] = gecerli(t["HUQ"].reindex(d.index)["HUQ010"], 5)
    slq = t["SLQ"].reindex(d.index)
    d["uyku_saat"] = gecerli(slq["SLD012"] if "SLD012" in slq else slq["SLD010H"], 24)

    dr = t["DR1TOT"].reindex(d.index)
    dr = dr[dr["DR1DRSTZ"] == 1].reindex(d.index)  # 1 = güvenilir ve eksiksiz kayıt
    kcal = dr["DR1TKCAL"].where(dr["DR1TKCAL"] > 300)
    d["kalori"] = kcal
    d["karbonhidrat_orani"] = 4 * dr["DR1TCARB"] / kcal
    d["seker_g"] = dr["DR1TSUGR"].where(kcal.notna())
    d["lif_g"] = dr["DR1TFIBE"].where(kcal.notna())
    d["doymus_yag_orani"] = 9 * dr["DR1TSFAT"] / kcal
    d["protein_orani"] = 4 * dr["DR1TPROT"] / kcal

    d["hba1c"] = t["GHB"].reindex(d.index)["LBXGH"]
    d["aclik_seker"] = t["GLU"].reindex(d.index)["LBXGLU"]
    return d


def ada_puani(d):
    """ADA Diyabet Risk Testi (Bang ve ark. 2009). 0-11 puan, >= 5 riskli kabul edilir."""
    p = pd.cut(d["yas"], [0, 39.99, 49.99, 59.99, 200], labels=[0, 1, 2, 3]).astype(float)
    p += d["erkek"]
    p += d["gebelik_diyabeti"]
    p += d["aile_diyabet"].fillna(0)
    p += ((d["tansiyon_tani"] == 1) | (d["tansiyon_ilac"] == 1)).astype(float)
    p += 1 - d["aktif"]
    p += pd.cut(d["bmi"], [0, 24.99, 29.99, 39.99, 200], labels=[0, 1, 2, 3]).astype(float)
    return p


def findrisc_puani(d):
    """FINDRISC (Lindström ve Tuomilehto, 2003), 0-26 puan.

    NHANES'te karşılığı olmayan iki uyarlama:
      - "Her gün sebze/meyve yer misiniz?" sorusu yok -> bu madde 0 puan sayılır.
      - Aile öyküsü anne-baba / büyükanne ayrımı yapmaz -> birinci derece (5 puan) kabul edilir.
    "Günde en az 30 dk aktivite" yerine ağır veya orta düzey aktivite yapıp yapmama kullanılır.
    """
    yas = pd.cut(d["yas"], [0, 44.99, 54.99, 64.99, 200], labels=[0, 2, 3, 4]).astype(float)
    bmi = pd.cut(d["bmi"], [0, 24.99, 30, 200], labels=[0, 1, 3]).astype(float)
    bel_e = pd.cut(d["bel"], [0, 93.99, 102, 400], labels=[0, 3, 4]).astype(float)
    bel_k = pd.cut(d["bel"], [0, 79.99, 88, 400], labels=[0, 3, 4]).astype(float)
    bel = bel_e.where(d["erkek"] == 1, bel_k).fillna(0)
    return (yas + bmi + bel + 2 * (1 - d["aktif"]) + 2 * d["tansiyon_ilac"].fillna(0)
            + 5 * d["prediyabet_tani"].fillna(0) + 5 * d["aile_diyabet"].fillna(0))


def main():
    parcalar = [donem_hazirla(y, e) for y, e in DONEMLER.items()]
    tum = pd.concat(parcalar)
    rapor = {"toplam_kisi": int(len(tum))}

    d = tum[(tum["yas"] >= 20) & ~tum["gebe"]]
    rapor["yetiskin_gebe_degil"] = int(len(d))
    d = d[d["hba1c"].notna() & d["bmi"].notna()]
    rapor["tahlili_olan"] = int(len(d))
    rapor["daha_once_tanili"] = int(d["diyabet_tanili"].sum())
    d = d[~d["diyabet_tanili"]].copy()
    rapor["model_kitlesi"] = int(len(d))

    seker_yuksek = d["aclik_seker"].fillna(0)
    d["tani_konmamis_diyabet"] = ((d["hba1c"] >= 6.5) | (seker_yuksek >= 126)).astype(int)
    d["disglisemi"] = ((d["hba1c"] >= 5.7) | (seker_yuksek >= 100)).astype(int)
    d["ada_puani"] = ada_puani(d)
    d["findrisc_puani"] = findrisc_puani(d)

    rapor["tani_konmamis_diyabet_orani"] = float(d["tani_konmamis_diyabet"].mean())
    rapor["disglisemi_orani"] = float(d["disglisemi"].mean())
    rapor["donem_dagilimi"] = d["donem"].value_counts().sort_index().to_dict()
    rapor["eksik_oranlari"] = {k: round(float(d[k].isna().mean()), 3) for k in NHANES_OZELLIKLER}

    rapor["diyet_kaydi_olan"] = float(d["kalori"].notna().mean())
    sutunlar = ["donem"] + list(NHANES_OZELLIKLER) + list(DIYET_OZELLIKLER) + ["aktif", "ada_puani", "findrisc_puani", "hba1c", "aclik_seker", "tani_konmamis_diyabet", "disglisemi"]
    d[sutunlar].to_csv(CIKTI)
    RAPOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAPOR_DIR / "nhanes_veri_ozeti.json", "w", encoding="utf-8") as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rapor.items() if k != "eksik_oranlari"}, indent=2, ensure_ascii=False))
    print("En çok eksik olanlar:", sorted(rapor["eksik_oranlari"].items(), key=lambda x: -x[1])[:6])
    print(f"Kaydedildi: {CIKTI}")


if __name__ == "__main__":
    main()

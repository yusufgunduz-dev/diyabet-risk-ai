"""NHANES 1. adım: CDC NHANES verilerini indirir (2007-2020, 6 dönem).

NHANES, BRFSS'ten farklı olarak anketin yanında fiziksel ölçüm (bel çevresi,
tansiyon) ve kan tahlili (HbA1c, açlık şekeri) içerir.

Çalıştırma:
    .venv/Scripts/python -m backend.ml.nhanes_indir
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor

from backend.ml.ozellikler import BACKEND_DIR

NHANES_DIR = BACKEND_DIR / "data" / "nhanes"
TEMEL_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{yil}/DataFiles/{dosya}.xpt"

# dönem başlangıç yılı -> dosya son eki. 2017-2020 dönemi "P_" önekini kullanır
# (COVID nedeniyle 2017-2018 ve 2019-Mart 2020 birleştirilmiştir).
DONEMLER = {2007: "_E", 2009: "_F", 2011: "_G", 2013: "_H", 2015: "_I", 2017: "P_"}

TABLOLAR = [
    "DEMO",  # yaş, cinsiyet, etnik köken, gelir, eğitim
    "BMX",   # boy, kilo, BMI, bel çevresi (ölçüm)
    "BPX",   # tansiyon ölçümü (2017-2020'de BPXO)
    "GHB",   # HbA1c (kan tahlili)
    "GLU",   # açlık kan şekeri (alt örneklem)
    "DIQ",   # "doktor size diyabet dedi mi?"
    "MCQ",   # ailede diyabet öyküsü
    "BPQ",   # tansiyon / kolesterol tanısı
    "PAQ",   # fiziksel aktivite
    "SMQ",   # sigara
    "ALQ",   # alkol
    "HUQ",   # genel sağlık algısı
    "SLQ",   # uyku
    "RHQ",   # gebelik diyabeti (kadınlar)
    "DR1TOT",  # 24 saatlik beslenme kaydı (1. gün toplamları)
]


def dosya_adi(tablo, ek):
    if ek == "P_":
        return f"P_{'BPXO' if tablo == 'BPX' else tablo}"
    return f"{tablo}{ek}"


def indir(yil, tablo, ek):
    ad = dosya_adi(tablo, ek)
    hedef = NHANES_DIR / f"{ad}.xpt"
    if hedef.exists() and hedef.stat().st_size > 10_000:
        return f"  var    {ad}"
    url = TEMEL_URL.format(yil=yil, dosya=ad)
    subprocess.run(["curl", "-sSL", "--fail", "-o", str(hedef), url], check=True)
    return f"  indi   {ad} ({hedef.stat().st_size / 1e6:.1f} MB)"


def main():
    NHANES_DIR.mkdir(parents=True, exist_ok=True)
    isler = [(y, t, e) for y, e in DONEMLER.items() for t in TABLOLAR]
    with ThreadPoolExecutor(max_workers=6) as havuz:
        for satir in havuz.map(lambda a: indir(*a), isler):
            print(satir, flush=True)
    toplam = sum(p.stat().st_size for p in NHANES_DIR.glob("*.xpt")) / 1e6
    print(f"{len(isler)} dosya hazır, toplam {toplam:.0f} MB -> {NHANES_DIR}")


if __name__ == "__main__":
    main()

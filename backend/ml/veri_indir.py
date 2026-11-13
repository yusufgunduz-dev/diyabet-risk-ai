"""1. adım: CDC Diabetes Health Indicators verisini UCI'dan indirir.

Çalıştırma (proje kök dizininden):
    .venv/Scripts/python -m backend.ml.veri_indir

Not: Bazı Windows kurulumlarında Python'un sertifika paketi UCI'ya bağlanırken
hata verir. Bu durumda dosya Windows'un kendi curl'üyle indirilir.
"""

import subprocess

import pandas as pd

from backend.ml.ozellikler import HEDEF, TUM_OZELLIKLER, VERI_DOSYASI

TUM_ADLAR = [o["ad"] for o in TUM_OZELLIKLER]

UCI_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
HAM_DOSYA = VERI_DOSYASI.parent / "ham" / "cdc_891.csv"


def indir():
    if HAM_DOSYA.exists():
        print(f"Ham dosya zaten var: {HAM_DOSYA}")
        return pd.read_csv(HAM_DOSYA)
    HAM_DOSYA.parent.mkdir(parents=True, exist_ok=True)
    print(f"İndiriliyor: {UCI_URL}")
    try:
        pd.read_csv(UCI_URL).to_csv(HAM_DOSYA, index=False)
    except Exception as hata:
        print(f"Python ile indirilemedi ({hata.__class__.__name__}), curl deneniyor...")
        subprocess.run(["curl", "-sSL", "-o", str(HAM_DOSYA), UCI_URL], check=True)
    return pd.read_csv(HAM_DOSYA)


def main():
    df = indir()
    eksik = set(TUM_ADLAR + [HEDEF]) - set(df.columns)
    if eksik:
        raise ValueError(f"Beklenen sütunlar bulunamadı: {eksik}")

    df = df[TUM_ADLAR + [HEDEF]].astype(float)
    df.to_csv(VERI_DOSYASI, index=False)
    print(f"Kaydedildi: {VERI_DOSYASI}  ({len(df):,} satır, {len(TUM_ADLAR)} özellik)")


if __name__ == "__main__":
    main()

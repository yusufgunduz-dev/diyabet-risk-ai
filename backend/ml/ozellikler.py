"""Veri setindeki özelliklerin tanımları.

Hem eğitim scriptleri hem de API bu dosyayı kullanır; böylece özellik sırası
ve Türkçe açıklamalar tek yerden yönetilir.
Kaynak: CDC BRFSS 2015 - Diabetes Health Indicators (UCI ID: 891)
"""

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
VERI_DOSYASI = BACKEND_DIR / "data" / "diabetes_cdc.csv"
MODEL_DIR = BACKEND_DIR / "models"
RAPOR_DIR = BACKEND_DIR / "reports"
GRAFIK_DIR = RAPOR_DIR / "figures"

HEDEF = "Diabetes_binary"
RASTGELE = 42

EVET_HAYIR = [{"deger": 0, "etiket": "Hayır"}, {"deger": 1, "etiket": "Evet"}]

YAS_GRUPLARI = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
    "55-59", "60-64", "65-69", "70-74", "75-79", "80+",
]

EGITIM = [
    "Okula gitmedi", "İlkokul (1-8. sınıf)", "Lise terk", "Lise mezunu",
    "Üniversite (1-3 yıl)", "Üniversite mezunu",
]

GELIR = [
    "1 (en düşük)", "2", "3", "4", "5", "6", "7", "8 (en yüksek)",
]

GENEL_SAGLIK = ["Mükemmel", "Çok iyi", "İyi", "Orta", "Kötü"]

# Veri setindeki 21 özelliğin tamamı (keşif analizi bunları kullanır).
TUM_OZELLIKLER = [
    {"ad": "HighBP", "etiket": "Yüksek tansiyon", "soru": "Doktor size yüksek tansiyon tanısı koydu mu?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "HighChol", "etiket": "Yüksek kolesterol", "soru": "Doktor size yüksek kolesterol tanısı koydu mu?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "CholCheck", "etiket": "Kolesterol kontrolü", "soru": "Son 5 yılda kolesterolünüzü ölçtürdünüz mü?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "BMI", "etiket": "Vücut kitle indeksi", "soru": "Vücut kitle indeksiniz (boy ve kilodan hesaplanır)", "tip": "sayi", "min": 12, "max": 98},
    {"ad": "Smoker", "etiket": "Sigara geçmişi", "soru": "Hayatınız boyunca en az 100 sigara (5 paket) içtiniz mi?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "Stroke", "etiket": "İnme geçmişi", "soru": "Hiç inme (felç) geçirdiniz mi?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "HeartDiseaseorAttack", "etiket": "Kalp hastalığı", "soru": "Koroner kalp hastalığı veya kalp krizi geçirdiniz mi?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "PhysActivity", "etiket": "Fiziksel aktivite", "soru": "Son 30 günde iş dışında spor / fiziksel aktivite yaptınız mı?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "Fruits", "etiket": "Meyve tüketimi", "soru": "Günde en az 1 kez meyve yer misiniz?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "Veggies", "etiket": "Sebze tüketimi", "soru": "Günde en az 1 kez sebze yer misiniz?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "HvyAlcoholConsump", "etiket": "Yoğun alkol", "soru": "Yoğun alkol tüketiyor musunuz? (Erkek: haftada 14+, Kadın: 7+ kadeh)", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "AnyHealthcare", "etiket": "Sağlık sigortası", "soru": "Herhangi bir sağlık sigortanız var mı?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "NoDocbcCost", "etiket": "Maddi engel", "soru": "Son 12 ayda parasızlık yüzünden doktora gidemediğiniz oldu mu?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "GenHlth", "etiket": "Genel sağlık algısı", "soru": "Genel sağlığınızı nasıl değerlendirirsiniz?", "tip": "secim", "secenekler": [{"deger": i + 1, "etiket": e} for i, e in enumerate(GENEL_SAGLIK)]},
    {"ad": "MentHlth", "etiket": "Ruhsal sağlık (gün)", "soru": "Son 30 günde kaç gün ruhsal sağlığınız iyi değildi?", "tip": "sayi", "min": 0, "max": 30},
    {"ad": "PhysHlth", "etiket": "Fiziksel sağlık (gün)", "soru": "Son 30 günde kaç gün fiziksel sağlığınız iyi değildi?", "tip": "sayi", "min": 0, "max": 30},
    {"ad": "DiffWalk", "etiket": "Yürüme güçlüğü", "soru": "Yürürken veya merdiven çıkarken ciddi zorluk yaşıyor musunuz?", "tip": "secim", "secenekler": EVET_HAYIR},
    {"ad": "Sex", "etiket": "Cinsiyet", "soru": "Cinsiyetiniz", "tip": "secim", "secenekler": [{"deger": 0, "etiket": "Kadın"}, {"deger": 1, "etiket": "Erkek"}]},
    {"ad": "Age", "etiket": "Yaş grubu", "soru": "Yaş grubunuz", "tip": "secim", "secenekler": [{"deger": i + 1, "etiket": e} for i, e in enumerate(YAS_GRUPLARI)]},
    {"ad": "Education", "etiket": "Eğitim düzeyi", "soru": "Eğitim düzeyiniz", "tip": "secim", "secenekler": [{"deger": i + 1, "etiket": e} for i, e in enumerate(EGITIM)]},
    {"ad": "Income", "etiket": "Gelir düzeyi", "soru": "Gelir düzeyiniz (8 basamaklı ölçek)", "tip": "secim", "secenekler": [{"deger": i + 1, "etiket": e} for i, e in enumerate(GELIR)]},
]

# Modelden çıkarılan sosyoekonomik sorular. Diyabetle ilişkililer ama etkileri zaten kilo,
# tansiyon ve genel sağlık sorularına yansıdığı için dördü birlikte AUC'ye yalnızca ~0.001
# katkı yapıyor; sigorta sorusu ayrıca ABD'ye özgü. Ölçüm: model_egitimi.sosyoekonomik_ablasyon.
CIKARILAN_OZELLIKLER = ["AnyHealthcare", "NoDocbcCost", "Education", "Income"]

# Modelin kullandığı 17 özellik. Sıra, modelin beklediği sütun sırasıdır.
OZELLIKLER = [o for o in TUM_OZELLIKLER if o["ad"] not in CIKARILAN_OZELLIKLER]
OZELLIK_ADLARI = [o["ad"] for o in OZELLIKLER]
TURKCE_ETIKET = {o["ad"]: o["etiket"] for o in TUM_OZELLIKLER}

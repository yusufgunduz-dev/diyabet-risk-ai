"""3. adım: Model eğitimi ve karşılaştırma - Hafta 4-7 ve 11-13.

Ne yapar?
  1. Tekrar eden satırları siler, veriyi eğitim / doğrulama / test olarak böler.
  2. Sınıf dengesizliği için 3 yaklaşımı karşılaştırır (hiçbir şey, class_weight, SMOTE).
  3. 7 klasik modeli GridSearchCV + 5-katlı Stratified K-Fold ile ayarlar.
  4. Keras ile bir yapay sinir ağı (MLP) eğitir.
  5. Hepsini aynı test setinde ölçer, grafikleri ve sonuç tablosunu kaydeder.
  6. Canlı uygulamada kullanılacak modeli (XGBoost) kalibre eder, eşik seçer ve kaydeder.

Çalıştırma:
    .venv/Scripts/python -m backend.ml.model_egitimi
"""

import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from backend.ml.ozellikler import (
    CIKARILAN_OZELLIKLER, GRAFIK_DIR, HEDEF, MODEL_DIR, OZELLIK_ADLARI, RAPOR_DIR, RASTGELE,
    TURKCE_ETIKET, VERI_DOSYASI,
)

sns.set_theme(style="whitegrid", font_scale=1.05)
AYAR_ORNEK = 30_000   # hiperparametre aramasında kullanılan alt örneklem
SVM_ORNEK = 20_000    # SVM eğitimi O(n^2) olduğu için alt örneklem


def log(mesaj):
    print(f"[{time.strftime('%H:%M:%S')}] {mesaj}", flush=True)


def kaydet(ad):
    plt.tight_layout()
    plt.savefig(GRAFIK_DIR / ad, dpi=150)
    plt.close()


def alt_ornek(X, y, n):
    if n >= len(X):
        return X, y
    X_a, _, y_a, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=RASTGELE)
    return X_a, y_a


def metrikler(y, skor, tahmin):
    return {
        "accuracy": accuracy_score(y, tahmin),
        "precision": precision_score(y, tahmin, zero_division=0),
        "recall": recall_score(y, tahmin),
        "f1": f1_score(y, tahmin),
        "roc_auc": roc_auc_score(y, skor),
        "pr_auc": average_precision_score(y, skor),
    }


def veri_hazirla():
    df = pd.read_csv(VERI_DOSYASI)
    once = len(df)
    df = df.drop_duplicates()
    log(f"Tekrar eden {once - len(df):,} satır silindi, kalan {len(df):,}")
    X = df[OZELLIK_ADLARI + CIKARILAN_OZELLIKLER]  # çıkarılanlar yalnızca ablasyon için
    y = df[HEDEF].astype(int)
    # %70 eğitim, %10 doğrulama, %20 test
    X_gecici, X_test, y_gecici, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RASTGELE)
    X_egitim, X_dog, y_egitim, y_dog = train_test_split(X_gecici, y_gecici, test_size=0.125, stratify=y_gecici, random_state=RASTGELE)
    log(f"Eğitim {len(X_egitim):,} | Doğrulama {len(X_dog):,} | Test {len(X_test):,} | Diyabet oranı %{100 * y.mean():.1f}")
    return X_egitim, X_dog, X_test, y_egitim, y_dog, y_test, len(df), once - len(df)


def sosyoekonomik_ablasyon(X_egitim, y_egitim, X_dog, y_dog):
    """Çıkarılan 4 sosyoekonomik sorunun modele katkısını doğrulama setinde ölçer."""
    log("Sosyoekonomik soru ablasyonu (XGBoost, doğrulama seti)")
    xgb = lambda: XGBClassifier(tree_method="hist", max_depth=3, learning_rate=0.03, n_estimators=300, subsample=0.8, n_jobs=-1, random_state=RASTGELE)
    auc = lambda sutun: roc_auc_score(y_dog, xgb().fit(X_egitim[sutun], y_egitim).predict_proba(X_dog[sutun])[:, 1])
    tumu = OZELLIK_ADLARI + CIKARILAN_OZELLIKLER
    sonuc = {"21_soru": float(auc(tumu)), "17_soru": float(auc(OZELLIK_ADLARI))}
    sonuc["tek_tek_katki"] = {o: float(sonuc["21_soru"] - auc([c for c in tumu if c != o])) for o in CIKARILAN_OZELLIKLER}
    log(f"  21 soru AUC={sonuc['21_soru']:.4f}  17 soru AUC={sonuc['17_soru']:.4f}  fark={sonuc['21_soru'] - sonuc['17_soru']:.4f}")
    return sonuc


def dengesizlik_deneyi(X_egitim, y_egitim, X_test, y_test):
    """Hafta 2/7: dengesiz veride 3 yaklaşımı lojistik regresyonla karşılaştırır."""
    log("Dengesizlik deneyi (Lojistik Regresyon ile)")
    deneyler = {
        "Hiçbir şey yapma": Pipeline([("olcek", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
        "class_weight='balanced'": Pipeline([("olcek", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "SMOTE": ImbPipeline([("olcek", StandardScaler()), ("smote", SMOTE(random_state=RASTGELE)), ("model", LogisticRegression(max_iter=2000))]),
    }
    sonuc = {}
    for ad, pipe in deneyler.items():
        pipe.fit(X_egitim, y_egitim)
        skor = pipe.predict_proba(X_test)[:, 1]
        sonuc[ad] = metrikler(y_test, skor, (skor >= 0.5).astype(int))
        log(f"  {ad:26s} accuracy={sonuc[ad]['accuracy']:.3f} recall={sonuc[ad]['recall']:.3f} f1={sonuc[ad]['f1']:.3f}")

    tablo = pd.DataFrame(sonuc).T[["accuracy", "recall", "precision", "f1"]]
    tablo.plot(kind="bar", figsize=(9, 4.5), color=["#8C9BB0", "#DD5555", "#4C72B0", "#55A868"])
    plt.title("Dengesiz veri: yaklaşım karşılaştırması (Lojistik Regresyon)")
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.legend(["Doğruluk", "Recall (duyarlılık)", "Precision (kesinlik)", "F1"], loc="upper right")
    kaydet("10_dengesizlik_karsilastirma.png")
    return sonuc


def asiri_ogrenme_egrisi(X_egitim, y_egitim, X_dog, y_dog):
    """Hafta 7: karar ağacı derinliği arttıkça overfitting'i gösterir."""
    log("Overfitting eğrisi (karar ağacı derinliği)")
    derinlikler = list(range(1, 26))
    egitim_auc, dog_auc = [], []
    for d in derinlikler:
        agac = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=RASTGELE).fit(X_egitim, y_egitim)
        egitim_auc.append(roc_auc_score(y_egitim, agac.predict_proba(X_egitim)[:, 1]))
        dog_auc.append(roc_auc_score(y_dog, agac.predict_proba(X_dog)[:, 1]))
    en_iyi = derinlikler[int(np.argmax(dog_auc))]
    plt.figure(figsize=(8, 4.5))
    plt.plot(derinlikler, egitim_auc, "o-", label="Eğitim", color="#4C72B0")
    plt.plot(derinlikler, dog_auc, "o-", label="Doğrulama", color="#DD5555")
    plt.axvline(en_iyi, ls="--", color="gray")
    plt.text(en_iyi + 0.3, min(dog_auc), f"en iyi derinlik = {en_iyi}", color="gray")
    plt.axvspan(0.5, 3.5, color="#4C72B0", alpha=0.08)
    plt.axvspan(14, 25.5, color="#DD5555", alpha=0.08)
    plt.text(1, max(egitim_auc), "Underfitting", fontsize=9)
    plt.text(17, max(egitim_auc), "Overfitting", fontsize=9)
    plt.xlabel("Karar ağacı maksimum derinliği")
    plt.ylabel("ROC-AUC")
    plt.title("Underfitting / Overfitting: karar ağacı")
    plt.legend()
    kaydet("11_overfitting_egrisi.png")
    return {"derinlik": derinlikler, "egitim_auc": egitim_auc, "dogrulama_auc": dog_auc, "en_iyi_derinlik": en_iyi}


def klasik_modeller(oran):
    """(ad, tahminci, parametre ızgarası, eğitim verisi boyutu, sample_weight kullanılsın mı)"""
    return [
        ("Lojistik Regresyon",
         Pipeline([("olcek", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
         {"model__C": [0.01, 0.1, 1, 10]}, None, False),
        ("KNN",
         ImbPipeline([("olcek", StandardScaler()), ("alt", RandomUnderSampler(random_state=RASTGELE)), ("model", KNeighborsClassifier(n_jobs=-1))]),
         {"model__n_neighbors": [15, 35, 75, 125], "model__weights": ["uniform", "distance"]}, None, False),
        ("SVM (RBF çekirdek)",
         Pipeline([("olcek", StandardScaler()), ("model", SVC(kernel="rbf", class_weight="balanced", cache_size=1000))]),
         {"model__C": [0.3, 1, 3], "model__gamma": ["scale", 0.01]}, SVM_ORNEK, False),
        ("Karar Ağacı",
         DecisionTreeClassifier(class_weight="balanced", random_state=RASTGELE),
         {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [20, 100, 300]}, None, False),
        ("Random Forest",
         RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=RASTGELE),
         {"max_depth": [8, 12, 16], "min_samples_leaf": [5, 20]}, None, False),
        ("AdaBoost",
         AdaBoostClassifier(random_state=RASTGELE),
         {"n_estimators": [100, 300], "learning_rate": [0.5, 1.0]}, None, True),
        ("XGBoost (Gradient Boosting)",
         XGBClassifier(tree_method="hist", scale_pos_weight=oran, eval_metric="logloss", n_jobs=-1, random_state=RASTGELE),
         {"max_depth": [3, 5], "n_estimators": [300, 600], "learning_rate": [0.03, 0.08], "subsample": [0.8]}, None, False),
    ]


def skor_al(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def keras_mlp(X_egitim, y_egitim, X_dog, y_dog):
    """Hafta 11-13: Keras ile ileri beslemeli yapay sinir ağı."""
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(RASTGELE)
    olcek = StandardScaler().fit(X_egitim)
    Xe, Xd = olcek.transform(X_egitim), olcek.transform(X_dog)
    agirlik = compute_sample_weight("balanced", y_egitim)
    sinif_agirligi = {0: float(agirlik[y_egitim.values == 0][0]), 1: float(agirlik[y_egitim.values == 1][0])}

    model = keras.Sequential([
        keras.layers.Input(shape=(Xe.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
    durdur = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)
    gecmis = model.fit(Xe, y_egitim, validation_data=(Xd, y_dog), epochs=60, batch_size=512,
                       class_weight=sinif_agirligi, callbacks=[durdur], verbose=0)
    h = gecmis.history

    fig, eksen = plt.subplots(1, 2, figsize=(11, 4))
    eksen[0].plot(h["loss"], label="Eğitim")
    eksen[0].plot(h["val_loss"], label="Doğrulama")
    eksen[0].set_title("Kayıp (loss)")
    eksen[0].set_xlabel("Epoch")
    eksen[1].plot(h["auc"], label="Eğitim")
    eksen[1].plot(h["val_auc"], label="Doğrulama")
    eksen[1].set_title("ROC-AUC")
    eksen[1].set_xlabel("Epoch")
    for e in eksen:
        e.legend()
    fig.suptitle("Yapay sinir ağı eğitim eğrileri (early stopping ile)")
    kaydet("12_mlp_egitim_egrileri.png")

    class Sarmalayici:
        def predict_proba(self, X):
            p = model.predict(olcek.transform(X), verbose=0, batch_size=4096).ravel()
            return np.column_stack([1 - p, p])

    mimari = f"Girdi({Xe.shape[1]}) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)"
    return Sarmalayici(), {"epoch": len(h["loss"]), "mimari": mimari}


def main():
    GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X_egitim, X_dog, X_test, y_egitim, y_dog, y_test, n_temiz, n_tekrar = veri_hazirla()
    oran = float((y_egitim == 0).sum() / (y_egitim == 1).sum())

    sosyoekonomik = sosyoekonomik_ablasyon(X_egitim, y_egitim, X_dog, y_dog)
    X_egitim, X_dog, X_test = X_egitim[OZELLIK_ADLARI], X_dog[OZELLIK_ADLARI], X_test[OZELLIK_ADLARI]

    dengesizlik = dengesizlik_deneyi(X_egitim, y_egitim, X_test, y_test)
    overfit = asiri_ogrenme_egrisi(X_egitim, y_egitim, X_dog, y_dog)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RASTGELE)
    X_ayar, y_ayar = alt_ornek(X_egitim, y_egitim, AYAR_ORNEK)
    sonuclar, egitilmis, skorlar = [], {}, {}

    for ad, tahminci, izgara, boyut, agirlikli in klasik_modeller(oran):
        log(f"{ad}: GridSearchCV (5-katlı) başlıyor")
        bas = time.time()
        Xa, ya = alt_ornek(X_ayar, y_ayar, 10_000 if boyut else AYAR_ORNEK)
        ek = {"sample_weight": compute_sample_weight("balanced", ya)} if agirlikli else {}
        arama = GridSearchCV(tahminci, izgara, scoring="roc_auc", cv=kfold, n_jobs=-1 if "KNN" not in ad else 1)
        arama.fit(Xa, ya, **ek)
        cv_ort = float(arama.best_score_)
        cv_std = float(arama.cv_results_["std_test_score"][arama.best_index_])
        log(f"  en iyi {arama.best_params_}  CV AUC={cv_ort:.4f} ± {cv_std:.4f}")

        Xf, yf = alt_ornek(X_egitim, y_egitim, boyut) if boyut else (X_egitim, y_egitim)
        model = arama.best_estimator_
        ek = {"sample_weight": compute_sample_weight("balanced", yf)} if agirlikli else {}
        model.fit(Xf, yf, **ek)
        sure = time.time() - bas

        skor = skor_al(model, X_test)
        tahmin = model.predict(X_test)
        m = metrikler(y_test, skor, tahmin)
        sonuclar.append({"model": ad, "cv_auc": cv_ort, "cv_std": cv_std, "en_iyi_parametreler": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in arama.best_params_.items()},
                         "egitim_boyutu": int(len(Xf)), "sure_sn": round(sure, 1), **m, "karmasiklik": confusion_matrix(y_test, tahmin).tolist()})
        egitilmis[ad], skorlar[ad] = model, skor
        log(f"  TEST: AUC={m['roc_auc']:.4f} recall={m['recall']:.3f} f1={m['f1']:.3f} ({sure:.0f} sn)")

    log("Keras MLP eğitiliyor")
    bas = time.time()
    mlp, mlp_bilgi = keras_mlp(X_egitim, y_egitim, X_dog, y_dog)
    skor = mlp.predict_proba(X_test)[:, 1]
    tahmin = (skor >= 0.5).astype(int)
    m = metrikler(y_test, skor, tahmin)
    dog_auc = roc_auc_score(y_dog, mlp.predict_proba(X_dog)[:, 1])
    sonuclar.append({"model": "Yapay Sinir Ağı (Keras MLP)", "cv_auc": float(dog_auc), "cv_std": None, "en_iyi_parametreler": mlp_bilgi,
                     "egitim_boyutu": int(len(X_egitim)), "sure_sn": round(time.time() - bas, 1), **m, "karmasiklik": confusion_matrix(y_test, tahmin).tolist()})
    skorlar["Yapay Sinir Ağı (Keras MLP)"] = skor
    log(f"  TEST: AUC={m['roc_auc']:.4f} recall={m['recall']:.3f} ({mlp_bilgi['epoch']} epoch)")

    tablo = pd.DataFrame(sonuclar).sort_values("roc_auc", ascending=False)
    print(tablo[["model", "cv_auc", "roc_auc", "pr_auc", "accuracy", "recall", "precision", "f1", "sure_sn"]].round(4).to_string(index=False))

    # --- Grafikler ---
    plt.figure(figsize=(7.5, 6.5))
    for ad in tablo["model"]:
        fpr, tpr, _ = roc_curve(y_test, skorlar[ad])
        plt.plot(fpr, tpr, label=f"{ad} (AUC={roc_auc_score(y_test, skorlar[ad]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Rastgele tahmin (AUC=0.5)")
    plt.xlabel("Yanlış pozitif oranı")
    plt.ylabel("Doğru pozitif oranı (Recall)")
    plt.title("ROC eğrileri - test seti")
    plt.legend(fontsize=8, loc="lower right")
    kaydet("13_roc_egrileri.png")

    plt.figure(figsize=(7.5, 6.5))
    for ad in tablo["model"]:
        p, r, _ = precision_recall_curve(y_test, skorlar[ad])
        plt.plot(r, p, label=f"{ad} (AP={average_precision_score(y_test, skorlar[ad]):.3f})")
    plt.axhline(y_test.mean(), ls="--", color="k", alpha=0.5, label=f"Rastgele (%{100 * y_test.mean():.0f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall eğrileri - test seti")
    plt.legend(fontsize=8)
    kaydet("14_pr_egrileri.png")

    uzun = tablo.melt(id_vars="model", value_vars=["roc_auc", "recall", "f1", "accuracy"], var_name="Metrik", value_name="Değer")
    uzun["Metrik"] = uzun["Metrik"].map({"roc_auc": "ROC-AUC", "recall": "Recall", "f1": "F1", "accuracy": "Doğruluk"})
    plt.figure(figsize=(12, 5))
    sns.barplot(data=uzun, x="model", y="Değer", hue="Metrik", palette="Set2")
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.title("Model karşılaştırması - test seti")
    plt.legend(loc="lower right", ncol=4)
    kaydet("15_model_karsilastirma.png")

    # --- Canlı uygulama modeli: XGBoost + kalibrasyon + eşik ---
    log("Uygulama modeli hazırlanıyor (XGBoost + izotonik kalibrasyon)")
    xgb = egitilmis["XGBoost (Gradient Boosting)"]
    kalibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xgb.predict_proba(X_dog)[:, 1], y_dog)
    kal_dog = kalibrator.predict(xgb.predict_proba(X_dog)[:, 1])
    esikler = np.linspace(0.05, 0.6, 111)
    f2 = [fbeta_score(y_dog, (kal_dog >= t).astype(int), beta=2) for t in esikler]
    esik = float(esikler[int(np.argmax(f2))])
    kal_test = kalibrator.predict(xgb.predict_proba(X_test)[:, 1])
    tahmin = (kal_test >= esik).astype(int)
    uygulama = {**metrikler(y_test, kal_test, tahmin), "esik": esik, "karmasiklik": confusion_matrix(y_test, tahmin).tolist()}
    log(f"  eşik={esik:.3f} (F2 ile seçildi) TEST recall={uygulama['recall']:.3f} precision={uygulama['precision']:.3f}")

    kk = np.array(uygulama["karmasiklik"])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(kk, annot=True, fmt=",", cmap="Blues", cbar=False,
                xticklabels=["Tahmin: Yok", "Tahmin: Var"], yticklabels=["Gerçek: Yok", "Gerçek: Var"])
    plt.title(f"XGBoost karmaşıklık matrisi (eşik = {esik:.2f})")
    kaydet("16_karmasiklik_matrisi.png")

    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(6, 5))
    for ad, s in [("Ham XGBoost", xgb.predict_proba(X_test)[:, 1]), ("Kalibre edilmiş", kal_test)]:
        gercek, tahmini = calibration_curve(y_test, s, n_bins=15, strategy="quantile")
        plt.plot(tahmini, gercek, "o-", label=ad)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Mükemmel kalibrasyon")
    plt.xlabel("Modelin verdiği olasılık")
    plt.ylabel("Gerçekte diyabetli oranı")
    plt.title("Olasılık kalibrasyonu")
    plt.legend()
    kaydet("17_kalibrasyon.png")

    onem = pd.Series(xgb.feature_importances_, index=[TURKCE_ETIKET[o] for o in OZELLIK_ADLARI]).sort_values()
    plt.figure(figsize=(7, 7))
    onem.plot(kind="barh", color="#4C72B0")
    plt.title("XGBoost özellik önemi (gain)")
    kaydet("18_ozellik_onemi.png")

    joblib.dump({"model": xgb, "kalibrator": kalibrator, "esik": esik, "ozellikler": OZELLIK_ADLARI,
                 "taban_oran": float(y_egitim.mean())}, MODEL_DIR / "diyabet_modeli.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test, "X_egitim": X_egitim, "y_egitim": y_egitim}, MODEL_DIR / "bolunmus_veri.joblib")

    rapor = {
        "veri": {"temiz_satir": n_temiz, "silinen_tekrar": n_tekrar, "egitim": len(X_egitim), "dogrulama": len(X_dog), "test": len(X_test),
                 "diyabet_orani": float(y_egitim.mean())},
        "modeller": tablo.astype(object).where(tablo.notna(), None).to_dict(orient="records"),
        "sosyoekonomik_ablasyon": sosyoekonomik,
        "dengesizlik_deneyi": dengesizlik,
        "overfitting": overfit,
        "uygulama_modeli": {"ad": "XGBoost (Gradient Boosting) + izotonik kalibrasyon", **uygulama},
    }
    with open(RAPOR_DIR / "model_sonuclari.json", "w", encoding="utf-8") as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False, default=float)
    log("Bitti. Model: backend/models/diyabet_modeli.joblib")


if __name__ == "__main__":
    main()

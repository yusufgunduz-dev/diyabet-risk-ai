// Diyabet Risk Analizi arayüzü.
// Risk değerlendirmesi formunu /api/meta'dan kurar, /api/tahmin ile tahmin alır,
// model raporunu /api/sonuclar'dan doldurur. Ortak yardımcılar gizli.js tarafından da kullanılır.

const GRUPLAR = [
  { baslik: "Temel bilgiler", alanlar: ["Sex", "Age", "BMI"] },
  { baslik: "Sağlık geçmişi", alanlar: ["HighBP", "HighChol", "CholCheck", "HeartDiseaseorAttack", "Stroke", "DiffWalk", "GenHlth", "PhysHlth", "MentHlth"] },
  { baslik: "Yaşam tarzı", alanlar: ["PhysActivity", "Smoker", "Fruits", "Veggies", "HvyAlcoholConsump"] },
];

const ORNEKLER = {
  saglikli: { Sex: 0, Age: 2, boy: 168, kilo: 60, HighBP: 0, HighChol: 0, CholCheck: 1, HeartDiseaseorAttack: 0, Stroke: 0, DiffWalk: 0, GenHlth: 1, PhysHlth: 0, MentHlth: 2, PhysActivity: 1, Smoker: 0, Fruits: 1, Veggies: 1, HvyAlcoholConsump: 0 },
  orta: { Sex: 1, Age: 8, boy: 176, kilo: 95, HighBP: 1, HighChol: 0, CholCheck: 1, HeartDiseaseorAttack: 0, Stroke: 0, DiffWalk: 0, GenHlth: 3, PhysHlth: 3, MentHlth: 3, PhysActivity: 0, Smoker: 1, Fruits: 0, Veggies: 1, HvyAlcoholConsump: 0 },
  riskli: { Sex: 1, Age: 11, boy: 172, kilo: 112, HighBP: 1, HighChol: 1, CholCheck: 1, HeartDiseaseorAttack: 1, Stroke: 0, DiffWalk: 1, GenHlth: 4, PhysHlth: 15, MentHlth: 5, PhysActivity: 0, Smoker: 1, Fruits: 0, Veggies: 0, HvyAlcoholConsump: 0 },
};

const GRAFIK_ACIKLAMA = {
  "01_sinif_dagilimi": ["Sınıf dağılımı", "Kişilerin %14'ü diyabetli; veri dengesiz.", "veri"],
  "02_bmi_dagilimi": ["BMI dağılımı", "Diyabetlilerde vücut kitle indeksi daha yüksek.", "veri"],
  "03_yas_diyabet": ["Yaş ve diyabet", "Yaşla birlikte diyabet oranı artıyor.", "veri"],
  "04_risk_faktorleri": ["Risk faktörleri", "Faktör varken ve yokken diyabet oranı.", "veri"],
  "05_genel_saglik": ["Genel sağlık algısı", "Algılanan sağlık kötüleştikçe oran artıyor.", "veri"],
  "06_korelasyon": ["Korelasyon matrisi", "Özellikler arası doğrusal ilişkiler.", "veri"],
  "07_hedef_korelasyon": ["Diyabetle korelasyon", "Hedefle en ilişkili özellikler.", "veri"],
  "08_bmi_aykiri": ["BMI aykırı değerleri", "IQR yöntemiyle aykırı değer analizi.", "veri"],
  "10_dengesizlik_karsilastirma": ["Dengesiz veri", "Önlem alınmazsa duyarlılık %15'te kalıyor.", "model"],
  "11_overfitting_egrisi": ["Aşırı öğrenme", "Ağaç derinleştikçe doğrulama başarısı düşüyor.", "model"],
  "12_mlp_egitim_egrileri": ["Sinir ağı eğitimi", "Kayıp ve AUC eğrileri; erken durdurma.", "model"],
  "13_roc_egrileri": ["ROC eğrileri", "Sekiz modelin test seti ROC eğrileri.", "model"],
  "14_pr_egrileri": ["Precision-recall eğrileri", "Dengesiz veride tamamlayıcı ölçüt.", "model"],
  "15_model_karsilastirma": ["Model karşılaştırması", "ROC-AUC, duyarlılık, F1 ve doğruluk.", "model"],
  "16_karmasiklik_matrisi": ["Karmaşıklık matrisi", "Seçilen modelin test seti tahminleri.", "model"],
  "17_kalibrasyon": ["Olasılık kalibrasyonu", "İzotonik regresyon öncesi ve sonrası.", "model"],
  "18_ozellik_onemi": ["Özellik önemi", "XGBoost kazanç tabanlı önem.", "model"],
  "19_ogrenme_egrisi": ["Veri miktarı ve eğitim süresi", "Belirli bir noktadan sonra kazanç yok.", "model"],
  "20_shap_ozet": ["SHAP özet", "Özellik değerlerinin tahmine etkisi.", "aciklama"],
  "21_shap_bar": ["SHAP önem sırası", "Ortalama mutlak SHAP değerleri.", "aciklama"],
  "22_kmeans_k_secimi": ["Küme sayısı seçimi", "Dirsek ve siluet yöntemleri.", "kume"],
  "23_kume_profilleri": ["Risk profilleri", "K-Means kümelerinin diyabet oranları.", "kume"],
  "24_pca_varyans": ["PCA varyansı", "Bileşen sayısına göre açıklanan varyans.", "kume"],
  "25_pca_2d": ["PCA iki boyut", "Kümeler ve gerçek sınıflar.", "kume"],
  "26_tsne_2d": ["t-SNE iki boyut", "Doğrusal olmayan boyut indirgeme.", "kume"],
  "27_dendrogram": ["Dendrogram", "Hiyerarşik kümeleme (Ward).", "kume"],
  "30_nhanes_tani_roc": ["Model ve ADA: ROC", "Teşhis edilmemiş diyabet, 2017–20.", "nhanes"],
  "30_nhanes_disglisemi_roc": ["Model ve ADA: disglisemi", "Prediyabet ve teşhis edilmemiş diyabet.", "nhanes"],
  "31_nhanes_tani_ablasyon": ["Bilgi setlerinin katkısı", "%95 güven aralıklarıyla AUC.", "nhanes"],
  "31_nhanes_disglisemi_ablasyon": ["Katkı: disglisemi", "Bilgi setlerine göre AUC.", "nhanes"],
  "32_nhanes_tani_shap": ["SHAP: teşhis edilmemiş diyabet", "En etkili özellikler.", "nhanes"],
  "32_nhanes_disglisemi_shap": ["SHAP: disglisemi", "En etkili özellikler.", "nhanes"],
  "33_gelistirme_eleme": ["Geliştirme adaylarının elenmesi", "Katlara göre AUC kazancı.", "nhanes"],
  "34_adalet_analizi": ["Alt grup analizi", "Model, FINDRISC ve ADA.", "nhanes"],
};
const FILTRELER = { hepsi: "Tümü", veri: "Veri", model: "Modeller", aciklama: "SHAP", kume: "Kümeleme", nhanes: "NHANES" };

// ---------- Ortak yardımcılar ----------
const $ = (s, e = document) => e.querySelector(s);
const yuzde = (x, b = 1) => (100 * x).toLocaleString("tr-TR", { maximumFractionDigits: b, minimumFractionDigits: b });
const sayi = (x, b = 3) => x.toLocaleString("tr-TR", { maximumFractionDigits: b, minimumFractionDigits: b });
const binlik = (x) => x.toLocaleString("tr-TR");
const AZ_HAREKET = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function secimKutusu(secenekler, onDegis) {
  const kap = document.createElement("div");
  kap.className = "secim";
  kap.setAttribute("role", "radiogroup");
  kap.innerHTML = secenekler.map((s) => `<button type="button" role="radio" data-deger="${s.deger}">${s.etiket}</button>`).join("");
  const sec = (b) => kap.querySelectorAll("button").forEach((x) => {
    x.classList.toggle("secili", x === b);
    x.setAttribute("aria-checked", x === b);
  });
  kap.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { sec(b); onDegis && onDegis(); }));
  sec(kap.querySelector("button"));
  return kap;
}

function secimYaz(kap, v) {
  kap.querySelectorAll("button").forEach((b) => {
    b.classList.toggle("secili", +b.dataset.deger === v);
    b.setAttribute("aria-checked", +b.dataset.deger === v);
  });
}

function formBolumu(no, baslik) {
  const fs = document.createElement("fieldset");
  fs.className = "form-bolum";
  fs.innerHTML = `<h3><span class="no">${String(no).padStart(2, "0")}</span>${baslik}</h3><div class="alanlar"></div>`;
  return fs;
}

function durumYaz(el, sinif, metin) {
  el.className = `durum ${sinif}`;
  el.textContent = metin;
}

function katMetni(kat) {
  return kat >= 1
    ? `nüfus ortalamasının <b>${kat.toFixed(1).replace(".", ",")} katı</b>`
    : `nüfus ortalamasının <b>%${Math.round(100 * (1 - kat))} altında</b>`;
}

function faktorCiz(kap, faktorler, etiketEk = () => "") {
  const enBuyuk = Math.max(...faktorler.map((f) => Math.abs(f.etki)), 0.01);
  kap.innerHTML = faktorler
    .map((f) => `<div class="faktor" title="Katkı: ${f.etki.toFixed(3)}">
      <div class="ad">${f.etiket}${etiketEk(f)}</div>
      <div class="cubuk"><span class="${f.etki > 0 ? "arti" : "eksi"}" style="width:${(50 * Math.abs(f.etki)) / enBuyuk}%"></span></div>
    </div>`).join("");
}

function sayacAnimasyonu(el, hedef) {
  const bas = +el.dataset.deger || 0, t0 = performance.now();
  el.dataset.deger = hedef;
  const yaz = (v) => (el.textContent = v.toFixed(1).replace(".", ","));
  if (AZ_HAREKET) return yaz(hedef);
  const adim = (t) => {
    const k = Math.min((t - t0) / 400, 1), e = 1 - Math.pow(1 - k, 3);
    yaz(bas + (hedef - bas) * e);
    if (k < 1) requestAnimationFrame(adim);
  };
  requestAnimationFrame(adim);
}

// ---------- Sekmeler ----------
document.querySelectorAll(".sekme").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".sekme").forEach((x) => {
      x.classList.toggle("aktif", x === b);
      x.setAttribute("aria-selected", x === b);
    });
    document.querySelectorAll(".sayfa").forEach((s) => s.classList.toggle("aktif", s.id === b.dataset.sayfa));
    if (b.dataset.sayfa === "lab") raporYukle();
    window.scrollTo(0, 0);
  })
);

// ---------- Risk değerlendirmesi formu ----------
let META = null;

function alanOlustur(o) {
  const kap = document.createElement("div");
  kap.className = "alan";
  kap.dataset.ad = o.ad;
  if (o.ad === "BMI") {
    kap.innerHTML = `<label>Boy (cm) ve kilo (kg)</label>
      <div class="ikili-giris">
        <input type="number" inputmode="numeric" id="boy" min="120" max="230" value="170" aria-label="Boy (cm)">
        <input type="number" inputmode="numeric" id="kilo" min="30" max="250" value="75" aria-label="Kilo (kg)">
      </div>
      <div class="yardim">Vücut kitle indeksi: <span id="bki-deger" class="sayi">-</span></div>`;
    return kap;
  }
  kap.innerHTML = `<label>${o.soru}</label>`;
  if (o.tip === "sayi") {
    kap.insertAdjacentHTML("beforeend", `<div class="kaydir"><input type="range" min="${o.min}" max="${o.max}" value="0" aria-label="${o.etiket}"><output>0</output></div>`);
    const r = $("input", kap);
    r.addEventListener("input", () => ($("output", kap).textContent = r.value));
  } else if (o.secenekler.length === 2) {
    kap.appendChild(secimKutusu(o.secenekler));
  } else {
    kap.insertAdjacentHTML("beforeend", `<select aria-label="${o.etiket}">${o.secenekler.map((s) => `<option value="${s.deger}">${s.etiket}</option>`).join("")}</select>`);
  }
  return kap;
}

function bkiHesapla() {
  const boy = +$("#boy").value / 100, kilo = +$("#kilo").value;
  const b = boy > 0 ? kilo / (boy * boy) : NaN;
  if (!isFinite(b)) return NaN;
  const sinif = b < 18.5 ? "zayıf" : b < 25 ? "normal" : b < 30 ? "fazla kilolu" : "obez";
  $("#bki-deger").textContent = `${b.toFixed(1).replace(".", ",")} (${sinif})`;
  return Math.round(b);
}

function degerAl(ad) {
  if (ad === "BMI") return bkiHesapla();
  const kap = $(`#form .alan[data-ad="${ad}"]`);
  const secili = $("button.secili", kap);
  if (secili) return +secili.dataset.deger;
  return +$("input, select", kap).value;
}

function degerYaz(ad, v) {
  const kap = $(`#form .alan[data-ad="${ad}"]`);
  const secim = $(".secim", kap);
  if (secim) return secimYaz(secim, v);
  const g = $("input, select", kap);
  g.value = v;
  g.dispatchEvent(new Event("input"));
}

async function formKur() {
  META = await (await fetch("/api/meta")).json();
  const harita = Object.fromEntries(META.ozellikler.map((o) => [o.ad, o]));
  const form = $("#form");
  GRUPLAR.forEach((g, i) => {
    const fs = formBolumu(i + 1, g.baslik);
    g.alanlar.forEach((a) => $(".alanlar", fs).appendChild(alanOlustur(harita[a])));
    form.appendChild(fs);
  });
  form.insertAdjacentHTML("beforeend", `<div class="form-alt"><span class="hata" id="hata" role="alert"></span><button class="btn" type="submit">Hesapla</button></div>`);
  $("#boy").addEventListener("input", bkiHesapla);
  $("#kilo").addEventListener("input", bkiHesapla);
  [["Age", 5], ["GenHlth", 2], ["CholCheck", 1]].forEach(([a, v]) => degerYaz(a, v));
  bkiHesapla();
  form.addEventListener("submit", (e) => { e.preventDefault(); tahminEt(); });
}

document.querySelectorAll("[data-ornek]").forEach((b) =>
  b.addEventListener("click", () => {
    const o = ORNEKLER[b.dataset.ornek];
    $("#boy").value = o.boy;
    $("#kilo").value = o.kilo;
    Object.entries(o).forEach(([k, v]) => { if (k !== "boy" && k !== "kilo") degerYaz(k, v); });
    bkiHesapla();
    tahminEt();
  })
);

async function tahminEt() {
  const btn = $("#form .btn"), hata = $("#hata");
  hata.textContent = "";
  const degerler = Object.fromEntries(META.ozellikler.map((o) => [o.ad, degerAl(o.ad)]));
  if (!(degerler.BMI >= 12 && degerler.BMI <= 98)) {
    hata.textContent = "Geçerli bir boy ve kilo girin.";
    return;
  }
  btn.disabled = true;
  try {
    const r = await fetch("/api/tahmin", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ degerler }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Sunucu hatası");
    sonucGoster(j);
  } catch (e) {
    hata.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
}

const SEVIYE_SINIF = { "Düşük": "dusuk", "Orta": "orta", "Yüksek": "yuksek", "Çok yüksek": "yuksek" };

function sonucGoster(s) {
  $("#sonuc-bos").hidden = true;
  $("#sonuc-dolu").hidden = false;
  if (s.olasilik < 0.005) {
    $("#yuzde").dataset.deger = 0;
    $("#yuzde").textContent = "<0,5";
  } else {
    sayacAnimasyonu($("#yuzde"), 100 * s.olasilik);
  }
  durumYaz($("#seviye"), SEVIYE_SINIF[s.seviye], `${s.seviye} risk`);

  const ust = 0.6;
  $("#olcek-dolu").style.width = `${Math.min(s.olasilik / ust, 1) * 100}%`;
  $("#olcek-ort").style.left = `${(s.taban_oran / ust) * 100}%`;
  $("#olcek-ort-etiket").textContent = `ortalama %${yuzde(s.taban_oran, 0)}`;
  $("#kat").innerHTML = `Tahmini olasılık ${katMetni(s.kat)}.`;
  $("#tarama").innerHTML = s.tarama_pozitif
    ? `<div class="kutu poz">Tarama eşiğinin üzerinde: açlık kan şekeri veya HbA1c ölçümü önerilir.</div>`
    : `<div class="kutu neg">Tarama eşiğinin altında: ek test önerilmiyor.</div>`;

  faktorCiz($("#faktorler"), s.faktorler.slice(0, 8), (f) => ` <small>${f.deger}</small>`);

  $("#senaryo-kutu").hidden = !s.senaryolar.length;
  $("#senaryolar").innerHTML = s.senaryolar
    .map((x) => `<tr><td class="metin-sol">${x.metin}</td><td>%${yuzde(x.yeni_olasilik)}</td><td>${yuzde(x.fark)} puan</td></tr>`).join("");
}

// ---------- Model raporu ----------
let raporYuklendi = false;

async function raporYukle() {
  if (raporYuklendi) return;
  raporYuklendi = true;
  const s = await (await fetch("/api/sonuclar")).json();
  const m = s.model, n = s.nhanes;

  const kpi = [];
  if (m) kpi.push([binlik(m.veri.temiz_satir), "BRFSS kişi (temizlenmiş)"], [sayi(m.modeller[0].roc_auc), `BRFSS en iyi AUC · ${m.modeller[0].model.split(" (")[0]}`]);
  if (n) kpi.push([sayi(n.model_auc), "NHANES model AUC"], [`${sayi(n.findrisc_auc)} <small>/ ${sayi(n.ada_auc)}</small>`, "FINDRISC / ADA AUC"], [`${n.model_esit_alarmda_yakalanan} <small>/ ${n.ada_yakalanan}</small>`, `${n.test_hasta} hastadan yakalanan · model / ADA`]);
  $("#kpiler").innerHTML = kpi.map(([d, e]) => `<div class="istatistik"><div class="d">${d}</div><div class="e">${e}</div></div>`).join("");

  if (m) {
    const kol = [["cv_auc", "CV AUC"], ["roc_auc", "Test AUC"], ["pr_auc", "PR-AUC"], ["accuracy", "Doğruluk"], ["recall", "Duyarlılık"], ["precision", "Kesinlik"], ["f1", "F1"]];
    $("#tablo").innerHTML = `<thead><tr><th>Model</th>${kol.map(([, b]) => `<th>${b}</th>`).join("")}<th>Süre</th></tr></thead><tbody>` +
      m.modeller.map((r, i) => `<tr class="${i === 0 ? "vurgulu" : ""}"><td>${r.model}</td>${kol.map(([k]) => `<td>${r[k] == null ? "–" : sayi(r[k])}</td>`).join("")}<td>${sayi(r.sure_sn, 1)} sn</td></tr>`).join("") + "</tbody>";
  }

  const so = m?.sosyoekonomik_ablasyon;
  if (so) {
    $("#sosyo-not").textContent = `Gelir, eğitim, sigorta ve maddi engel soruları çıkarıldı: doğrulama AUC'si 21 soruyla ${sayi(so["21_soru"])}, 17 soruyla ${sayi(so["17_soru"])} (fark ${sayi(so["21_soru"] - so["17_soru"], 4)}).`;
  }

  const g = s.gelistirme;
  if (g) {
    $("#eleme-kural").textContent = `Yalnızca eğitim dönemleri (2007–16), 5-katlı CV. Kabul kuralı: ${g.kural}. Temel model (lojistik regresyon) CV AUC ${sayi(g.temel_auc)}.`;
    $("#eleme-tablo").innerHTML = `<thead><tr><th>Aday</th><th>CV AUC</th><th>Kazanç</th><th>Üstün olduğu kat</th><th>Karar</th></tr></thead><tbody>` +
      Object.entries(g.adaylar).map(([ad, a]) => `<tr><td>${ad}</td><td>${sayi(a.ortalama_auc)}</td><td>${a.kazanc >= 0 ? "+" : ""}${sayi(a.kazanc, 4)}</td><td>${a.kazandigi_kat} / 5</td>
        <td class="${a.karar === "DAHİL" ? "etiket-dahil" : "etiket-elendi"}">${a.karar === "DAHİL" ? "Dahil" : "Elendi"}</td></tr>`).join("") + "</tbody>";
  }

  if (n?.adalet) {
    $("#adalet-tablo").innerHTML = `<thead><tr><th>Boyut</th><th style="text-align:left">Grup</th><th>Kişi</th><th>Hasta</th><th>Model</th><th>FINDRISC</th><th>ADA</th></tr></thead><tbody>` +
      n.adalet.map((r) => `<tr><td>${r.boyut}</td><td style="text-align:left">${r.grup}</td><td>${binlik(r.n)}</td><td>${r.hasta}</td><td><b>${sayi(r["Yapay zekâ"])}</b></td><td>${sayi(r.FINDRISC)}</td><td>${sayi(r.ADA)}</td></tr>`).join("") + "</tbody>";
  }

  const k = s.aciklama?.kumeleme;
  if (k) {
    $("#kumeler").innerHTML = `<thead><tr><th>Profil</th><th>Kişi payı</th><th>Diyabet oranı</th><th>Ort. BMI</th><th>Yüksek tansiyon</th><th>Fiziksel aktif</th></tr></thead><tbody>` +
      k.profiller.map((p) => `<tr><td>${p.ad}</td><td>%${yuzde(p.kisi_orani, 0)}</td><td><b>%${yuzde(p.diyabet_orani)}</b></td><td>${sayi(p.ortalama.BMI, 1)}</td><td>%${yuzde(p.ortalama.HighBP, 0)}</td><td>%${yuzde(p.ortalama.PhysActivity, 0)}</td></tr>`).join("") + "</tbody>";
  } else {
    $("#kume-kart").hidden = true;
  }

  const grafikler = s.grafikler.map((d) => ({ dosya: d, bilgi: GRAFIK_ACIKLAMA[d.replace(".png", "")] || [d, "", "veri"] }));
  const ciz = (f) => {
    $("#galeri").innerHTML = grafikler.filter((x) => f === "hepsi" || x.bilgi[2] === f)
      .map((x) => `<figure class="gorsel" data-src="/figures/${x.dosya}" data-baslik="${x.bilgi[0]}"><img loading="lazy" src="/figures/${x.dosya}" alt="${x.bilgi[0]}"><figcaption><b>${x.bilgi[0]}</b><span>${x.bilgi[1]}</span></figcaption></figure>`).join("");
  };
  $("#filtre").innerHTML = Object.entries(FILTRELER).map(([k, v], i) => `<button class="btn-ikincil ${i ? "" : "secili"}" data-f="${k}">${v}</button>`).join("");
  $("#filtre").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    document.querySelectorAll("#filtre button").forEach((x) => x.classList.toggle("secili", x === b));
    ciz(b.dataset.f);
  });
  ciz("hepsi");
}

$("#galeri").addEventListener("click", (e) => {
  const g = e.target.closest(".gorsel");
  if (!g) return;
  const lb = $("#lightbox");
  $("img", lb).src = g.dataset.src;
  $("p", lb).textContent = g.dataset.baslik;
  lb.hidden = false;
});
$("#lightbox").addEventListener("click", () => ($("#lightbox").hidden = true));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").hidden = true; });

formKur().catch(() => {
  $("#form").innerHTML = `<div class="panel-ic hata">Sunucuya bağlanılamadı. Sunucuyu başlatın: .venv/Scripts/python -m uvicorn backend.api.main:app</div>`;
});

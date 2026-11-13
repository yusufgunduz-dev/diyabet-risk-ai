// "Klinik karşılaştırma" sekmesi: NHANES modeli, FINDRISC ve ADA Risk Testi aynı kişi için.
// app.js'teki ortak yardımcıları ($, yuzde, sayi, secimKutusu, formBolumu, durumYaz, faktorCiz...) kullanır.

const GIZLI_ORNEKLER = {
  dusuk: { erkek: 0, yas: 28, boy: 165, kilo: 58, egitim: 5, aile_diyabet: 0, prediyabet_tani: 0, gebelik_diyabeti: 0, tansiyon_tani: 0, tansiyon_ilac: 0, kolesterol_tani: 0, kalp_hastaligi: 0, inme: 0, genel_saglik: 2, agir_aktivite: 1, orta_aktivite: 1, oturma_dk: 5, uyku_saat: 7, sigara_hic: 0, sigara_simdi: 0, bel: 72, sistolik: 112, diastolik: 70 },
  fark: { erkek: 0, yas: 47, boy: 160, kilo: 70, egitim: 2, aile_diyabet: 0, prediyabet_tani: 1, gebelik_diyabeti: 0, tansiyon_tani: 0, tansiyon_ilac: 0, kolesterol_tani: 1, kalp_hastaligi: 0, inme: 0, genel_saglik: 4, agir_aktivite: 0, orta_aktivite: 1, oturma_dk: 8, uyku_saat: 6, sigara_hic: 0, sigara_simdi: 0, bel: 102, sistolik: 138, diastolik: 84 },
  yuksek: { erkek: 1, yas: 63, boy: 175, kilo: 104, egitim: 3, aile_diyabet: 1, prediyabet_tani: 1, gebelik_diyabeti: 0, tansiyon_tani: 1, tansiyon_ilac: 1, kolesterol_tani: 1, kalp_hastaligi: 0, inme: 0, genel_saglik: 4, agir_aktivite: 0, orta_aktivite: 0, oturma_dk: 9, uyku_saat: 6, sigara_hic: 1, sigara_simdi: 0, bel: 118, sistolik: 146, diastolik: 88 },
};

let GMETA = null;
const gKap = (ad) => $(`#gform .alan[data-ad="${ad}"]`);

function gAlan(s) {
  const kap = document.createElement("div");
  kap.className = "alan";
  kap.dataset.ad = s.ad;
  kap.innerHTML = `<label>${s.soru}${s.istege_bagli ? " <small>(isteğe bağlı)</small>" : ""}</label>`;
  if (s.tip === "sayi") {
    kap.insertAdjacentHTML("beforeend", `<input type="number" inputmode="decimal" min="${s.min}" max="${s.max}" step="any" value="${s.varsayilan ?? ""}" placeholder="${s.min}–${s.max}${s.birim ? " " + s.birim : ""}">`);
  } else if (s.secenekler.length === 2) {
    kap.appendChild(secimKutusu(s.secenekler, gBagimlilik));
  } else {
    kap.insertAdjacentHTML("beforeend", `<select>${s.secenekler.map((o) => `<option value="${o.deger}">${o.etiket}</option>`).join("")}</select>`);
  }
  return kap;
}

function gOku(ad) {
  const kap = gKap(ad);
  if (kap.classList.contains("gizlendi")) return null;
  const b = $("button.secili", kap);
  if (b) return +b.dataset.deger;
  const g = $("input, select", kap);
  return g.value === "" ? null : +g.value;
}

function gYaz(ad, v) {
  const kap = gKap(ad);
  if (!kap) return;
  const secim = $(".secim", kap);
  if (secim) return secimYaz(secim, v);
  $("input, select", kap).value = v ?? "";
}

// Bazı sorular yalnızca belirli cevaplardan sonra anlamlı
function gBagimlilik() {
  if (!gKap("sigara_simdi")) return;
  gKap("gebelik_diyabeti").classList.toggle("gizlendi", gOku("erkek") === 1);
  gKap("tansiyon_ilac").classList.toggle("gizlendi", gOku("tansiyon_tani") !== 1);
  gKap("sigara_simdi").classList.toggle("gizlendi", gOku("sigara_hic") !== 1);
}

function yarisTablosu(r) {
  const g = r.guven_araliklari;
  const ara = (a) => `<small>${sayi(a[0])}–${sayi(a[1])}</small>`;
  $("#yaris-not").textContent = `Test: 2017–20, ${binlik(r.test_n)} kişi, ${r.test_hasta} teşhis edilmemiş diyabet`;
  $("#yaris-tablo").innerHTML = `<thead><tr><th>Yöntem</th><th>Soru</th><th>ROC-AUC</th><th>%95 GA</th><th>Kendi eşiğinde yakalanan</th><th>Yanlış alarm</th><th>Aynı alarmda model</th></tr></thead><tbody>
    <tr class="vurgulu"><td>Model (lojistik regresyon)</td><td>23</td><td>${sayi(r.model_auc)}</td><td>${ara(g.model)}</td><td>${r.model_yakalanan}</td><td>${binlik(r.model_yanlis_alarm)}</td><td>–</td></tr>
    <tr><td>FINDRISC (≥ ${r.findrisc_esik})</td><td>8</td><td>${sayi(r.findrisc_auc)}</td><td>${ara(g.findrisc)}</td><td>${r.findrisc_yakalanan}</td><td>${binlik(r.findrisc_yanlis_alarm)}</td><td>${r.model_findrisc_alarminda_yakalanan}</td></tr>
    <tr><td>ADA Diyabet Risk Testi (≥ 5)</td><td>7</td><td>${sayi(r.ada_auc)}</td><td>${ara(g.ada)}</td><td>${r.ada_yakalanan}</td><td>${binlik(r.ada_yanlis_alarm)}</td><td>${r.model_esit_alarmda_yakalanan}</td></tr>
  </tbody>`;
}

async function gizliKur() {
  GMETA = await (await fetch("/api/gizli/meta")).json();
  yarisTablosu(GMETA.sonuc);

  const form = $("#gform");
  [...new Set(GMETA.sorular.map((s) => s.grup))].forEach((g, i) => {
    const fs = formBolumu(i + 1, g);
    GMETA.sorular.filter((s) => s.grup === g).forEach((s) => $(".alanlar", fs).appendChild(gAlan(s)));
    form.appendChild(fs);
  });
  form.insertAdjacentHTML("beforeend", `<div class="form-alt"><span class="hata" id="ghata" role="alert"></span><button class="btn" type="submit">Karşılaştır</button></div>`);
  gYaz("genel_saglik", 3);
  gYaz("egitim", 3);
  gBagimlilik();
  form.addEventListener("submit", (e) => { e.preventDefault(); gizliTahmin(); });
}

document.querySelectorAll("[data-gornek]").forEach((b) =>
  b.addEventListener("click", () => {
    Object.entries(GIZLI_ORNEKLER[b.dataset.gornek]).forEach(([k, v]) => gYaz(k, v));
    gBagimlilik();
    gizliTahmin();
  })
);

async function gizliTahmin() {
  const hata = $("#ghata"), btn = $("#gform .btn");
  hata.textContent = "";
  const degerler = Object.fromEntries(GMETA.sorular.map((s) => [s.ad, gOku(s.ad)]));
  btn.disabled = true;
  try {
    const r = await fetch("/api/gizli/tahmin", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ degerler }) });
    const j = await r.json();
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : "Geçersiz giriş");
    gizliGoster(j);
  } catch (e) {
    hata.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
}

function puanTablosu(tbody, maddeler) {
  tbody.innerHTML = maddeler
    .map((m) => `<tr class="${m.puan ? "" : "sifir"}"><td class="metin-sol">${m.madde}${m.not ? ` <small>${m.not}</small>` : ""}</td><td>${m.puan ? "+" + m.puan : "0"}</td></tr>`).join("");
}

function gizliGoster(s) {
  $("#gsonuc-bos").hidden = true;
  $("#gsonuc-dolu").hidden = false;
  const karar = (el, uyari) => durumYaz(el, uyari ? "yuksek" : "dusuk", uyari ? "Riskli: tahlil önerilir" : "Eşiğin altında");

  // Gösterilen olasılık Türkiye'ye kalibre edilmiştir (IDF 2024); karar ve sıralama değişmez
  const p = s.model.olasilik_tr;
  if (p < 0.005) {
    $("#ai-yuzde").dataset.deger = 0;
    $("#ai-yuzde").textContent = "<0,5";
  } else {
    sayacAnimasyonu($("#ai-yuzde"), 100 * p);
  }
  karar($("#ai-karar"), s.model.uyari);
  $("#ai-kat").innerHTML = `Teşhis edilmemiş diyabet olasılığı ${katMetni(p / s.model.taban_tr)}. ` +
    `<small>Türkiye'ye kalibre edildi: ortalama %${yuzde(s.model.taban_tr)} (IDF 2024). ABD verisine göre %${yuzde(s.model.olasilik)}.</small>`;
  faktorCiz($("#ai-faktorler"), s.model.faktorler, (f) => (f.eksik ? " <small>girilmedi</small>" : ""));

  $("#fin-puan").textContent = s.findrisc.puan;
  karar($("#fin-karar"), s.findrisc.uyari);
  puanTablosu($("#fin-maddeler"), s.findrisc.maddeler);

  $("#ada-puan").textContent = s.ada.puan;
  karar($("#ada-karar"), s.ada.uyari);
  puanTablosu($("#ada-maddeler"), s.ada.maddeler);
}

// Doğrudan bağlantı: #test=riskli, #gizli=fark, #lab gibi adresler ilgili sekmeyi açar ve örneği çalıştırır
async function adrestenAc() {
  const [sayfa, ornek] = location.hash.slice(1).split("=");
  const sekme = sayfa && $(`.sekme[data-sayfa="${sayfa}"]`);
  if (!sekme) return;
  sekme.click();
  const dugme = ornek && $(`#${sayfa} [data-ornek="${ornek}"], #${sayfa} [data-gornek="${ornek}"]`);
  if (dugme) dugme.click();
}

Promise.all([
  gizliKur().catch(() => ($("#gform").innerHTML = `<div class="panel-ic hata">NHANES modeli yüklenemedi.</div>`)),
  new Promise((r) => { const t = setInterval(() => { if (META) { clearInterval(t); r(); } }, 50); }),
]).then(adrestenAc);

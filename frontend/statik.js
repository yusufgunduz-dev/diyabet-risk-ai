// Statik mod (Netlify): Python API'si yerine tahminleri tarayıcıda yapar.
// backend/api/main.py'deki /api/* uç noktalarının birebir JavaScript karşılığıdır;
// modeller model/*.json dosyalarından okunur (backend/ml/statik_disa_aktar.py üretir).
// Python ile aynı sonucu verdiği tests/test_statik.py ile doğrulanır.

const Statik = (() => {
  const yukle = (ad) => fetch(`model/${ad}.json`).then((r) => r.json());
  const veri = { brfss: yukle("brfss"), nhanes: yukle("nhanes"), sonuclar: yukle("sonuclar") };

  class ApiHatasi extends Error {}

  // ---------- Ortak ----------
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  function izotonik(k, x) {
    // np.interp: uçlarda sınır değerine sabitlenir
    const { x: xs, y: ys } = k;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (xs[m] <= x) lo = m; else hi = m;
    }
    return xs[hi] === xs[lo] ? ys[hi] : ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo]);
  }

  // Python'daki f"{v:g}" karşılığı
  const g = (v) => String(Number(v.toPrecision(6)));

  // ---------- BRFSS: XGBoost ----------
  function dugumler(agac) {
    const harita = new Map();
    (function gez(n) {
      harita.set(n.nodeid, n);
      (n.children || []).forEach(gez);
    })(agac);
    return harita;
  }

  function agacDegeri(n, x) {
    while (n.leaf === undefined) {
      const v = x[n.split];
      const sonraki = v === null || v === undefined || Number.isNaN(v) ? n.missing : v < n.split_condition ? n.yes : n.no;
      n = n.children.find((c) => c.nodeid === sonraki);
    }
    return n.leaf;
  }

  // TreeSHAP (Lundberg ve ark., 2018, path-dependent): shap.TreeExplainer ile aynı değerler
  function treeShap(agac, harita, x, phi) {
    const genislet = (m, pz, po, pi) => {
      const l = m.length;
      m.push({ d: pi, z: pz, o: po, w: l === 0 ? 1 : 0 });
      for (let i = l - 1; i >= 0; i--) {
        m[i + 1].w += (po * m[i].w * (i + 1)) / (l + 1);
        m[i].w = (pz * m[i].w * (l - i)) / (l + 1);
      }
    };
    const geriSar = (m, i) => {
      const l = m.length - 1, o = m[i].o, z = m[i].z;
      let n = m[l].w;
      for (let j = l - 1; j >= 0; j--) {
        if (o !== 0) {
          const t = m[j].w;
          m[j].w = (n * (l + 1)) / ((j + 1) * o);
          n = t - (m[j].w * z * (l - j)) / (l + 1);
        } else {
          m[j].w = (m[j].w * (l + 1)) / (z * (l - j));
        }
      }
      for (let j = i; j < l; j++) Object.assign(m[j], { d: m[j + 1].d, z: m[j + 1].z, o: m[j + 1].o });
      m.pop();
    };
    const geriSarToplam = (m, i) => {
      const l = m.length - 1, o = m[i].o, z = m[i].z;
      let n = m[l].w, toplam = 0;
      for (let j = l - 1; j >= 0; j--) {
        if (o !== 0) {
          const t = (n * (l + 1)) / ((j + 1) * o);
          toplam += t;
          n = m[j].w - (t * z * (l - j)) / (l + 1);
        } else {
          toplam += m[j].w / z / ((l - j) / (l + 1));
        }
      }
      return toplam;
    };
    const ozyineleme = (n, yol, pz, po, pi) => {
      const m = yol.map((e) => ({ ...e }));
      genislet(m, pz, po, pi);
      if (n.leaf !== undefined) {
        for (let i = 1; i < m.length; i++) phi[m[i].d] = (phi[m[i].d] || 0) + geriSarToplam(m, i) * (m[i].o - m[i].z) * n.leaf;
        return;
      }
      const v = x[n.split];
      const sicakId = v === null || v === undefined || Number.isNaN(v) ? n.missing : v < n.split_condition ? n.yes : n.no;
      const sicak = harita.get(sicakId);
      const soguk = harita.get(sicakId === n.yes ? n.no : n.yes);
      let iz = 1, io = 1;
      const k = m.findIndex((e, i) => i > 0 && e.d === n.split);
      if (k > 0) {
        iz = m[k].z;
        io = m[k].o;
        geriSar(m, k);
      }
      ozyineleme(sicak, m, (iz * sicak.cover) / n.cover, io, n.split);
      ozyineleme(soguk, m, (iz * soguk.cover) / n.cover, 0, n.split);
    };
    ozyineleme(agac, [], 1, 1, null);
  }

  function brfssOlasilik(B, x) {
    const marj = B.taban_marj + B.agaclar.reduce((t, a) => t + agacDegeri(a, x), 0);
    return izotonik(B.kalibrator, sigmoid(marj));
  }

  const SENARYOLAR = [
    ["BMI", (v) => Math.min(v, 24.9), "Kilonuzu normal aralığa (BMI < 25) indirirseniz"],
    ["PhysActivity", () => 1, "Düzenli fiziksel aktivite yaparsanız"],
    ["HvyAlcoholConsump", () => 0, "Yoğun alkol tüketimini bırakırsanız"],
    ["Veggies", () => 1, "Her gün sebze yerseniz"],
    ["Fruits", () => 1, "Her gün meyve yerseniz"],
  ];

  const seviye = (p) => (p >= 0.4 ? "Çok yüksek" : p >= 0.25 ? "Yüksek" : p >= 0.1 ? "Orta" : "Düşük");

  function brfssTahmin(B, degerler) {
    const meta = Object.fromEntries(B.meta.map((o) => [o.ad, o]));
    const eksik = B.ozellikler.filter((a) => !(a in degerler));
    if (eksik.length) throw new ApiHatasi(`Eksik alanlar: ${eksik.join(", ")}`);
    for (const a of B.ozellikler) {
      const o = meta[a], v = degerler[a];
      if (o.tip === "sayi" && !(o.min <= v && v <= o.max)) throw new ApiHatasi(`${o.etiket} ${o.min}-${o.max} arasında olmalı`);
      if (o.tip === "secim" && !o.secenekler.some((s) => s.deger === v)) throw new ApiHatasi(`${o.etiket} için geçersiz değer: ${v}`);
    }
    const x = Object.fromEntries(B.ozellikler.map((a) => [a, degerler[a]]));
    const p = brfssOlasilik(B, x);

    if (!B.haritalar) B.haritalar = B.agaclar.map(dugumler);
    const phi = {};
    B.agaclar.forEach((a, i) => treeShap(a, B.haritalar[i], x, phi));
    const metni = (a, v) => (meta[a].tip === "sayi" ? g(v) : meta[a].secenekler.find((s) => s.deger === v).etiket);
    const faktorler = B.ozellikler
      .map((a) => ({ ozellik: a, etiket: meta[a].etiket, deger: metni(a, x[a]), etki: phi[a] || 0 }))
      .sort((f1, f2) => Math.abs(f2.etki) - Math.abs(f1.etki));

    const senaryolar = [];
    for (const [a, hedef, metin] of SENARYOLAR) {
      const yeni = hedef(x[a]);
      if (yeni === x[a]) continue;
      const p2 = brfssOlasilik(B, { ...x, [a]: yeni });
      if (p2 < p - 0.005) senaryolar.push({ metin, yeni_olasilik: p2, fark: p2 - p });
    }
    senaryolar.sort((s1, s2) => s1.fark - s2.fark);

    return {
      olasilik: p, seviye: seviye(p), tarama_pozitif: p >= B.esik, esik: B.esik,
      taban_oran: B.taban_oran, kat: p / B.taban_oran, faktorler, senaryolar,
    };
  }

  // ---------- NHANES: lojistik regresyon + klinik testler ----------
  function standartlastir(b, x) {
    return x.map((v, i) => ((v === null || v === undefined || Number.isNaN(v) ? b.medyan[i] : v) - b.ortalama[i]) / b.olcek[i]);
  }

  const karar = (b, x) => standartlastir(b, x).reduce((t, z, i) => t + z * b.katsayi[i], b.sabit);

  function turkiyeyeKaydir(p, kaynak, hedef) {
    const k = (hedef / (1 - hedef)) / (kaynak / (1 - kaynak));
    return (p * k) / (p * k + (1 - p));
  }

  function adaDetay(d) {
    const { yas, bmi } = d;
    return [
      ["Yaş", yas < 40 ? 0 : yas < 50 ? 1 : yas < 60 ? 2 : 3, `${g(yas)} yaş`],
      ["Erkek olmak", d.erkek === 1 ? 1 : 0, d.erkek === 1 ? "Erkek" : "Kadın"],
      ["Gebelik diyabeti", d.erkek === 0 && d.gebelik_diyabeti === 1 ? 1 : 0, ""],
      ["Ailede diyabet", d.aile_diyabet === 1 ? 1 : 0, ""],
      ["Yüksek tansiyon", d.tansiyon_tani === 1 || d.tansiyon_ilac === 1 ? 1 : 0, ""],
      ["Fiziksel olarak aktif değil", d.agir_aktivite !== 1 && d.orta_aktivite !== 1 ? 1 : 0, ""],
      ["Vücut kitle indeksi", bmi < 25 ? 0 : bmi < 30 ? 1 : bmi < 40 ? 2 : 3, `BMI ${bmi.toFixed(1)}`],
    ].map(([madde, puan, not]) => ({ madde, puan, not }));
  }

  function findriscDetay(d) {
    const { yas, bmi, bel } = d;
    let belPuan = 0, belNot = "ölçülmedi";
    if (bel !== null && bel !== undefined) {
      const [a, b] = d.erkek === 1 ? [94, 102] : [80, 88];
      belPuan = bel < a ? 0 : bel <= b ? 3 : 4;
      belNot = `${g(bel)} cm`;
    }
    const aktif = d.agir_aktivite === 1 || d.orta_aktivite === 1;
    return [
      ["Yaş", yas < 45 ? 0 : yas < 55 ? 2 : yas < 65 ? 3 : 4, `${g(yas)} yaş`],
      ["Vücut kitle indeksi", bmi < 25 ? 0 : bmi <= 30 ? 1 : 3, `BMI ${bmi.toFixed(1)}`],
      ["Bel çevresi", belPuan, belNot],
      ["Fiziksel olarak aktif değil", aktif ? 0 : 2, ""],
      ["Tansiyon ilacı", d.tansiyon_ilac === 1 ? 2 : 0, ""],
      ["Daha önce şekeri yüksek bulunmuş", d.prediyabet_tani === 1 ? 5 : 0, ""],
      ["Ailede diyabet", d.aile_diyabet === 1 ? 5 : 0, ""],
    ].map(([madde, puan, not]) => ({ madde, puan, not }));
  }

  function nhanesTahmin(N, girdi) {
    const d = { ...girdi };
    for (const s of N.sorular) {
      const v = d[s.ad];
      if (v === null || v === undefined) {
        if (s.istege_bagli || s.sadece_kadin || s.kosullu) { d[s.ad] = null; continue; }
        throw new ApiHatasi(`Eksik alan: ${s.soru}`);
      }
      if (s.tip === "sayi" && !(s.min <= v && v <= s.max)) throw new ApiHatasi(`${s.soru}: ${s.min}-${s.max} arasında olmalı`);
      if (s.carpan) d[s.ad] = v * s.carpan;
    }
    if (d.erkek === 1) d.gebelik_diyabeti = 0;
    if (d.tansiyon_tani !== 1) d.tansiyon_ilac = 0;
    if (d.sigara_hic !== 1) d.sigara_simdi = 0;
    d.bmi = d.kilo / (d.boy / 100) ** 2;
    d.bel_boy_orani = d.bel ? d.bel / d.boy : null;

    const x = N.ozellikler.map((o) => (d[o] === undefined ? null : d[o]));
    // CalibratedClassifierCV: her katta izotonik(karar fonksiyonu), katların ortalaması
    const p = N.katlar.reduce((t, k) => t + Math.min(1, Math.max(0, izotonik(k.kalibrator, karar(k.boru, x)))), 0) / N.katlar.length;

    const z = standartlastir(N.ham, x);
    const faktorler = N.ozellikler
      .map((o, i) => ({ ozellik: o, etiket: N.etiketler[o], etki: N.ham.katsayi[i] * z[i], eksik: d[o] === null || d[o] === undefined }))
      .sort((f1, f2) => Math.abs(f2.etki) - Math.abs(f1.etki));

    const S = N.sonuc;
    const ada = adaDetay(d), fin = findriscDetay(d);
    const adaPuan = ada.reduce((t, m) => t + m.puan, 0), finPuan = fin.reduce((t, m) => t + m.puan, 0);
    return {
      model: {
        olasilik: p, uyari: p >= N.esik, esik: N.esik, kat: p / S.taban_oran, faktorler: faktorler.slice(0, 8),
        olasilik_tr: turkiyeyeKaydir(p, S.taban_oran, S.turkiye.gizli_oran), taban_tr: S.turkiye.gizli_oran,
      },
      ada: { puan: adaPuan, uyari: adaPuan >= 5, maddeler: ada },
      findrisc: { puan: finPuan, uyari: finPuan >= S.findrisc_esik, maddeler: fin },
      bmi: d.bmi,
      taban_oran: S.taban_oran,
    };
  }

  // ---------- /api/* uç noktaları ----------
  async function cevapla(yol, govde) {
    const B = await veri.brfss, N = await veri.nhanes;
    switch (yol) {
      case "/api/saglik": return { durum: "ok", model: "XGBoost + izotonik kalibrasyon (tarayıcıda)", esik: B.esik, taban_oran: B.taban_oran };
      case "/api/meta": return { ozellikler: B.meta, esik: B.esik, taban_oran: B.taban_oran };
      case "/api/sonuclar": return veri.sonuclar;
      case "/api/tahmin": return brfssTahmin(B, govde.degerler);
      case "/api/gizli/meta": return { sorular: N.sorular, sonuc: N.sonuc };
      case "/api/gizli/tahmin": return nhanesTahmin(N, govde.degerler);
    }
    throw new ApiHatasi(`Bilinmeyen uç nokta: ${yol}`);
  }

  const asilFetch = window.fetch.bind(window);
  window.fetch = async (adres, ayar = {}) => {
    const yol = new URL(adres, location.href).pathname;
    if (!yol.startsWith("/api/")) return asilFetch(adres, ayar);
    try {
      const sonuc = await cevapla(yol, ayar.body ? JSON.parse(ayar.body) : null);
      return new Response(JSON.stringify(sonuc), { status: 200, headers: { "Content-Type": "application/json" } });
    } catch (e) {
      const kod = e instanceof ApiHatasi ? 422 : 500;
      return new Response(JSON.stringify({ detail: e.message }), { status: kod, headers: { "Content-Type": "application/json" } });
    }
  };

  return { brfssTahmin, nhanesTahmin, veri };
})();

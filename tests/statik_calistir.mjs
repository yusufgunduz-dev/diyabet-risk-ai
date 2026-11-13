// tests/test_statik.py tarafından çağrılır: statik.js'i Node'da çalıştırıp verilen girdilerin sonuçlarını yazar.
// Kullanım: node tests/statik_calistir.mjs <site klasörü> <girdiler.json>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import vm from "node:vm";

const [site, girdiDosyasi] = process.argv.slice(2);
const girdiler = JSON.parse(readFileSync(girdiDosyasi, "utf8"));

const baglam = {
  URL, Response, console, JSON, Math, Number, Object, Map, Promise, Error,
  location: { href: "http://yerel/" },
  fetch: async (adres) => ({ json: async () => JSON.parse(readFileSync(join(site, adres), "utf8")) }),
};
baglam.window = baglam;
vm.createContext(baglam);
vm.runInContext(readFileSync(join(site, "statik.js"), "utf8") + "\nglobalThis.Statik = Statik;", baglam);

const S = baglam.Statik;
const B = await S.veri.brfss, N = await S.veri.nhanes;
const sonuc = {
  brfss: girdiler.brfss.map((d) => S.brfssTahmin(B, d)),
  nhanes: girdiler.nhanes.map((d) => S.nhanesTahmin(N, d)),
};
process.stdout.write(JSON.stringify(sonuc));

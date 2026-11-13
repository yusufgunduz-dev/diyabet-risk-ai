# Security Policy / Güvenlik Politikası

## English

### Supported versions
Only the latest commit on `main` receives fixes.

### Reporting a vulnerability
Please **do not open a public issue** for security problems. Email **gunduzy718@gmail.com** with the subject
`[SECURITY] diyabet-risk-ai`, and include the affected file or feature, steps to reproduce and the impact.
You can expect an acknowledgement within 7 days.

This project contains no credentials. It downloads public CDC datasets at training time and does not store personal data. The risk model is for education and research, not for clinical use.

> The `.joblib` files in `backend/models/` are Python pickles. Loading a pickle from an untrusted source can execute code, so only load models from this repository or retrain them with the provided scripts.

## Türkçe

### Desteklenen sürümler
Yalnızca `main` dalındaki son sürüm düzeltme alır.

### Güvenlik açığı bildirimi
Güvenlik sorunları için **herkese açık issue açmayın**. **gunduzy718@gmail.com** adresine `[SECURITY] diyabet-risk-ai`
konusuyla e-posta gönderin; etkilenen dosya veya özelliği, yeniden üretme adımlarını ve etkisini yazın.
7 gün içinde yanıt vermeyi hedefliyoruz.

Bu proje kimlik bilgisi içermez. Eğitim sırasında herkese açık CDC veri setlerini indirir ve kişisel veri saklamaz. Risk modeli eğitim ve araştırma amaçlıdır, klinik kullanım için değildir.

> `backend/models/` içindeki `.joblib` dosyaları Python pickle dosyalarıdır. Güvenilmeyen kaynaktan yüklenen pickle kod çalıştırabilir; bu yüzden yalnızca bu depodaki modelleri yükleyin veya verilen betiklerle yeniden eğitin.

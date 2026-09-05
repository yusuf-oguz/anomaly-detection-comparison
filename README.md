# Comparative Study of Traditional and Backpropagation-Free Deep Learning Methods for Anomaly Detection

İTÜ "Learning From Data" dönem projesi. **Hipotez:** Geleneksel yöntemler (Isolation Forest, One-Class SVM) hızlı ama yeterince doğru değil; standart derin öğrenme (Autoencoder) doğru ama backpropagation nedeniyle yavaş/hesaplama maliyeti yüksek. **Backpropagation-free** derin öğrenme yöntemleri (PatchCore, PaDiM) hem yüksek doğruluk hem düşük hesaplama maliyeti sağlayabilir mi?

## Sonuç

Hipotez doğrulandı: **PatchCore ve PaDiM, standart bir Autoencoder'a kıyasla çok daha az hesaplama süresiyle %98'in üzerinde AUROC** elde etti — geleneksel yöntemleri hem hız hem doğrulukta geride bıraktı.

## Karşılaştırılan 5 Yöntem

| Yöntem | Tip |
|---|---|
| Isolation Forest | Geleneksel ML |
| One-Class SVM | Geleneksel ML |
| Autoencoder | Derin öğrenme (backpropagation ile) |
| PaDiM | Derin öğrenme (backpropagation-free, önceden eğitilmiş özellik çıkarıcı + istatistiksel modelleme) |
| PatchCore | Derin öğrenme (backpropagation-free, bellek bankası / memory bank tabanlı) |

**Veri seti:** MVTec AD (bottle, carpet, hazelnut kategorileri) — endüstriyel ürün görüntülerinde kusur tespiti için standart akademik benchmark.

## Klasör Yapısı

```
LFD_PROJECT_FULL/
├── ALL CODES/           # 5 yöntemin her biri için ayrı notebook
├── ALL RESULTS/         # Yöntem bazlı sonuçlar + karşılaştırmalı tablo/grafikler
│   └── TABLES-CHARTS/     # Genel karşılaştırma tabloları ve bar chart'lar
├── LFD_Project/
│   └── anomalib_patchcore/  # PatchCore'un eğitilmiş model ağırlıkları (birden fazla deneme versiyonu)
├── LFD_FINAL_REPORT.pdf   # Yazılı final rapor
└── LFD-PRESENTATION.pdf/.pptx/.mp4  # Sunum + kayıt
```

## Veri Seti Notu

`LFD_Project/dataset/` bilinçli olarak bu depoya dahil edilmedi — [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), halka açık, ücretsiz indirilebilir standart bir akademik veri seti (kendi üretilen veri değil). Kod, veri setinin `bottle/`, `carpet/`, `hazelnut/` alt klasörlerinin proje kökünde `LFD_Project/dataset/` altında olmasını bekliyor.

## Kullanılan Araçlar

Python — [anomalib](https://github.com/openvinotoolkit/anomalib) (PatchCore/PaDiM), scikit-learn (Isolation Forest, One-Class SVM), PyTorch (Autoencoder).

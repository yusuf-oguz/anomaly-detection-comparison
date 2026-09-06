# Comparative Study of Traditional and Backpropagation-Free Deep Learning Methods for Anomaly Detection

<details>
<summary>🇹🇷 Türkçe özet için tıklayın</summary>

**Hipotez:** Geleneksel yöntemler (Isolation Forest, One-Class SVM) hızlı ama yeterince doğru değil; standart derin öğrenme (Autoencoder) doğru ama backpropagation nedeniyle yavaş/hesaplama maliyeti yüksek. **Backpropagation-free** derin öğrenme yöntemleri (PatchCore, PaDiM) hem yüksek doğruluk hem düşük hesaplama maliyeti sağlayabilir mi?

**Sonuç:** Hipotez doğrulandı. **PatchCore ve PaDiM, standart bir Autoencoder'a kıyasla çok daha az hesaplama süresiyle %98'in üzerinde AUROC elde etti** ve geleneksel yöntemleri hem hız hem doğrulukta geride bıraktı.

**Karşılaştırılan 5 yöntem:** Isolation Forest ve One-Class SVM (geleneksel ML), Autoencoder (backpropagation ile derin öğrenme), PaDiM ve PatchCore (backpropagation-free derin öğrenme). Veri seti: MVTec AD (bottle, carpet, hazelnut kategorileri), endüstriyel kusur tespiti için standart akademik benchmark.

Klasör yapısı, veri seti notu ve araçlar için aşağıdaki İngilizce bölümlere bakılabilir (tablo/kod/isimler zaten dil bağımsız).

</details>

**Hypothesis:** traditional methods (Isolation Forest, One-Class SVM) are fast but not accurate enough; standard deep learning (Autoencoder) is accurate but slow/computationally expensive due to backpropagation. Can **backpropagation-free** deep learning methods (PatchCore, PaDiM) deliver both high accuracy and low computational cost?

## Result

Hypothesis confirmed. **PatchCore and PaDiM achieve over 98% AUROC with far less compute than a standard Autoencoder**, outperforming the traditional methods on both speed and accuracy.

## 5 Methods Compared

| Method | Type |
|---|---|
| Isolation Forest | Traditional ML |
| One-Class SVM | Traditional ML |
| Autoencoder | Deep learning (with backpropagation) |
| PaDiM | Deep learning, backpropagation-free (pretrained feature extractor + statistical modeling) |
| PatchCore | Deep learning, backpropagation-free (memory-bank based) |

**Dataset:** MVTec AD (bottle, carpet, hazelnut categories), the standard academic benchmark for industrial defect detection.

## Folder Structure

```
.
├── ALL CODES/           # separate notebook per method
├── ALL RESULTS/         # per-method results + comparison tables/charts
│   └── TABLES-CHARTS/     # overall comparison tables and bar charts
├── LFD_Project/
│   └── anomalib_patchcore/  # PatchCore's trained model weights (multiple runs)
├── LFD_FINAL_REPORT.pdf   # written final report
└── LFD-PRESENTATION.pdf/.pptx/.mp4  # presentation + recording
```

## Dataset Note

`LFD_Project/dataset/` is deliberately excluded from this repo. It's [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), a public, freely downloadable standard academic dataset, not self-generated data. The code expects the dataset's `bottle/`, `carpet/`, `hazelnut/` subfolders under `LFD_Project/dataset/` at the project root.

## Tools

Python, [anomalib](https://github.com/openvinotoolkit/anomalib) for PatchCore/PaDiM, scikit-learn for Isolation Forest and One-Class SVM, PyTorch for the Autoencoder.

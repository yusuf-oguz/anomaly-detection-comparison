# Comparative Study of Traditional and Backpropagation-Free Deep Learning Methods for Anomaly Detection

<details>
<summary>🇹🇷 Türkçe özet için tıklayın</summary>

**Hipotez:** Geleneksel yöntemler (Isolation Forest, One-Class SVM) hızlı ama yeterince doğru değil; standart derin öğrenme (Autoencoder) doğru ama backpropagation nedeniyle yavaş/hesaplama maliyeti yüksek. **Backpropagation-free** derin öğrenme yöntemleri (PatchCore, PaDiM) hem yüksek doğruluk hem düşük hesaplama maliyeti sağlayabilir mi?

**Sonuç:** Hipotez doğrulandı. **PatchCore ve PaDiM, standart bir Autoencoder'a kıyasla çok daha az hesaplama süresiyle %98'in üzerinde AUROC elde etti** ve geleneksel yöntemleri hem hız hem doğrulukta geride bıraktı.

**Karşılaştırılan 5 yöntem:** Isolation Forest ve One-Class SVM (geleneksel ML), Autoencoder (backpropagation ile derin öğrenme), PaDiM ve PatchCore (backpropagation-free derin öğrenme). Veri seti: MVTec AD (bottle, carpet, hazelnut kategorileri), endüstriyel kusur tespiti için standart akademik benchmark.

Detaylı sonuç tablosu (5 yöntem × 3 kategori, AUROC ve süre), grafikler, klasör yapısı, veri seti notu ve araçlar için aşağıdaki İngilizce bölümlere bakılabilir (tablo/kod/isimler zaten dil bağımsız).

</details>

---

**Hypothesis:** traditional methods (Isolation Forest, One-Class SVM) are fast but not accurate enough; standard deep learning (Autoencoder) is accurate but slow/computationally expensive due to backpropagation. Can **backpropagation-free** deep learning methods (PatchCore, PaDiM) deliver both high accuracy and low computational cost?

## Results

Hypothesis confirmed. **PatchCore and PaDiM achieve over 98% AUROC with far less compute than a standard Autoencoder**, outperforming the traditional methods on both speed and accuracy.

| Category | Method | AUROC | Total Time |
|---|---|---:|---:|
| Carpet | Isolation Forest | 0.46 | 2 min |
| Carpet | One-Class SVM | 0.37 | 4 min |
| Carpet | Autoencoder | 0.31 | 22 min |
| Carpet | PatchCore | 0.98 | 10 min |
| Carpet | PaDiM | 1.00 | 4 min |
| Hazelnut | Isolation Forest | 0.09 | 2 min |
| Hazelnut | One-Class SVM | 0.65 | 3 min |
| Hazelnut | Autoencoder | 0.89 | 7 min |
| Hazelnut | PaDiM | 0.99 | 2 min |
| Hazelnut | PatchCore | 1.00 | 12 min |
| Bottle | Isolation Forest | 0.69 | 1 min |
| Bottle | One-Class SVM | 0.95 | 2 min |
| Bottle | Autoencoder | 0.53 | 10 min |
| Bottle | PaDiM | 1.00 | 2 min |
| Bottle | PatchCore | 1.00 | 4 min |

![Performance and computation time by method](<ALL RESULTS/TABLES-CHARTS/methods_bar_chart.png>)
![Mean performance and time by category](<ALL RESULTS/TABLES-CHARTS/categories_bar_chart.png>)

**Key findings:**

- **PaDiM was the strongest method overall:** a perfect 1.00 average AUROC with the lowest average preparation time (about 3 minutes) among the two backpropagation-free methods.
- **PatchCore was a close second on accuracy** (0.99 average AUROC), but took noticeably longer on the harder categories (12 minutes on hazelnut, versus PaDiM's 2).
- **Traditional methods were fast but unreliable on complex textures.** Isolation Forest scored 0.09 AUROC on hazelnut, worse than random guessing (0.50).
- **The standard Autoencoder, included as the expected accurate-but-slow reference point, turned out to be the least efficient method in the study.** It had both the highest average preparation time (13 minutes) and a lower average AUROC (0.57) than even the traditional One-Class SVM (0.65).

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
├── model_checkpoints/
│   └── anomalib_patchcore/  # PatchCore's trained model weights (multiple runs)
├── final_report.pdf     # written final report
└── presentation.pptx/.mp4  # presentation + recording
```

## Dataset Note

`model_checkpoints/dataset/` is deliberately excluded from this repo. It's [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), a public, freely downloadable standard academic dataset, not self-generated data. The notebooks expect the dataset's `bottle/`, `carpet/`, `hazelnut/` subfolders under that path at the project root (they were originally run in Google Colab against a Google Drive path, so re-running them locally would need the data paths updated first).

## Tools

Python, [anomalib](https://github.com/openvinotoolkit/anomalib) for PatchCore/PaDiM, scikit-learn for Isolation Forest and One-Class SVM, PyTorch for the Autoencoder.

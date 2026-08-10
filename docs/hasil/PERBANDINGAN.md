# Perbandingan Arsitektur — split `official`, augmentasi `codec`

Test set: **1088** berkas. 1 berkas = 0.092 pp.

Ambang: prior-matched (transduktif, tanpa label test).

## 1. Rerata ± simpangan baku atas seed

| model | n seed | akurasi | EER | AUC | seed individual |
|---|---|---|---|---|---|
| `ast` | 3 | **86.43%** ±2.94 | 13.42% | 0.9418 | 89.34, 86.49, 83.46 |
| `cnn_asp` | 3 | **91.94%** ±3.50 | 8.06% | 0.9712 | 95.04, 92.65, 88.14 |
| `cnnlstm` | 3 | **83.52%** ±2.28 | 16.48% | 0.8976 | 84.93, 84.74, 80.88 |
| `hubert` | 8 | **97.29%** ±2.16 | 2.71% | 0.9963 | 99.45, 99.08, 98.90, 97.79, 97.61, 97.52, 94.03, 93.93 |
| `wav2vec2` | 3 | **90.75%** ±0.51 | 9.34% | 0.9689 | 91.08, 90.99, 90.17 |
| `wavlm` | 3 | **96.51%** ±2.41 | 3.55% | 0.9941 | 98.71, 96.88, 93.93 |

Selisih peringkat 1 (`hubert`) dan 2 (`wavlm`): **0.78 pp**, sedangkan std gabungan **±2.29 pp**.
→ **Selisih berada DI DALAM derau — peringkat tidak dapat dipertahankan.**

## 2. Uji McNemar berpasangan (seed terbaik per model)

Test set identik untuk semua model → data berpasangan. n01 = A benar & B salah; n10 = A salah & B benar.

| perbandingan | n01 | n10 | p mentah | p terkoreksi | signifikan? |
|---|---|---|---|---|---|
| ast vs hubert | 4 | 114 | 0 | 0 | ✅ ya |
| ast vs wavlm | 11 | 113 | 0 | 0 | ✅ ya |
| cnn_asp vs cnnlstm | 137 | 27 | 0 | 0 | ✅ ya |
| cnnlstm vs hubert | 4 | 162 | 0 | 0 | ✅ ya |
| cnnlstm vs wavlm | 10 | 160 | 0 | 0 | ✅ ya |
| hubert vs wav2vec2 | 96 | 5 | 0 | 0 | ✅ ya |
| wav2vec2 vs wavlm | 4 | 87 | 0 | 0 | ✅ ya |
| cnn_asp vs hubert | 5 | 53 | 6.769e-10 | 5.415e-09 | ✅ ya |
| ast vs cnn_asp | 44 | 106 | 6.338e-07 | 4.436e-06 | ✅ ya |
| cnn_asp vs wavlm | 11 | 51 | 7.308e-07 | 4.436e-06 | ✅ ya |
| cnnlstm vs wav2vec2 | 79 | 146 | 1.083e-05 | 5.413e-05 | ✅ ya |
| cnn_asp vs wav2vec2 | 91 | 48 | 0.0003675 | 0.00147 | ✅ ya |
| ast vs cnnlstm | 130 | 82 | 0.001247 | 0.00374 | ✅ ya |
| hubert vs wavlm | 13 | 5 | 0.09625 | 0.1925 | ❌ tidak |
| ast vs wav2vec2 | 88 | 107 | 0.1974 | 0.1974 | ❌ tidak |

Koreksi Holm-Bonferroni, α = 0,05.

## 3. Korelasi error antar model

φ rendah (< 0,5) → model gagal pada berkas berbeda → ensembling berguna.

| pasangan | φ | Jaccard error |
|---|---|---|
| ast vs cnn_asp | 0.058 | 0.062 |
| ast vs cnnlstm | 0.137 | 0.138 |
| ast vs hubert | 0.055 | 0.017 |
| ast vs wav2vec2 | -0.014 | 0.044 |
| ast vs wavlm | 0.040 | 0.024 |
| cnn_asp vs cnnlstm | 0.223 | 0.141 |
| cnn_asp vs hubert | 0.040 | 0.017 |
| cnn_asp vs wav2vec2 | 0.018 | 0.041 |
| cnn_asp vs wavlm | 0.087 | 0.046 |
| cnnlstm vs hubert | 0.038 | 0.012 |
| cnnlstm vs wav2vec2 | 0.030 | 0.074 |
| cnnlstm vs wavlm | 0.043 | 0.023 |
| hubert vs wav2vec2 | 0.020 | 0.010 |
| hubert vs wavlm | 0.102 | 0.053 |
| wav2vec2 vs wavlm | 0.250 | 0.099 |

## 4. Ensemble (rata-rata skor)

| ensemble | anggota | akurasi | EER | AUC |
|---|---|---|---|---|
| `ast` (semua seed) | 3 | **89.71%** | 10.29% | 0.9651 |
| `cnn_asp` (semua seed) | 3 | **95.40%** | 4.87% | 0.9877 |
| `cnnlstm` (semua seed) | 3 | **86.03%** | 13.97% | 0.9304 |
| `hubert` (semua seed) | 8 | **99.82%** | 0.18% | 0.9999 |
| `wav2vec2` (semua seed) | 3 | **93.01%** | 6.99% | 0.9832 |
| `wavlm` (semua seed) | 3 | **97.61%** | 2.11% | 0.9974 |
| **semua model** (seed terbaik) | 6 | **98.53%** | 1.65% | 0.9987 |
| **semua run** | 23 | **98.71%** | 1.19% | 0.9989 |

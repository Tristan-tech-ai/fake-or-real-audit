# Pencarian Ensemble Terbaik

Seluruh run split resmi + augmentasi codec, ambang prior-matched, test set 1088 berkas.

## 1. Per arsitektur (ensemble antar-seed)

| arsitektur | n run | akurasi | EER | AUC | salah |
|---|---|---|---|---|---|
| `ast` | 3 | **89.71%** | 10.29% | 0.9651 | 112/1088 |
| `cnn_asp` | 6 | **95.96%** | 3.95% | 0.9877 | 44/1088 |
| `cnnlstm` | 3 | **86.03%** | 13.97% | 0.9304 | 152/1088 |
| `hubert` | 8 | **99.82%** | 0.18% | 0.9999 | 2/1088 |
| `wav2vec2` | 3 | **93.01%** | 6.99% | 0.9832 | 76/1088 |
| `wavlm` | 3 | **97.61%** | 2.11% | 0.9974 | 26/1088 |

## 2. Korelasi error antar arsitektur (pada ensemble per-arsitektur)

| pasangan | φ | error bersama | hanya A | hanya B |
|---|---|---|---|---|
| ast vs cnn_asp | 0.038 | 7 | 105 | 37 |
| ast vs cnnlstm | 0.169 | 35 | 77 | 117 |
| ast vs hubert | 0.056 | 1 | 111 | 1 |
| ast vs wav2vec2 | 0.026 | 10 | 102 | 66 |
| ast vs wavlm | -0.033 | 1 | 111 | 25 |
| cnn_asp vs cnnlstm | 0.240 | 24 | 20 | 128 |
| cnn_asp vs hubert | 0.100 | 1 | 43 | 1 |
| cnn_asp vs wav2vec2 | 0.035 | 5 | 39 | 71 |
| cnn_asp vs wavlm | 0.090 | 4 | 40 | 22 |
| cnnlstm vs hubert | 0.045 | 1 | 151 | 1 |
| cnnlstm vs wav2vec2 | 0.098 | 20 | 132 | 56 |
| cnnlstm vs wavlm | -0.011 | 3 | 149 | 23 |
| hubert vs wav2vec2 | 0.072 | 1 | 1 | 75 |
| hubert vs wavlm | -0.007 | 0 | 2 | 26 |
| wav2vec2 vs wavlm | 0.052 | 4 | 72 | 22 |

## 3. Seluruh kombinasi arsitektur

| kombinasi | akurasi | EER | AUC | salah |
|---|---|---|---|---|
| ast + cnn_asp | **97.43%** | 2.57% | 0.9945 | 28/1088 |
| ast + cnnlstm | **91.18%** | 9.01% | 0.9722 | 96/1088 |
| ast + hubert | **98.35%** | 1.65% | 0.9986 | 18/1088 |
| ast + wav2vec2 | **96.51%** | 3.40% | 0.9944 | 38/1088 |
| ast + wavlm | **95.96%** | 3.95% | 0.9935 | 44/1088 |
| cnn_asp + cnnlstm | **95.40%** | 4.60% | 0.9832 | 50/1088 |
| cnn_asp + hubert | **98.16%** | 1.84% | 0.9979 | 20/1088 |
| cnn_asp + wav2vec2 | **97.24%** | 2.67% | 0.9956 | 30/1088 |
| cnn_asp + wavlm | **97.24%** | 2.67% | 0.9949 | 30/1088 |
| cnnlstm + hubert | **96.14%** | 3.86% | 0.9930 | 42/1088 |
| cnnlstm + wav2vec2 | **92.65%** | 7.35% | 0.9797 | 80/1088 |
| cnnlstm + wavlm | **93.01%** | 6.99% | 0.9789 | 76/1088 |
| hubert + wav2vec2 | **99.26%** | 0.74% | 0.9998 | 8/1088 |
| hubert + wavlm | **99.82%** | 0.18% | 1.0000 | 2/1088 |
| wav2vec2 + wavlm | **96.69%** | 3.22% | 0.9956 | 36/1088 |
| ast + cnn_asp + cnnlstm | **97.06%** | 2.85% | 0.9901 | 32/1088 |
| ast + cnn_asp + hubert | **98.35%** | 1.65% | 0.9990 | 18/1088 |
| ast + cnn_asp + wav2vec2 | **98.16%** | 1.75% | 0.9980 | 20/1088 |
| ast + cnn_asp + wavlm | **98.16%** | 1.93% | 0.9977 | 20/1088 |
| ast + cnnlstm + hubert | **97.43%** | 2.67% | 0.9960 | 28/1088 |
| ast + cnnlstm + wav2vec2 | **95.40%** | 4.69% | 0.9920 | 50/1088 |
| ast + cnnlstm + wavlm | **95.40%** | 4.60% | 0.9900 | 50/1088 |
| ast + hubert + wav2vec2 | **98.53%** | 1.38% | 0.9989 | 16/1088 |
| ast + hubert + wavlm | **98.71%** | 1.29% | 0.9990 | 14/1088 |
| ast + wav2vec2 + wavlm | **97.06%** | 2.94% | 0.9964 | 32/1088 |
| cnn_asp + cnnlstm + hubert | **97.61%** | 2.39% | 0.9947 | 26/1088 |
| cnn_asp + cnnlstm + wav2vec2 | **96.69%** | 3.40% | 0.9930 | 36/1088 |
| cnn_asp + cnnlstm + wavlm | **96.88%** | 3.12% | 0.9903 | 34/1088 |
| cnn_asp + hubert + wav2vec2 | **98.35%** | 1.65% | 0.9984 | 18/1088 |
| cnn_asp + hubert + wavlm | **98.35%** | 1.75% | 0.9984 | 18/1088 |
| cnn_asp + wav2vec2 + wavlm | **97.61%** | 2.30% | 0.9967 | 26/1088 |
| cnnlstm + hubert + wav2vec2 | **96.88%** | 3.12% | 0.9946 | 34/1088 |
| cnnlstm + hubert + wavlm | **97.24%** | 2.76% | 0.9946 | 30/1088 |
| cnnlstm + wav2vec2 + wavlm | **94.12%** | 5.88% | 0.9868 | 64/1088 |
| hubert + wav2vec2 + wavlm | **99.45%** | 0.55% | 0.9999 | 6/1088 |
| ast + cnn_asp + cnnlstm + hubert | **98.16%** | 1.56% | 0.9967 | 20/1088 |
| ast + cnn_asp + cnnlstm + wav2vec2 | **97.61%** | 2.39% | 0.9957 | 26/1088 |
| ast + cnn_asp + cnnlstm + wavlm | **97.79%** | 2.11% | 0.9941 | 24/1088 |
| ast + cnn_asp + hubert + wav2vec2 | **98.71%** | 1.19% | 0.9992 | 14/1088 |
| ast + cnn_asp + hubert + wavlm | **98.71%** | 1.29% | 0.9992 | 14/1088 |
| ast + cnn_asp + wav2vec2 + wavlm | **98.35%** | 1.47% | 0.9984 | 18/1088 |
| ast + cnnlstm + hubert + wav2vec2 | **97.79%** | 2.30% | 0.9970 | 24/1088 |
| ast + cnnlstm + hubert + wavlm | **97.98%** | 2.11% | 0.9970 | 22/1088 |
| ast + cnnlstm + wav2vec2 + wavlm | **96.14%** | 3.86% | 0.9939 | 42/1088 |
| ast + hubert + wav2vec2 + wavlm | **98.71%** | 1.19% | 0.9992 | 14/1088 |
| cnn_asp + cnnlstm + hubert + wav2vec2 | **97.98%** | 2.02% | 0.9963 | 22/1088 |
| cnn_asp + cnnlstm + hubert + wavlm | **97.98%** | 1.93% | 0.9959 | 22/1088 |
| cnn_asp + cnnlstm + wav2vec2 + wavlm | **97.06%** | 3.03% | 0.9940 | 32/1088 |
| cnn_asp + hubert + wav2vec2 + wavlm | **98.53%** | 1.47% | 0.9987 | 16/1088 |
| cnnlstm + hubert + wav2vec2 + wavlm | **97.24%** | 2.76% | 0.9955 | 30/1088 |
| ast + cnn_asp + cnnlstm + hubert + wav2vec2 | **98.16%** | 2.02% | 0.9976 | 20/1088 |
| ast + cnn_asp + cnnlstm + hubert + wavlm | **98.16%** | 2.11% | 0.9973 | 20/1088 |
| ast + cnn_asp + cnnlstm + wav2vec2 + wavlm | **97.79%** | 2.21% | 0.9962 | 24/1088 |
| ast + cnn_asp + hubert + wav2vec2 + wavlm | **98.71%** | 1.10% | 0.9993 | 14/1088 |
| ast + cnnlstm + hubert + wav2vec2 + wavlm | **98.35%** | 1.65% | 0.9975 | 18/1088 |
| cnn_asp + cnnlstm + hubert + wav2vec2 + wavlm | **98.16%** | 1.84% | 0.9966 | 20/1088 |
| ast + cnn_asp + cnnlstm + hubert + wav2vec2 + wavlm | **98.16%** | 2.02% | 0.9978 | 20/1088 |

**Seluruh 26 run digabung:** 98.71%  EER 1.19%  AUC 0.9984  salah 14/1088

## 4. Terbaik

**hubert** → **99.82%**, EER 0.18%, AUC 0.9999, **2 salah dari 1088**

### Sisa error

| label benar | skor | posisi vs ambang |
|---|---|---|
| real | 0.0545 | di atas (ambang 0.0403) |
| fake | 0.0272 | di bawah (ambang 0.0403) |

Skor `real` yang salah: [0.0545]
Skor `fake` yang salah: [0.0272]
**Terbalik urutannya: YA — tidak ada ambang yang bisa memperbaikinya**

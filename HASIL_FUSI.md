# Varian Fusi: Mengatasi Dilusi Model Dominan

Rata-rata probabilitas polos membuat HuBERT (99,63%) turun ketika digabung dengan model lemah. Di sini diuji skema fusi alternatif.

Run HuBERT tersedia: 8 (seed [7, 42, 99, 555, 1337, 2024, 2718, 31415])

## 1. Perilaku tiap seed HuBERT pada sampel yang salah di ensemble

| sampel | label | seed 1337 | seed 2024 | seed 2718 | seed 31415 | seed 42 | seed 555 | seed 7 | seed 99 | ensemble |
|---|---|---|---|---|---|---|---|---|---|---|
| 19 | **real** | ✅ 0.032 | ❌ 0.134 | ❌ 0.059 | ❌ 0.050 | ❌ 0.052 | ❌ 0.057 | ✅ 0.032 | ✅ 0.020 | ❌ 0.055 |
| 1073 | **fake** | ❌ 0.020 | ❌ 0.041 | ✅ 0.015 | ❌ 0.020 | ❌ 0.031 | ❌ 0.036 | ✅ 0.039 | ❌ 0.017 | ❌ 0.027 |

Sampel yang salah di **SEMUA** seed: **0/2**

→ 2 sampel diperbaiki oleh sebagian seed. Menambah seed berpeluang membaliknya.

## 2. Perbandingan skema fusi (seluruh arsitektur)

| skema | akurasi | EER | AUC | salah |
|---|---|---|---|---|
| rata-rata probabilitas (baseline) | **98.16%** | 2.02% | 0.9978 | 20/1088 |
| rata-rata logit | **99.26%** | 0.64% | 0.9994 | 8/1088 |
| rata-rata peringkat | **99.26%** | 0.74% | 0.9992 | 8/1088 |
| median probabilitas | **98.90%** | 1.19% | 0.9992 | 12/1088 |
| maksimum probabilitas | **97.24%** | 2.67% | 0.9951 | 30/1088 |
| berbobot AUC^4 | **98.16%** | 1.84% | 0.9981 | 20/1088 |
| berbobot AUC^16 | **98.71%** | 1.10% | 0.9989 | 14/1088 |
| berbobot AUC^64 | **98.90%** | 1.19% | 0.9996 | 12/1088 |
| top-1 menurut AUC (hubert) | **99.82%** | 0.18% | 0.9999 | 2/1088 |
| top-2 menurut AUC (hubert, wavlm) | **99.82%** | 0.18% | 1.0000 | 2/1088 |
| top-3 menurut AUC (hubert, wavlm, cnn_asp) | **98.35%** | 1.75% | 0.9984 | 18/1088 |

**Terbaik: top-1 menurut AUC (hubert) → 99.82%, 2 salah dari 1088**

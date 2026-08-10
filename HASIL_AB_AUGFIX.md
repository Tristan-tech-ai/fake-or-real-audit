# A/B: Perbaikan Bug Augmentasi Statis (P0-1) pada HuBERT Large

Bug: RNG augmentasi di-seed hanya dari (nama berkas, seed run) sehingga tiap
berkas menerima satu varian augmentasi yang sama di seluruh epoch.
Perbaikan: epoch ikut masuk ke seed + `persistent_workers=False` pada loader latih.

Semua run: HuBERT Large, split resmi, augmentasi codec, batch 16, 10 epoch,
ambang prior-matched, seed {42, 1337, 2024}.

| kondisi | n | akurasi | std | EER | AUC | terbaik | salah (terbaik) |
|---|---|---|---|---|---|---|---|
| Augmentasi **beku** (bug) | 0 | — | | | | | |
| Augmentasi **per-epoch** (fix) | 14 | **96.53%** | ±1.93 | 3.46% | 0.9949 | **99.45%** | 6/1088 |

## Ensemble antar-seed

| kondisi | akurasi | EER | AUC | salah |
|---|---|---|---|---|
| Augmentasi per-epoch | **99.45%** | 0.64% | 0.9999 | 6/1088 |
| **Gabungan seluruh 14 run HuBERT** | **99.45%** | 0.64% | 0.9999 | 6/1088 |

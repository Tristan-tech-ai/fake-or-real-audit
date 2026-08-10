# Uji Penentu: Apakah Sisa Error Bersifat Struktural?

Sampel uji: 1088. Ensemble HuBERT salah pada **2** berkas.

Pertanyaan: adakah arsitektur lain yang benar pada berkas-berkas itu?

## Prediksi tiap arsitektur pada sampel keras

| idx | berkas | label | `ast` | `cnn_asp` | `cnnlstm` | `hubert` | `wav2vec2` | `wavlm` | ada yg benar? |
|---|---|---|---|---|---|---|---|---|---|
| 19 | `file1097.wav_16k.wav_norm.` | **real** | ❌0.170 | ❌0.388 | ❌0.523 | ❌0.055 | ❌0.050 | ✅0.008 | ✅ ADA |
| 1073 | `file932.wav_16k.wav_norm.w` | **fake** | ✅0.295 | ✅0.459 | ✅0.180 | ❌0.027 | ✅0.031 | ✅0.026 | ✅ ADA |

**Sampel yang salah di SELURUH 6 arsitektur: 0/2**

## Cek lebih ketat: seluruh 21 run individual

| idx | label | run yang BENAR | dari total |
|---|---|---|---|
| 19 | real | hubert/s1337, hubert/s7, hubert/s99, wavlm/s2024, wavlm/s42 | 5/26 |
| 1073 | fake | ast/s1337, ast/s2024, ast/s42, cnn_asp/s1337, cnn_asp/s2024, cnn_asp/s42… | 19/26 |

**Sampel yang salah di SELURUH 26 run: 0**

## Kesimpulan

**2 dari 2** sampel BISA diklasifikasi benar oleh setidaknya satu run. Artinya masih ada ruang: fusi/gating yang lebih cerdas berpotensi menaikkan hasil hingga **100.00%**.

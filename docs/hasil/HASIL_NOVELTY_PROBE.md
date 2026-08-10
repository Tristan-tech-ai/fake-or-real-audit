# Dua Probe Novelty

## Probe A — Inversi polaritas lintas korpus

AUC < 0,5 berarti model memberi skor 'palsu' lebih tinggi kepada audio ASLI. Ini bukan sekadar performa buruk; ini pembalikan arah keputusan.

Total pengukuran AUC yang diperiksa: **119**
Yang terbalik (AUC < 0,5): **3**

| model | dataset | AUC | AUC bila dibalik |
|---|---|---|---|
| Nes2Net-X SOTA (ASVspoof) | FoR-2sec | **0.0233** | 0.9767 |
| ?[none] | for-rerec | **0.3137** | 0.6863 |
| ?[codec] | for-rerec | **0.4997** | 0.5003 |

Inversi terjadi HANYA pada model yang dilatih di korpus lain lalu diuji lintas korpus. Model yang dilatih pada FoR tidak menunjukkannya di FoR — jadi ini spesifik pergeseran domain, bukan cacat arsitektur.

## Probe B — Ketergantungan pintasan sebagai prediktor generalisasi

Proksi ketergantungan pintasan: recall pada TTS **2019 non-MP3** (Tacotron2, SpeedySpeech, VITS). Model yang belajar jejak MP3 dari FoR akan buta terhadap TTS lama yang TIDAK dikompresi MP3.

| model | recall TTS-2019 non-MP3 | recall TTS 2025-26 | selisih | akurasi FoR |
|---|---|---|---|---|
| hubert[fullbg] | 63.1% | 87.3% | -24.2 pp | 95.7% |
| hubert[fullbg] | 59.3% | 95.4% | -36.1 pp | 95.7% |
| hubert[fullbg] | 52.8% | 91.3% | -38.5 pp | 95.7% |
| hubert[full] | 59.9% | 92.2% | -32.4 pp | 95.0% |
| hubert[full] | 25.4% | 80.2% | -54.7 pp | 95.0% |
| wavlm[fullbgrb] | 98.4% | 87.8% | +10.6 pp | 98.9% |
| wavlm[fullbgrb] | 100.0% | 93.1% | +6.9 pp | 98.9% |
| wavlm[fullbgrb] | 99.6% | 97.2% | +2.4 pp | 98.9% |
| wavlm[fullbg] | 99.9% | 94.5% | +5.4 pp | 98.7% |
| wavlm[fullbg] | 83.4% | 90.3% | -6.8 pp | 98.7% |
| wavlm[fullbg] | 99.4% | 92.9% | +6.5 pp | 98.7% |
| wavlm[full] | 93.9% | 88.2% | +5.6 pp | 98.4% |
| wavlm[full] | 99.3% | 95.8% | +3.6 pp | 98.4% |
| nes2net[fullbgrb] | 76.8% | 92.8% | -16.0 pp | 97.5% |
| nes2net[fullbgrb] | 98.2% | 97.5% | +0.7 pp | 97.5% |
| nes2net[fullbgrb] | 80.1% | 90.8% | -10.6 pp | 97.5% |
| nes2net[fullbg] | 87.3% | 92.6% | -5.2 pp | 97.1% |
| nes2net[fullbg] | 95.0% | 89.2% | +5.7 pp | 97.1% |
| nes2net[fullbg] | 94.1% | 98.7% | -4.6 pp | 97.1% |
| nes2net[soft] | 17.0% | 64.2% | -47.2 pp | 96.8% |
| nes2net[soft] | 91.7% | 83.2% | +8.5 pp | 96.8% |
| nes2net[soft] | 89.8% | 95.5% | -5.7 pp | 96.8% |
| nes2net[full] | 93.8% | 98.5% | -4.7 pp | 93.8% |
| nes2net[full] | 96.9% | 97.6% | -0.7 pp | 93.8% |
| nes2net[full] | 55.7% | 88.8% | -33.2 pp | 93.8% |
| nes2net[fullrb] | 66.4% | 91.0% | -24.6 pp | 98.5% |
| nes2net[fullrb] | 29.8% | 80.3% | -50.5 pp | 98.5% |
| nes2net[fullrb] | 85.8% | 91.7% | -5.9 pp | 98.5% |
| wavlm[full] | 99.2% | 81.2% | +18.1 pp | 98.4% |
| hubert[full] | 2.3% | 84.6% | -82.2 pp | 95.0% |

- Korelasi **akurasi FoR** vs recall TTS modern: **r = -0.011**
- Korelasi **recall TTS-2019 non-MP3** vs recall TTS modern: **r = +0.637**

Recall pada TTS lama non-MP3 lebih memprediksi performa pada TTS modern daripada akurasi FoR.

> Catatan: hanya 30 model. Korelasi pada n sekecil ini bersifat indikatif, belum kesimpulan.

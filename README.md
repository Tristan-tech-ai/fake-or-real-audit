# Audit Deteksi Deepfake Audio pada Fake-or-Real

Angka akurasi tunggal pada satu dataset mencampur tiga hal yang gagal dengan cara
berbeda: apa yang dipelajari model, seberapa baik ia memisahkan kedua kelas, dan
di mana ambang keputusannya berada. Repositori ini memisahkan ketiganya dan
melaporkan mana yang bertahan setelah ragam antar inisialisasi diukur.

**Mulai dari [NASKAH.pdf](NASKAH.pdf).** Naskah lengkap, delapan belas halaman,
sembilan gambar, empat puluh enam rujukan, dan satu bagian berisi klaim yang
ditarik beserta alasannya. Setiap angka di dalamnya dihitung ulang dari berkas
hasil ketika dokumen dibangun, sehingga tidak mungkin ada angka usang.

## Tiga temuan yang bertahan

1. **Riwayat kompresi berkorelasi dengan label, tetapi hanya pada sebagian
   partisi.** 90,7 persen sampel palsu pada data latih berasal dari berkas MP3
   sementara tidak satu pun sampel asli. Pada data uji korelasi itu nol. Dihitung
   langsung dari manifest, tanpa model dan tanpa keacakan.
2. **Sebagian besar selisih antar protokol berasal dari ambang, bukan protokol.**
   Pada partisi resmi, akurasi jatuh ke 50,03 persen pada ambang tetap 0,5 tetapi
   bertahan 92,56 persen pada ambang prior-matched, dengan AUC 97,56 persen.
   Daya pisah model tidak runtuh; letak ambangnya yang bergeser.
3. **Empat dari enam perbandingan utama tidak melampaui ragam antar
   inisialisasi.** Yang bertahan hanya dua, keduanya berselisih puluhan poin.

## Menjalankan

```bash
pip install -e ".[dev]"
```

Membangun ulang seluruh laporan, gambar, dan naskah dalam urutan yang benar:

```bash
py -m forensik.pipeline
```

Menjalankan satu laporan saja:

```bash
py -m forensik.laporan.signifikansi
```

Melatih satu model, memerlukan dataset dan GPU:

```bash
py -m forensik.latih --model wavlm --split official --augment full --epochs 10
```

Menguji:

```bash
py -m pytest
```

## Susunan

| Jalur | Isi |
|---|---|
| `forensik/runs.py` | pemuatan dan pengelompokan run; kunci kelompok selalu memuat konfigurasi |
| `forensik/stats.py` | uji Welch, koreksi Holm, uji permutasi, bootstrap |
| `forensik/metrics.py` | akurasi, AUC, EER, ambang prior-matched, kalibrasi |
| `forensik/manifest.py` | audit provenance codec langsung dari manifest |
| `forensik/laporan/` | pembangkit laporan, keluarannya ke `docs/hasil/` |
| `forensik/evaluasi/` | evaluasi lintas dataset dan lintas generasi TTS |
| `forensik/audit/` | penyelidikan pintasan pada dataset |
| `forensik/gambar.py` | sembilan gambar naskah |
| `forensik/naskah/` | pembangkit NASKAH.pdf beserta daftar pustakanya |
| `forensik/periksa_konfigurasi.py` | pengaman yang menolak membandingkan run berbeda konfigurasi |
| `runs/` | skor uji tiap run; bobot model tidak disertakan |
| `docs/hasil/` | laporan yang dibangkitkan ulang |
| `docs/catatan-riset/` | catatan tahap awal, sebagian memuat angka yang sudah ditarik |

## Pengaman

Tujuh kekeliruan dalam penelitian ini berasal dari pola yang sama: dua run
dibandingkan sebagai pasangan terkontrol padahal konfigurasi pelatihannya
berbeda, sehingga ragam antar konfigurasi terlaporkan sebagai ragam antar
inisialisasi. Dua hal dipasang supaya pola itu tidak terulang.

Pertama, `Config` pada `forensik/runs.py` selalu memuat ukuran batch dan jumlah
epoch, sehingga dua run yang berbeda konfigurasi tidak mungkin masuk kelompok
yang sama. Kedua, `forensik.periksa_konfigurasi` berjalan lebih dahulu di dalam
pipeline dan menghentikan semuanya bila menemukan kelompok tercampur yang belum
terdaftar beserta alasannya.

## Data

Dataset Fake-or-Real varian for-2sec tersedia dari penerbitnya dan tidak
disertakan di sini. `manifest.csv` disertakan karena audit provenance codec
bergantung padanya dan tidak memerlukan berkas audionya. Bobot model tidak
disertakan karena ukurannya; `runs/` hanya memuat skor uji dan berkas hasil.

## Lisensi

Kode di bawah MIT. Dataset tunduk pada lisensi penerbitnya masing-masing.

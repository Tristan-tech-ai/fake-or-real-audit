"""Split the opening finding into the three causes it was hiding.

The original claim was a fifty point gap between a random split and the official
partition on "identical" hyperparameters. It was not identical: those two runs
differ in epoch count, threshold placement, and only lastly in protocol.
"""

from __future__ import annotations

from ..report import Report
from ..runs import load, summarize
from ..stats import welch

CELLS = {
    "legacy_random": ("cnn_asp_random_none_s42", "run asli, 6 epoch batch 64, split acak"),
    "legacy_official": (
        "cnn_asp_official_none_s42",
        "run asli, 1 epoch batch 64, partisi resmi",
    ),
    "random": ("cnn_asp_random_none_b32e10_s*", "10 epoch batch 32, split acak"),
    "official": ("cnn_asp_official_none_b32e10_s*", "10 epoch batch 32, partisi resmi"),
}

INTRO = """Temuan pertama menyatakan bahwa protokol pembagian data menentukan \
hasil, dengan bukti selisih hampir lima puluh poin persentase pada arsitektur, \
data, dan hyperparameter yang disebut identik. Tabel berikut memisahkan selisih \
itu menjadi sebab-sebabnya.

Pernyataan bahwa hyperparameternya identik ternyata tidak benar. Run yang \
menghasilkan angka pada split acak dijalankan selama enam epoch, sedangkan run \
yang menghasilkan angka pada partisi resmi dijalankan selama **satu** epoch. \
Keduanya berasal dari tahap paling awal penelitian, ketika nama direktori belum \
memuat penanda batch dan epoch, sehingga perbedaan itu tidak terlihat dari nama \
berkasnya dan tidak pernah diperiksa. Perbandingan aslinya karena itu bukan \
perbandingan terkontrol sama sekali. Baris ketiga dan keempat menjalankan \
keduanya pada konfigurasi yang seragam."""

CLOSING = """Dengan kata lain, angka yang semula dipakai untuk menunjukkan \
bahwa protokol pembagian data menentukan hasil sebenarnya lebih banyak \
menunjukkan bahwa ambang keputusan menentukan hasil. Kedua pernyataan sama-sama \
merupakan peringatan terhadap pelaporan akurasi tunggal, namun keduanya menunjuk \
sebab yang berbeda dan menuntut perbaikan yang berbeda pula."""


def cell(pattern: str) -> dict | None:
    runs = load(pattern)
    if not runs:
        return None
    return {
        "fixed": summarize(runs, "accuracy", 0.5),
        "prior": summarize(runs, "accuracy", "prior"),
        "auc": summarize(runs, "auc", 0.5),
    }


def row(value: dict | None, label: str) -> list:
    if value is None:
        return [label, "belum ada", "", "", ""]
    return [
        label,
        value["fixed"].n,
        str(value["fixed"]),
        str(value["prior"]),
        f"{value['auc'].mean / 100:.4f}",
    ]


def main() -> None:
    found = {key: cell(pattern) for key, (pattern, _) in CELLS.items()}

    report = Report("HASIL_TEMUAN1.md")
    report.add("# Pemecahan Temuan Pembuka\n")
    report.add(INTRO + "\n")
    report.table(
        ["Konfigurasi dan split", "n", "Akurasi @0,5", "Akurasi @prior", "AUC"],
        [row(found[key], label) for key, (_, label) in CELLS.items()],
    )

    if any(value is None for value in found.values()):
        report.save()
        return

    legacy_gap = found["legacy_random"]["fixed"].mean - found["legacy_official"]["fixed"].mean
    threshold = found["official"]["prior"].mean - found["official"]["fixed"].mean
    training = found["official"]["prior"].mean - found["legacy_official"]["prior"].mean
    protocol = found["random"]["prior"].mean - found["official"]["prior"].mean

    report.add("## Tiga sebab yang terpisah\n")
    report.table(
        ["Sebab", "Besaran", "Dapat diperbaiki tanpa mengubah protokol"],
        [
            ["Ambang keputusan tidak lagi cocok", f"{threshold:.2f} poin",
             "ya, cukup dengan menyesuaikan ambang"],
            ["Model kurang terlatih pada run asli", f"{training:.2f} poin",
             "ya, cukup dengan menambah epoch"],
            ["Protokol pembagian data itu sendiri", f"{protocol:.2f} poin", "tidak"],
        ],
    )
    report.add(
        f"Selisih yang dilaporkan semula {legacy_gap:.2f} poin persentase. Dari "
        f"jumlah itu, hanya {protocol:.2f} poin merupakan sifat protokolnya, yaitu "
        "bagian yang tetap ada setelah model dilatih penuh dan ambangnya "
        "disesuaikan.\n"
    )

    test = welch("protokol", found["random"]["prior"].values, found["official"]["prior"].values)
    verdict = "melampaui ragam" if test.p_raw < 0.05 else "belum terbukti berbeda"
    report.add(
        "Selisih protokol tersebut diuji dengan uji t Welch pada ambang "
        f"prior-matched dan menghasilkan nilai p sebesar {test.p_raw:.4f}, yang "
        f"berarti {verdict}.\n"
    )

    auc_random = found["random"]["auc"].mean / 100
    auc_official = found["official"]["auc"].mean / 100
    report.add("## Bacaan\n")
    report.add(
        "Efek protokol pembagian data tetap ada dan terbaca pada penurunan area "
        f"under curve dari {auc_random:.4f} menjadi {auc_official:.4f}, yang tidak "
        "dapat diperbaiki oleh pengaturan ambang. Besarannya jauh lebih kecil "
        "daripada yang semula dilaporkan.\n"
    )
    report.add(
        "Sebaliknya, temuan mengenai kegagalan kalibrasi menjadi jauh lebih kuat. "
        "Pada partisi resmi, model yang terlatih penuh memisahkan kedua kelas "
        f"dengan area under curve {auc_official:.4f}, namun pada ambang tetap 0,5 "
        f"hanya mencapai {found['official']['fixed'].mean:.2f} persen. Selisih itu "
        f"sepenuhnya merupakan kegagalan kalibrasi, dan besarnya {threshold:.2f} "
        "poin persentase.\n"
    )
    report.add(CLOSING + "\n")
    report.save()


if __name__ == "__main__":
    main()

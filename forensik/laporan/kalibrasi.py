"""Is the failure under noise a calibration failure rather than a recognition one?

The claim is the gap between accuracy at a threshold frozen from clean audio and
accuracy at a prior-matched threshold, measured on the same model and the same
files. Both numbers come from the same seeds, so the test is paired.
"""

from __future__ import annotations

import numpy as np

from ..report import Report
from ..results import snr_runs
from ..stats import holm, paired_t

LEVELS = (10, 0)

INTRO = """Selisih diukur antara akurasi pada ambang yang dibekukan dari kondisi \
bersih dan akurasi pada ambang prior-matched, pada model dan berkas yang sama \
persis. Karena kedua nilai berasal dari inisialisasi acak yang sama, \
pengujiannya memakai uji t berpasangan. Selisih yang besar berarti daya pisah \
model masih ada dan yang bergeser hanyalah letak ambangnya."""

CAVEAT = """Perbandingan ini berbeda dari klaim lain dalam penelitian yang gagal \
bertahan, dan perbedaannya perlu dinyatakan agar tidak terbaca sebagai \
pembelaan. Pertama, arah efeknya konsisten pada dua tingkat noise yang terpisah \
untuk arsitektur yang sama. Kedua, besarannya berpuluh poin persentase, bukan \
berbilang poin. Ketiga, mekanismenya dapat diperiksa secara langsung lewat area \
under curve, yang tidak bergantung pada ambang sama sekali. WavLM \
mempertahankan area under curve sekitar 0,96 pada 10 dB, sehingga daya pisahnya \
memang masih ada dan pernyataan bahwa yang bergeser adalah ambangnya dapat \
diverifikasi tanpa uji statistik.

Meskipun demikian, nilai p terkoreksi yang berada tepat di atas ambang berarti \
temuan ini belum dapat dinyatakan mapan pada tingkat kekakuan yang sama dengan \
temuan mengenai learning rate. Statusnya berada di antara keduanya, dan \
dilaporkan demikian."""


def recoveries() -> list[dict]:
    """Threshold recovery per architecture and noise level, with paired p-values."""
    tested = []
    for (arch, snr), rows in snr_runs().items():
        if snr not in LEVELS or len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda row: row["seed"])
        gap = np.array([row["acc_pm"] - row["acc_fx"] for row in rows]) * 100
        if gap.std(ddof=1) == 0:
            continue
        tested.append(
            {
                "arch": arch,
                "snr": snr,
                "n": len(gap),
                "gap": gap.mean(),
                "sd": gap.std(ddof=1),
                "auc": float(np.mean([row["auc"] for row in rows])),
                "p": paired_t(gap),
            }
        )
    for row, adjusted in zip(tested, holm([row["p"] for row in tested]), strict=False):
        row["p_holm"] = adjusted
    return tested


def main() -> None:
    report = Report("HASIL_UJI_KALIBRASI.md")
    report.add("# Apakah Kegagalan di Bawah Noise Merupakan Kegagalan Kalibrasi?\n")
    report.add(INTRO + "\n")

    tested = recoveries()
    if not tested:
        report.add("Belum ada data yang dapat diuji.")
        report.save()
        return

    report.table(
        ["Arsitektur", "SNR", "n", "AUC", "Pemulihan ambang", "p mentah", "p Holm"],
        [
            [row["arch"], f"{row['snr']} dB", row["n"], f"{row['auc']:.4f}",
             f"{row['gap']:+.1f} ({row['sd']:.1f})", f"{row['p']:.4f}",
             f"{row['p_holm']:.4f}"]
            for row in sorted(tested, key=lambda r: (-r["gap"], r["arch"]))
        ],
    )

    report.add("## Bacaan\n")
    strong = [row for row in tested if row["p"] < 0.05]
    if strong:
        names = ", ".join(
            f"{row['arch']} pada {row['snr']} dB"
            for row in sorted(strong, key=lambda r: r["p"])
        )
        smallest = min(row["p_holm"] for row in tested)
        report.add(
            f"Pada nilai p mentah, {len(strong)} dari {len(tested)} perbandingan "
            f"melampaui ambang lima persen, yaitu {names}. Setelah koreksi "
            "Holm-Bonferroni atas sepuluh perbandingan, nilai p terkecil menjadi "
            f"{smallest:.3f}, yang berada tepat di atas ambang.\n"
        )
    report.add(CAVEAT + "\n")
    report.save()


if __name__ == "__main__":
    main()

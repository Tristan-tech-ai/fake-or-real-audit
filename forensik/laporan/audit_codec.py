"""Codec provenance leakage, computed straight from the manifest.

No model, no training, no randomness, so this number has no seed spread and
needs no significance test. That is why it survives unconditionally while most
other findings in this study do not.
"""

from __future__ import annotations

from ..manifest import codec_provenance
from ..report import Report

LABELS = {"real": "asli", "fake": "palsu"}

INTRO = """Perhitungan ini membaca nama berkas dan label pada manifest secara \
langsung. Tidak ada model, pelatihan, maupun keacakan yang terlibat, sehingga \
hasilnya tidak memiliki ragam antar inisialisasi dan tidak memerlukan pengujian \
statistik. Inilah sebabnya temuan ini bertahan tanpa syarat sementara sebagian \
besar temuan lain dalam penelitian ini tidak."""

CLOSING = """Akibatnya, sebuah model yang belajar mengenali jejak kompresi akan \
mencapai akurasi tinggi pada data latih dan validasi, lalu kehilangan seluruh \
isyarat itu pada data uji. Isyarat yang dipelajari bukan jejak sintesis \
melainkan riwayat berkas, dan riwayat itu berkorelasi dengan label hanya pada \
sebagian partisi."""


def main() -> None:
    provenance = codec_provenance()

    report = Report("audit_codec_report.md")
    report.add("# Audit Provenance Codec pada Fake-or-Real\n")
    report.add(INTRO + "\n")
    report.table(
        ["Partisi resmi", "Kelas", "Total berkas", "Berasal MP3", "Persen"],
        [
            [split, LABELS[cls], value.total, value.from_mp3,
             f"**{value.percent:.1f}**"]
            for (split, cls), value in sorted(
                provenance.items(), key=lambda item: (item[0][0], LABELS[item[0][1]])
            )
        ],
    )

    train = provenance[("training", "fake")]
    test = provenance[("testing", "fake")]
    report.add(
        f"Pada data latih, {train.from_mp3} dari {train.total} sampel palsu berasal "
        f"dari berkas MP3, yaitu {train.percent:.1f} persen. Pada data uji, "
        f"{test.from_mp3} dari {test.total} sampel palsu berasal dari MP3, yaitu "
        f"{test.percent:.1f} persen. Tidak ada satu pun sampel asli yang berasal "
        "dari MP3 pada partisi mana pun.\n"
    )
    report.add(CLOSING + "\n")
    report.save()


if __name__ == "__main__":
    main()

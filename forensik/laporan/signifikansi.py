"""Does each difference exceed the spread between random initialisations?

Six comparisons are tested at once, so raw p-values cannot be read as they
stand and Holm-Bonferroni correction decides the verdict.
"""

from __future__ import annotations

from ..report import Report
from ..runs import scores
from ..stats import correct, welch

COMPARISONS = [
    (
        "AST: encoder dilatih vs dibekukan",
        "ast_official_fullUF_b32e10_s*",
        "ast_official_full_b32e10_s*",
    ),
    (
        "AST: encoder dilatih vs proposal",
        "ast_official_fullUF_b32e10_s*",
        "ast_official_proposalULRPK_b32e20_s*",
    ),
    (
        "WavLM: encoder dibekukan vs dilatih",
        "wavlm_official_full_b16e10_s*",
        "wavlm_official_fullUF_b16e10_s*",
    ),
    (
        "HuBERT: encoder dilatih vs dibekukan",
        "hubert_official_fullUF_b32e10_s*",
        "hubert_official_full_b32e10_s*",
    ),
    (
        "WavLM: rekayasa dibekukan vs proposal",
        "wavlm_official_full_b16e10_s*",
        "wavlm_official_proposalULRPK_b16e20_s*",
    ),
    (
        "HuBERT: rekayasa dilatih vs proposal",
        "hubert_official_fullUF_b32e10_s*",
        "hubert_official_proposalULRPK_b32e20_s*",
    ),
]

INTRO = """Tiap baris membandingkan dua konfigurasi pada partisi resmi dengan \
ambang prior-matched, memakai uji t Welch yang tidak mengandaikan ragam kedua \
kelompok sama. Ukuran sampel kecil, yaitu paling banyak tiga inisialisasi per \
sel, sehingga uji ini berdaya rendah. Nilai p yang besar berarti belum terbukti \
berbeda, dan bukan terbukti sama.

Enam perbandingan diuji sekaligus, sehingga nilai p mentah tidak dapat dibaca \
apa adanya. Menguji enam hipotesis pada ambang 0,05 memberi peluang sekitar 26 \
persen untuk mendapatkan setidaknya satu hasil yang tampak bermakna semata \
karena kebetulan. Karena itu koreksi Holm-Bonferroni diterapkan, dan keputusan \
diambil dari nilai p terkoreksi."""

DISCUSSION = """Hasilnya terbelah bersih menjadi dua kelompok yang tidak saling \
berdekatan.

Kelompok pertama adalah perbandingan antara konfigurasi proposal dan \
konfigurasi rekayasa pada kedua model swa-selia berukuran besar. Selisihnya \
berpuluh poin persentase dan nilai p terkoreksinya jauh di bawah ambang, \
sehingga kesimpulannya kokoh. Perlu dicatat bahwa pada tahap sebelumnya, ketika \
tiap sel baru memiliki tiga inisialisasi, kedua perbandingan ini justru \
berhenti pada nilai p terkoreksi 0,0520 yaitu tepat di atas ambang. Penyebabnya \
bukan efek yang kecil melainkan derajat bebas yang sangat sedikit dan simpangan \
baku yang besar pada sel konfigurasi proposal. Menambah inisialisasi keempat \
dan kelima menyelesaikannya, dan itu memang tanggapan yang tepat terhadap nilai \
p yang berhenti di ambang.

Kelompok kedua adalah perbandingan antara membekukan dan melatih encoder. \
Selisihnya berada pada orde yang sama dengan simpangan bakunya sendiri, dan \
tidak satu pun terbukti berbeda meskipun sebagian sel sudah memiliki empat atau \
lima inisialisasi. Untuk kelompok ini, penelitian ini tidak berhak menyatakan \
bahwa satu perlakuan lebih baik daripada yang lain.

Kesimpulan keseluruhannya sempit tetapi jelas. Yang menentukan hasil pada \
partisi resmi adalah besaran learning rate relatif terhadap encodernya, bukan \
keputusan membekukan atau melatih encoder itu sendiri.

Konsekuensinya bagi keseluruhan penelitian cukup besar dan perlu dinyatakan \
terus terang. Beberapa kesimpulan yang sempat ditarik lebih awal, ketika tiap \
sel baru dijalankan sekali, ternyata tidak bertahan setelah ragam antar \
inisialisasi diukur. Yang tersisa sebagai temuan yang kokoh adalah hal-hal yang \
selisihnya berpuluh poin, bukan berbilang poin."""


def verdict(p_holm: float) -> str:
    if p_holm < 0.05:
        return "selisih melampaui ragam"
    if p_holm < 0.15:
        return "di garis batas, belum meyakinkan"
    return "belum terbukti berbeda"


def main() -> None:
    tested, missing = [], []
    for name, pattern_a, pattern_b in COMPARISONS:
        a, b = scores(pattern_a), scores(pattern_b)
        if a is None or b is None or a.n < 2 or b.n < 2:
            missing.append(name)
        else:
            tested.append(welch(name, a.values, b.values))

    report = Report("HASIL_SIGNIFIKANSI.md")
    report.add("# Apakah Selisihnya Lebih Besar daripada Ragam Antar Inisialisasi?\n")
    report.add(INTRO + "\n")
    report.table(
        ["Perbandingan", "n", "Rerata A", "Rerata B", "Selisih", "p mentah",
         "p Holm", "Bacaan"],
        [
            [c.name, f"{len(c.a)}/{len(c.b)}", f"{c.a.mean():.2f}",
             f"{c.b.mean():.2f}", f"{c.diff:+.2f}", f"{c.p_raw:.4f}",
             f"{c.p_holm:.4f}", f"**{verdict(c.p_holm)}**"]
            for c in correct(tested)
        ]
        + [[name, "", "belum ada", "", "", "", "", ""] for name in missing],
    )
    report.add("## Bacaan\n")
    report.add(DISCUSSION + "\n")
    report.save()


if __name__ == "__main__":
    main()

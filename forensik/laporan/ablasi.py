"""Ablation ladder: which improvement buys how much.

Every rung uses AST on the official partition, adding one change at a time so
each gap reflects a single variable. Accuracy is reported at the prior-matched
threshold throughout, keeping the threshold axis out of the ladder.
"""

from __future__ import annotations

import re

from ..report import Report
from ..runs import load, summarize
from ..stats import correct, welch

LADDER = [
    (
        "L1",
        "Konfigurasi proposal apa adanya",
        "ast_official_proposalULRPK_b32e20_s42",
        "LR 0,001 seragam dengan encoder ikut dilatih, normalisasi peak, "
        "20 epoch tanpa early stopping, augmentasi noise saja",
    ),
    (
        "L2",
        "Normalisasi loudness",
        "ast_official_proposalULR_b32e20_s42",
        "normalisasi peak diganti loudness, selebihnya sama",
    ),
    (
        "L3",
        "LR per model dan encoder dibekukan",
        "ast_official_proposal_b32e20_s42",
        "encoder tidak lagi dilatih, head 0,001 dan encoder 2e-5, ditambah "
        "agregasi berbobot antar lapisan",
    ),
    (
        "L4",
        "Early stopping pada EER",
        "ast_official_proposal_b32e10_s42",
        "10 epoch dengan pemilihan bobot terbaik menurut EER validasi",
    ),
    (
        "L5",
        "Augmentasi penuh",
        "ast_official_full_b32e10_s42",
        "augmentasi noise saja diganti augmentasi penuh, yaitu codec, noise, "
        "reverb, dan band-gain",
    ),
]

INTRO = """Semua langkah memakai AST pada partisi resmi Fake-or-Real, batch 32, \
seed 42. Tiap baris menambahkan satu perbaikan di atas baris sebelumnya, \
sehingga selisih antar baris hanya mencerminkan satu variabel.

Akurasi dilaporkan pada ambang prior-matched untuk seluruh langkah agar sumbu \
ambang tidak bercampur ke dalam tangga. Sumbangan ambang itu sendiri dipisahkan \
tersendiri di HASIL_DEKOMPOSISI.md.

Selisih tiap langkah diuji terhadap langkah sebelumnya dengan uji t Welch. \
Seluruh tangga dijalankan pada AST, yaitu arsitektur dengan ragam antar \
inisialisasi terbesar di antara yang diuji, sehingga selisih yang kecil di sini \
menuntut kehati-hatian khusus."""

CLOSING = """Pola yang muncul cukup jelas. Dua langkah dengan selisih terbesar, \
yaitu pembekuan encoder dan early stopping, memiliki nilai p mentah di bawah \
0,05 sedangkan dua langkah dengan selisih kecil tidak. Setelah koreksi untuk \
empat pengujian sekaligus, tidak ada satu pun yang bertahan di bawah ambang. \
Perlu diingat bahwa seluruh tangga ini dijalankan pada AST, yaitu arsitektur \
dengan ragam antar inisialisasi terbesar di antara yang diuji, sehingga daya \
ujinya paling rendah di sini dan bukan karena efeknya tidak ada.

Kesimpulan yang dapat dipertanggungjawabkan dari tangga ini karena itu terbatas. \
Arah tiap langkah konsisten dengan penjelasan mekanistik yang diajukan, tetapi \
besarannya belum dapat dipisahkan dari ragam pada ukuran sampel ini. Tangga \
ablasi lebih tepat dibaca sebagai peta kemungkinan sebab, bukan sebagai \
pengukuran sumbangan tiap perbaikan."""


def rung(pattern: str) -> dict | None:
    """A ladder step, averaged over every seed that exists for it."""
    runs = load(re.sub(r"_s\d+$", "_s*", pattern))
    if not runs:
        return None
    return {
        "accuracy": summarize(runs, "accuracy", "prior"),
        "auc": summarize(runs, "auc", 0.5).mean,
        "eer": summarize(runs, "eer", 0.5).mean,
    }


def main() -> None:
    steps = []
    previous = None
    for code, name, pattern, change in LADDER:
        value = rung(pattern)
        gap = None if value is None or previous is None else (
            value["accuracy"].mean - previous["accuracy"].mean
        )
        steps.append({"code": code, "name": name, "change": change,
                      "value": value, "gap": gap, "previous": previous})
        if value is not None:
            previous = value

    testable = [
        s for s in steps
        if s["value"] and s["previous"] and s["value"]["accuracy"].n > 1
        and s["previous"]["accuracy"].n > 1
    ]
    tested = correct([
        welch(s["code"], s["value"]["accuracy"].values, s["previous"]["accuracy"].values)
        for s in testable
    ])
    p_values = {c.name: c for c in tested}

    report = Report("HASIL_ABLASI.md")
    report.add("# Tangga Ablasi: Perbaikan Mana yang Membeli Berapa\n")
    report.add(INTRO + "\n")
    report.table(
        ["Langkah", "Perbaikan yang ditambahkan", "n", "Akurasi", "Selisih",
         "p mentah", "p Holm", "AUC", "EER"],
        [
            [s["code"], s["name"], "", "belum ada", "", "", "", "", ""]
            if not s["value"]
            else [
                s["code"], s["name"], s["value"]["accuracy"].n,
                str(s["value"]["accuracy"]),
                "" if s["gap"] is None else f"**{s['gap']:+.2f}**",
                "" if s["code"] not in p_values else f"{p_values[s['code']].p_raw:.3f}",
                "" if s["code"] not in p_values else f"{p_values[s['code']].p_holm:.3f}",
                f"{s['value']['auc'] / 100:.4f}", f"{s['value']['eer']:.2f}",
            ]
            for s in steps
        ],
    )

    present = [s for s in steps if s["value"]]
    if len(present) < 2:
        report.save()
        return

    first, last = present[0]["value"]["accuracy"].mean, present[-1]["value"]["accuracy"].mean
    report.add(
        f"Total kenaikan sepanjang tangga adalah {last - first:+.2f} poin "
        f"persentase, dari {first:.2f} persen menjadi {last:.2f} persen.\n"
    )

    report.add("## Rincian tiap langkah\n")
    for step in present:
        value, gap = step["value"], step["gap"]
        if gap is None:
            report.add(
                f"**{step['code']}, {step['name']}.** Titik tolak, yaitu "
                f"{step['change']}. Akurasi {value['accuracy'].mean:.2f} persen "
                f"dengan AUC {value['auc'] / 100:.4f}.\n"
            )
        else:
            direction = "menaikkan" if gap > 0 else "menurunkan" if gap < 0 else "tidak mengubah"
            report.add(
                f"**{step['code']}, {step['name']}.** Perubahan yang dilakukan "
                f"adalah {step['change']}. Langkah ini {direction} akurasi sebesar "
                f"{abs(gap):.2f} poin persentase menjadi "
                f"{value['accuracy'].mean:.2f} persen, dengan AUC "
                f"{value['auc'] / 100:.4f} dan EER {value['eer']:.2f} persen.\n"
            )

    report.add("## Bacaan\n")
    report.add(
        "Empat selisih diuji sekaligus, sehingga koreksi Holm-Bonferroni "
        "diterapkan dan keputusan diambil dari kolom p Holm.\n"
    )
    buckets = [
        ("Langkah yang selisihnya melampaui ragam antar inisialisasi: ",
         lambda p: p < 0.05),
        ("Langkah yang berada di garis batas dan belum dapat dinyatakan mapan: ",
         lambda p: 0.05 <= p < 0.15),
        ("Langkah yang selisihnya belum terbukti berbeda dari nol: ",
         lambda p: p >= 0.15),
    ]
    gaps = {s["code"]: s["gap"] for s in steps}
    for prefix, keep in buckets:
        chosen = [c for c in tested if keep(c.p_holm)]
        if chosen:
            report.add(
                prefix
                + "; ".join(
                    f"{c.name} ({gaps[c.name]:+.2f} poin, p Holm {c.p_holm:.3f})"
                    for c in chosen
                )
                + ".\n"
            )
    report.add(CLOSING + "\n")
    report.save()


if __name__ == "__main__":
    main()

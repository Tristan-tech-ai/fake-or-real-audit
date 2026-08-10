"""Does the band-gain generalisation claim survive seed variance?

The claim was stated on two axes, recall on 2025-2026 TTS and recall on older
uncompressed TTS, as a mean over three seeds whose spread was never tested.
RawBoost is tested the same way: testing band-gain while accepting its
comparison baseline unexamined would be a rigged contest.
"""

from __future__ import annotations

import numpy as np

from ..report import Report
from ..results import MODERN_ERA, _load
from ..stats import correct, welch

OLD_UNCOMPRESSED = (
    "tts_models_en_ljspeech_tacotron2-DDC",
    "tts_models_en_ljspeech_speedy-speech",
    "tts_models_en_ljspeech_vits",
)
SEEDS = ("s42", "s1337", "s2024")
ARCHITECTURES = ("nes2net", "wavlm", "hubert")
ADDITIONS = [
    ("fullbg", "band-gain"),
    ("fullrb", "RawBoost"),
    ("fullbgrb", "band-gain + RawBoost"),
]

INTRO = """Klaim mengenai band-gain dinyatakan pada dua sumbu, yaitu recall \
terhadap sistem text-to-speech generasi 2025 sampai 2026 dan recall terhadap \
sistem lama yang tidak dikompresi MP3. Angka yang dilaporkan berupa rerata atas \
tiga inisialisasi acak, namun selisihnya belum pernah diuji terhadap sebaran itu \
sendiri. Tabel berikut melakukan pengujian tersebut dengan uji t Welch dan \
koreksi Holm-Bonferroni."""

DISCUSSION = """Penyebabnya terlihat langsung pada kolom simpangan baku. Recall \
pada sumbu-sumbu ini jauh lebih tidak stabil daripada akurasi Fake-or-Real. \
Sebagai contoh, Nes2Net tanpa band-gain menghasilkan recall 93,8 dan 96,9 dan \
55,7 persen pada sistem lama non-MP3, sehingga selisih 10 poin persentase yang \
sempat dilaporkan sebagai keunggulan band-gain sebenarnya ditentukan hampir \
seluruhnya oleh satu inisialisasi yang buruk.

Hal yang sama berlaku bagi pembandingnya. Klaim bahwa RawBoost menurunkan \
generalisasi juga tidak bertahan, dengan selisih 7,33 dan 21,44 poin persentase \
yang keduanya berada di dalam ragam. Menguji band-gain sambil menerima klaim \
pembandingnya apa adanya akan menjadi pemilihan yang tidak sah, sehingga \
keduanya diuji dengan cara yang sama dan keduanya sama-sama tidak terbukti.

Satu pola tetap terlihat, yaitu pada sebarannya dan bukan pada reratanya. Pada \
lima dari enam perbandingan, band-gain menghasilkan simpangan baku yang lebih \
kecil, dalam dua kasus sekitar lima setengah kali lebih kecil. Pada satu \
perbandingan polanya terbalik. Pola ini dilaporkan sebagai pengamatan deskriptif \
dan tidak diuji secara formal, karena pengujian kesamaan ragam pada tiga \
inisialisasi memiliki daya yang bahkan lebih rendah daripada pengujian rerata.

Konsekuensinya, klaim mengenai keunggulan generalisasi band-gain harus ditarik \
sebagai temuan dan dinyatakan ulang sebagai pengamatan yang belum diuji.

Sumbu recall menuntut jumlah inisialisasi yang jauh lebih banyak daripada tiga. \
Dengan simpangan baku belasan poin persentase, mendeteksi selisih 10 poin secara \
meyakinkan membutuhkan puluhan inisialisasi, dan itu di luar anggaran komputasi \
penelitian ini. Keterbatasan tersebut dilaporkan apa adanya."""


def recalls(generations: dict, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Modern and old-uncompressed recall, one entry per seed."""
    modern, old = [], []
    for seed in SEEDS:
        row = generations.get(f"{prefix}_b16e10_{seed}")
        if not row:
            continue
        systems = row["tts"]
        recent = [s["recall"] for s in systems.values() if s["era"] == MODERN_ERA]
        legacy = [systems[k]["recall"] for k in OLD_UNCOMPRESSED if k in systems]
        if recent:
            modern.append(np.mean(recent) * 100)
        if legacy:
            old.append(np.mean(legacy) * 100)
    return np.array(modern), np.array(old)


def comparisons(generations: dict) -> list[dict]:
    rows = []
    for arch in ARCHITECTURES:
        base_modern, base_old = recalls(generations, f"{arch}_official_full")
        for suffix, label in ADDITIONS:
            with_modern, with_old = recalls(generations, f"{arch}_official_{suffix}")
            for axis, treated, baseline in [
                ("recall TTS 2025-2026", with_modern, base_modern),
                ("recall TTS 2019 non-MP3", with_old, base_old),
            ]:
                if len(treated) >= 2 and len(baseline) >= 2:
                    rows.append({"arch": arch, "axis": axis, "addition": label,
                                 "treated": treated, "baseline": baseline})
    tests = correct([welch(str(i), r["treated"], r["baseline"]) for i, r in enumerate(rows)])
    for row, test in zip(rows, tests, strict=False):
        row["test"] = test
    return rows


def verdict(p_holm: float) -> str:
    if p_holm < 0.05:
        return "melampaui ragam"
    return "di garis batas" if p_holm < 0.15 else "belum terbukti berbeda"


def spread_ratio(baseline: np.ndarray, treated: np.ndarray) -> str:
    without, with_ = baseline.std(ddof=1), treated.std(ddof=1)
    if with_ > 0 and without > with_:
        return f"{without / with_:.1f}x lebih kecil"
    return f"{with_ / without:.1f}x lebih besar" if without > 0 else "n/a"


def main() -> None:
    generations = _load("generations_results.json")
    if generations is None:
        print("generations_results.json belum ada")
        return

    report = Report("HASIL_UJI_KLAIM_BANDGAIN.md")
    report.add("# Apakah Klaim Generalisasi Band-Gain Bertahan?\n")
    report.add(INTRO + "\n")

    rows = comparisons(generations)
    if not rows:
        report.add("Belum ada pasangan yang dapat diuji.")
        report.save()
        return

    report.table(
        ["Arsitektur", "Tambahan", "Sumbu", "n", "Tanpa", "Dengan", "Selisih",
         "p mentah", "p Holm", "Bacaan"],
        [
            [r["arch"], r["addition"], r["axis"],
             f"{len(r['treated'])}/{len(r['baseline'])}",
             f"{r['baseline'].mean():.2f} ({r['baseline'].std(ddof=1):.2f})",
             f"{r['treated'].mean():.2f} ({r['treated'].std(ddof=1):.2f})",
             f"{r['test'].diff:+.2f}", f"{r['test'].p_raw:.4f}",
             f"{r['test'].p_holm:.4f}", f"**{verdict(r['test'].p_holm)}**"]
            for r in rows
        ],
    )

    report.add("## Sebaran, bukan rerata\n")
    report.add(
        "Rerata tidak dapat dibedakan, tetapi sebarannya berbeda secara konsisten. "
        "Tabel berikut membandingkan simpangan baku antar inisialisasi acak.\n"
    )
    report.table(
        ["Arsitektur", "Sumbu", "Simpangan tanpa tambahan",
         "Simpangan dengan band-gain", "Rasio"],
        [
            [r["arch"], r["axis"], f"{r['baseline'].std(ddof=1):.2f}",
             f"{r['treated'].std(ddof=1):.2f}",
             spread_ratio(r["baseline"], r["treated"])]
            for r in rows
            if r["addition"] == "band-gain"
        ],
    )

    report.add("## Bacaan\n")
    if not any(r["test"].significant for r in rows):
        report.add(
            "Tidak satu pun klaim bertahan. Seluruh selisih, termasuk yang "
            "besarannya belasan sampai puluhan poin persentase, berada di dalam "
            "ragam antar inisialisasi acak.\n"
        )
    report.add(DISCUSSION + "\n")
    report.save()


if __name__ == "__main__":
    main()

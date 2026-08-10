"""Parameter sweep for the band-gain augmentation.

Band-gain is this study's own proposal, yet its three parameters were fixed once
from mechanistic reasoning and never tested. One parameter moves at a time while
the other two stay at their defaults.

The sweep runs on frozen WavLM Large because that configuration has the smallest
seed-to-seed spread here; sweeping on a noisy architecture would bury the
differences between parameter values.
"""

from __future__ import annotations

import re

from ..report import Report
from ..runs import RUNS, load, summarize
from ..stats import correct, welch

DEFAULT = ("3000", "6", "12")
ZERO = ("3000", "6", "0")

# F marks f_lo, N the band count, D the maximum attenuation.
VARIANT = re.compile(
    r"^wavlm_official_fullbg(?:F(?P<f>\d+))?(?:N(?P<n>\d+))?(?:D(?P<d>\d+))?"
    r"_b16e10_s(?P<seed>\d+)$"
)

INTRO = """Seluruh sapuan dijalankan pada WavLM Large berencoder beku dengan \
preset augmentasi penuh ditambah band-gain, pada partisi resmi dan ambang \
prior-matched. Satu parameter diubah pada satu waktu, dua yang lain \
dipertahankan pada nilai bawaannya, yaitu f_lo 3000 Hz, enam pita, dan redaman \
sampai 12 dB.

Konfigurasi ini dipilih karena simpangan bakunya terkecil di antara seluruh \
konfigurasi dalam penelitian ini. Menyapu parameter pada arsitektur dengan ragam \
besar akan menenggelamkan selisih antar parameter di dalam derau antar \
inisialisasi."""

TWO_ANCHORS = """Sebuah titik hanya berguna bila mengungguli konfigurasi bawaan \
dan titik tanpa band-gain sekaligus. Mengungguli bawaan saja tidak cukup, karena \
hal itu juga akan terjadi bila band-gain sebaiknya dilemahkan sampai hampir \
tidak ada."""

NOT_TESTABLE = """Belum ada titik selain bawaan yang memiliki lebih dari satu \
inisialisasi acak, sehingga belum ada yang dapat diuji. Sapuan dengan satu \
inisialisasi per titik hanya menunjukkan bentuk kurva, bukan besaran yang dapat \
dipertanggungjawabkan, dan itu berlaku juga bila selisihnya tampak besar."""


def collect() -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    """Sweep points keyed by (f_lo, bands, dB), plus tags the pattern missed.

    The zero anchor matters: without a point that has band-gain switched off
    entirely, a "gentler is better" trend reads as if the augmentation only ever
    hurts. Preset ``full`` is that anchor, identical apart from band-gain.
    """
    points: dict[tuple[str, str, str], list] = {}
    skipped: list[str] = []

    zero_runs = load("wavlm_official_full_b16e10_s*")
    if zero_runs:
        points[ZERO] = zero_runs

    for path in sorted(RUNS.glob("wavlm_official_fullbg*")):
        match = VARIANT.match(path.name)
        if not match:
            if (path / "test_scores.npy").exists():
                skipped.append(path.name)
            continue
        runs = load(path.name)
        if not runs:
            continue
        key = (match["f"] or "3000", match["n"] or "6", match["d"] or "12")
        points.setdefault(key, []).extend(runs)

    return (
        {
            key: {
                "accuracy": summarize(runs, "accuracy", "prior"),
                "auc": summarize(runs, "auc", 0.5).mean,
                "eer": summarize(runs, "eer", 0.5).mean,
            }
            for key, runs in points.items()
        },
        skipped,
    )


def anchored_tests(points: dict) -> list[tuple[str, str, object]]:
    """Every sweep point against both the default and the zero anchor."""
    anchors = [("bawaan 12 dB", points.get(DEFAULT)), ("tanpa band-gain", points.get(ZERO))]
    pending = []

    default, zero = anchors[0][1], anchors[1][1]
    if default and zero and default["accuracy"].n >= 2 and zero["accuracy"].n >= 2:
        pending.append(("bawaan 12 dB", "tanpa band-gain",
                        default["accuracy"].values, zero["accuracy"].values))

    for key, value in points.items():
        if key in (DEFAULT, ZERO) or value["accuracy"].n < 2:
            continue
        for anchor_name, anchor in anchors:
            if anchor and anchor["accuracy"].n >= 2:
                pending.append((
                    f"f_lo {key[0]}, {key[1]} pita, {key[2]} dB", anchor_name,
                    value["accuracy"].values, anchor["accuracy"].values,
                ))

    tests = correct([welch(str(i), a, b) for i, (_, _, a, b) in enumerate(pending)])
    return [(point, anchor, test) for (point, anchor, _, _), test in zip(pending, tests, strict=False)]


def verdict(p_holm: float) -> str:
    if p_holm < 0.05:
        return "melampaui ragam"
    return "di garis batas" if p_holm < 0.15 else "belum terbukti berbeda"


def main() -> None:
    points, skipped = collect()

    report = Report("HASIL_BANDGAIN.md")
    report.add("# Sapuan Parameter Augmentasi Band-Gain\n")
    report.add(INTRO + "\n")

    if not points:
        report.add("Belum ada hasil.")
        report.save()
        return

    default = points.get(DEFAULT)
    rows = []
    for key in sorted(points, key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
        value = points[key]
        note = (
            " (bawaan)" if key == DEFAULT
            else " (tanpa band-gain)" if key[2] == "0" else ""
        )
        gap = (
            "" if default is None
            else f"{value['accuracy'].mean - default['accuracy'].mean:+.2f}"
        )
        rows.append([
            f"{key[0]}{note}", key[1], key[2], value["accuracy"].n,
            str(value["accuracy"]), gap, f"{value['auc'] / 100:.4f}",
            f"{value['eer']:.2f}",
        ])
    report.table(
        ["f_lo (Hz)", "jumlah pita", "redaman maks (dB)", "n", "Akurasi",
         "Selisih dari bawaan", "AUC", "EER"],
        rows,
    )

    if default is not None and len(points) > 1:
        others = [(k, v) for k, v in points.items() if k != DEFAULT]
        best_key, best = max(others, key=lambda item: item[1]["accuracy"].mean)
        report.add(
            f"Nilai bawaan mencapai {default['accuracy'].mean:.2f} persen. "
            f"Kombinasi terbaik dalam sapuan ini adalah f_lo {best_key[0]} Hz "
            f"dengan {best_key[1]} pita dan redaman {best_key[2]} dB, yaitu "
            f"{best['accuracy'].mean:.2f} persen atau "
            f"{best['accuracy'].mean - default['accuracy'].mean:+.2f} poin "
            "persentase.\n"
        )

    tests = anchored_tests(points)
    if tests:
        report.add("## Pengujian terhadap dua acuan\n")
        report.add(TWO_ANCHORS + "\n")
        report.table(
            ["Titik", "Acuan", "n", "Selisih", "p mentah", "p Holm", "Bacaan"],
            [
                [point, anchor, f"{len(test.a)}/{len(test.b)}", f"{test.diff:+.2f}",
                 f"{test.p_raw:.4f}", f"{test.p_holm:.4f}",
                 f"**{verdict(test.p_holm)}**"]
                for point, anchor, test in tests
            ],
        )
    else:
        report.add(NOT_TESTABLE + "\n")

    if skipped:
        report.add("## Cakupan\n")
        report.add(
            f"Sebanyak {len(skipped)} run bertag fullbg tidak masuk sapuan ini "
            "karena nama tag-nya di luar pola yang ditangani. Jumlahnya dicatat "
            "supaya pelewatan tidak berlangsung tanpa diketahui.\n"
        )
        report.add(
            "Tag yang dilewati: "
            + ", ".join(f"`{tag}`" for tag in skipped[:8])
            + (f", dan {len(skipped) - 8} lainnya." if len(skipped) > 8 else ".")
        )
        report.add("")

    report.save()
    if skipped:
        print(f"   catatan: {len(skipped)} run fullbg dilewati pola tag")


if __name__ == "__main__":
    main()

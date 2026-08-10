"""Do the reported correlations survive their own sample size?

Two correlations were reported without a p-value or an interval. At these sample
sizes a coefficient can look large while coming from unrelated data, so the
p-value here is a permutation test and the interval is a percentile bootstrap.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..report import Report
from ..results import MODERN_ERA, _load
from ..runs import load, parse
from ..stats import bootstrap_correlation_ci, permutation_correlation

FOR_TEST_SIZE = 1088

INTRO = """Dua korelasi dilaporkan dalam penelitian ini tanpa nilai p maupun \
selang kepercayaan. Pada ukuran sampel kecil, koefisien korelasi memiliki \
sebaran yang sangat lebar sehingga nilai besar sekalipun dapat muncul dari data \
yang sebenarnya tidak berhubungan. Nilai p di bawah dihitung dengan uji \
permutasi lengkap, yang eksak untuk ukuran sampel sekecil ini dan tidak \
mengandaikan kenormalan. Selang kepercayaan dihitung dengan bootstrap \
persentil."""

CAVEAT = """Perlu ditegaskan bahwa yang menopang temuan mengenai hubungan \
terbalik ini bukan koefisien korelasinya, melainkan mekanismenya yang \
terdokumentasi secara terpisah, yaitu audit kebocoran codec pada dataset, \
eksperimen augmentasi terkontrol, dan pola kebutaan model terhadap sistem yang \
tidak dikompresi. Koefisien korelasi pada ukuran sampel sekecil ini sebaiknya \
dibaca sebagai ringkasan deskriptif, bukan sebagai bukti."""


def paired_configurations() -> dict[tuple[str, str], tuple[float, float]]:
    """Mean Fake-or-Real accuracy against mean modern-TTS recall, per configuration."""
    generations = _load("generations_results.json") or {}
    accuracy = {
        run.tag: run.metrics("prior")["accuracy"] * 100
        for run in load("*")
        if len(run.scores[0]) == FOR_TEST_SIZE
    }

    grouped: dict[tuple[str, str], tuple[list, list]] = defaultdict(lambda: ([], []))
    for tag, row in generations.items():
        config = parse(tag)
        if not config or config[0].split != "official" or tag not in accuracy:
            continue
        modern = [
            system["recall"] for system in row["tts"].values()
            if system["era"] == MODERN_ERA
        ]
        if not modern:
            continue
        key = (config[0].model, f"{config[0].aug}@{config[0].suffix}")
        grouped[key][0].append(accuracy[tag])
        grouped[key][1].append(float(np.mean(modern)) * 100)

    return {
        key: (float(np.mean(acc)), float(np.mean(recall)))
        for key, (acc, recall) in sorted(grouped.items())
    }


def main() -> None:
    report = Report("HASIL_UJI_KORELASI.md")
    report.add("# Apakah Korelasi yang Dilaporkan Bertahan pada Ukuran Sampelnya?\n")
    report.add(INTRO + "\n")

    pairs = paired_configurations()
    if len(pairs) < 4:
        report.add("Titik data belum cukup untuk menguji korelasi.")
        report.save()
        return

    x = [value[0] for value in pairs.values()]
    y = [value[1] for value in pairs.values()]
    correlation = permutation_correlation(x, y)
    interval = bootstrap_correlation_ci(x, y)

    report.add("## Akurasi Fake-or-Real terhadap recall TTS generasi terbaru\n")
    report.add(
        f"Titik data: {len(x)} konfigurasi, yaitu pasangan arsitektur dan strategi "
        "augmentasi, masing-masing dirata-ratakan atas inisialisasi acak yang "
        "tersedia.\n"
    )
    report.table(
        ["Konfigurasi", "Akurasi FoR", "Recall TTS 2025-2026"],
        [
            [f"{model} + {aug}", f"{acc:.2f}", f"{recall:.2f}"]
            for (model, aug), (acc, recall) in sorted(
                pairs.items(), key=lambda item: -item[1][0]
            )
        ],
    )

    how = (
        "lengkap atas " + f"{correlation.permutations:,}".replace(",", ".") + " permutasi"
        if correlation.exact
        else "acak"
    )
    report.add(
        f"Koefisien korelasi Pearson r = {correlation.r:.3f} dengan n = {len(x)}. "
        f"Nilai p dua sisi dari uji permutasi {how} adalah {correlation.p:.4f}."
    )
    if interval:
        report.add(
            "Selang kepercayaan bootstrap 95 persen membentang dari "
            f"{interval[0]:.3f} sampai {interval[1]:.3f}."
        )
    report.add("")

    if interval and interval[0] < 0 < interval[1]:
        report.add(
            "Selang kepercayaannya memuat nol. Arah hubungan karena itu belum dapat "
            "ditetapkan dari data ini saja, sekalipun koefisien titiknya bertanda "
            "negatif.\n"
        )
    elif correlation.p < 0.05:
        report.add("Korelasi ini bertahan pada tingkat lima persen.\n")
    else:
        report.add("Korelasi ini belum terbukti berbeda dari nol.\n")

    report.add(CAVEAT + "\n")
    report.save()


if __name__ == "__main__":
    main()

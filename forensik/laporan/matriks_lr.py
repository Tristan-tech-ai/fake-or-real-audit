"""Architecture against encoder treatment, one variable at a time.

The proposal fixed one learning rate for every architecture; the engineered
configuration froze the encoder for every architecture. Each choice happens to
suit one architecture and to hurt another, which this matrix makes visible.
"""

from __future__ import annotations

from ..report import Report
from ..runs import scores

TREATMENTS = [
    ("Encoder dibekukan", "{model}_official_full_b{batch}e10_s*"),
    ("Encoder dilatih, laju wajar per model", "{model}_official_fullUF_b{batch}e10_s*"),
    ("Encoder dilatih, laju 0,001", "{model}_official_fullUFENC0.001_b{batch}e10_s*"),
    (
        "Proposal apa adanya, laju 0,001 seragam",
        "{model}_official_proposalULRPK_b{batch}e20_s*",
    ),
]

ARCHITECTURES = [
    ("ast", 32, "AST, 86 juta parameter, pra-latih terselia"),
    ("wavlm", 16, "WavLM Large, 300 juta parameter, swa-selia"),
    ("hubert", 32, "HuBERT Large, 300 juta parameter, swa-selia"),
]

INTRO = """Seluruh angka diukur pada partisi resmi Fake-or-Real dengan ambang \
prior-matched. Tiga baris pertama memakai paket rekayasa yang sama persis, yaitu \
10 epoch dengan early stopping pada equal error rate, augmentasi penuh, \
normalisasi loudness, dan agregasi berbobot antar lapisan. Hanya perlakuan \
encoder yang berbeda di antara ketiganya, sehingga perbandingannya bersifat satu \
variabel. Baris keempat disertakan sebagai acuan, yaitu konfigurasi proposal apa \
adanya."""

CLOSING = """Arah pengaruhnya tidak sama antar arsitektur. Tidak ada satu \
perlakuan encoder yang benar untuk semuanya. Inilah sebabnya baik penyeragaman \
learning rate pada proposal maupun penyeragaman pembekuan encoder pada \
konfigurasi rekayasa sama-sama menghasilkan kerugian pada arsitektur yang tidak \
cocok dengan pilihan tersebut. Keputusan ini seharusnya ditetapkan per \
arsitektur dan dipilih menggunakan data validasi, bukan diseragamkan di muka."""


def cell(model: str, batch: int, pattern: str) -> dict | None:
    glob = pattern.format(model=model, batch=batch)
    accuracy = scores(glob, "accuracy", "prior")
    if accuracy is None:
        return None
    return {
        "accuracy": accuracy,
        "auc": scores(glob, "auc", 0.5).mean,
        "eer": scores(glob, "eer", 0.5).mean,
    }


def main() -> None:
    report = Report("HASIL_MATRIKS_LR.md")
    report.add("# Matriks Arsitektur terhadap Perlakuan Encoder\n")
    report.add(INTRO + "\n")

    present = []
    for model, batch, caption in ARCHITECTURES:
        rows = [(name, cell(model, batch, pattern)) for name, pattern in TREATMENTS]
        if not any(value for _, value in rows):
            continue
        present.append((caption, rows))

        best = max((v["accuracy"].mean for _, v in rows if v), default=None)
        report.add(f"## {caption}\n")
        report.table(
            ["Perlakuan encoder", "n", "Akurasi", "AUC", "EER"],
            [
                [name, "", "belum ada", "", ""]
                if not value
                else [
                    name,
                    value["accuracy"].n,
                    f"**{value['accuracy']}**"
                    if value["accuracy"].mean == best
                    else str(value["accuracy"]),
                    f"{value['auc'] / 100:.4f}",
                    f"{value['eer']:.2f}",
                ]
                for name, value in rows
            ],
        )

    if len(present) >= 2:
        report.add("## Bacaan\n")
        for caption, rows in present:
            found = {name: value for name, value in rows if value}
            frozen = found.get("Encoder dibekukan")
            tuned = found.get("Encoder dilatih, laju wajar per model")
            if frozen and tuned:
                diff = tuned["accuracy"].mean - frozen["accuracy"].mean
                direction = "lebih baik daripada" if diff > 0 else "lebih buruk daripada"
                report.add(
                    f"Pada {caption.split(',')[0]}, melatih encoder pada laju wajar "
                    f"{direction} membekukannya, dengan selisih {diff:+.2f} poin "
                    "persentase.\n"
                )
        report.add(CLOSING + "\n")

    report.save()


if __name__ == "__main__":
    main()

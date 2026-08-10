"""Guard against comparing runs that differ in more than their seed.

Seven mistakes in this study came from the same pattern: two runs compared as a
controlled pair while their epoch counts differed. This groups every run by
architecture, split, and augmentation, then fails if a group holds more than one
training configuration without a recorded reason.

Exit code 1 stops the reporting pipeline, so a mixed group has to be resolved or
registered in ``ALLOWED`` before anything is published.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from .runs import load

ALLOWED = {
    ("cnn_asp", "random", "none"): (
        "run enam epoch dari tahap awal sengaja disimpan sebagai catatan sejarah "
        "di samping run sepuluh epoch, lihat HASIL_TEMUAN1.md"
    ),
    ("cnn_asp", "official", "none"): (
        "run satu epoch dari tahap awal sengaja disimpan sebagai catatan sejarah "
        "di samping run sepuluh epoch, lihat HASIL_TEMUAN1.md"
    ),
    ("ast", "official", "proposal"): (
        "dua langkah tangga ablasi yang memang berbeda jumlah epoch, yaitu L3 "
        "dengan dua puluh epoch dan L4 dengan sepuluh epoch"
    ),
    ("cnn_asp", "wavval", "codec"): (
        "run empat belas epoch dari tahap awal, tidak dipakai dalam naskah"
    ),
    ("hubert", "official", "full"): (
        "dijalankan pada batch enam belas untuk pengujian lintas generasi dan "
        "batch tiga puluh dua untuk matriks perlakuan encoder"
    ),
}


def configurations_per_group() -> dict[tuple[str, str, str], set]:
    """Distinct (epochs, batch) pairs seen under each architecture and setup."""
    groups: dict[tuple[str, str, str], set] = defaultdict(set)
    for run in load("*"):
        args = run.args.get("args")
        if not args:
            continue
        key = (run.config.model, run.config.split, run.config.aug)
        groups[key].add((args.get("epochs"), args.get("batch")))
    return groups


def main() -> int:
    groups = configurations_per_group()
    mixed = {key: value for key, value in groups.items() if len(value) > 1}
    unregistered = {key: value for key, value in mixed.items() if key not in ALLOWED}

    print(f"kelompok run diperiksa : {len(groups)}")
    print(f"memuat lebih dari satu konfigurasi : {len(mixed)}")
    print(f"sudah didaftarkan sebagai pengecualian : {len(mixed) - len(unregistered)}")

    if mixed:
        print("\nkelompok yang tercampur:")
        for key, value in sorted(mixed.items()):
            print(f"  [{'terdaftar' if key in ALLOWED else 'BARU'}] "
                  f"{'_'.join(key)}: {sorted(value)}")
            if key in ALLOWED:
                print(f"           alasan: {ALLOWED[key]}")

    if unregistered:
        print(
            "\nGAGAL. Kelompok di atas yang bertanda BARU memuat lebih dari satu "
            "konfigurasi pelatihan tanpa alasan yang tercatat."
            "\nMenggabungkannya akan melaporkan ragam antar konfigurasi sebagai "
            "ragam antar inisialisasi acak."
            "\nPerbaiki dengan menyamakan konfigurasinya, atau daftarkan pada "
            "ALLOWED di forensik/periksa_konfigurasi.py beserta alasannya."
        )
        return 1

    print("\nOK. Tidak ada kelompok yang tercampur tanpa alasan yang tercatat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

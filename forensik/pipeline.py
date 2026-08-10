"""Regenerate every report, figure, and the manuscript, in dependency order.

    py -m forensik.pipeline

The configuration guard runs first and aborts everything on failure: a mixed
group would report the spread between configurations as if it were seed noise.
"""

from __future__ import annotations

import importlib
import sys
import traceback

from . import periksa_konfigurasi

REPORTS = [
    "forensik.laporan.audit_codec",
    "forensik.laporan.temuan1",
    "forensik.laporan.signifikansi",
    "forensik.laporan.matriks_lr",
    "forensik.laporan.ablasi",
    "forensik.laporan.bandgain",
    "forensik.laporan.kalibrasi",
    "forensik.laporan.korelasi",
    "forensik.laporan.klaim_bandgain",
]


def run(module: str) -> bool:
    try:
        importlib.import_module(module).main()
        return True
    except Exception:
        print(f"GAGAL {module}", file=sys.stderr)
        traceback.print_exc()
        return False


def main() -> int:
    if periksa_konfigurasi.main() != 0:
        print("DIHENTIKAN: ada kelompok run dengan konfigurasi tercampur.")
        return 1

    failed = [module for module in REPORTS if not run(module)]

    from . import gambar
    from .naskah import build

    gambar.main()
    build.main()

    if failed:
        print(f"\nselesai dengan {len(failed)} laporan gagal: {', '.join(failed)}")
        return 1
    print(f"\nselesai: {len(REPORTS)} laporan, {len(gambar.FIGURES)} gambar, NASKAH.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())

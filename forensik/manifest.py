"""Dataset-level audit of the Fake-or-Real manifest.

Reads filenames and labels only. No model, no training, no randomness, so the
numbers here have no seed-to-seed spread and need no significance test.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.csv"

SPLITS = ("training", "validation", "testing")
CLASSES = ("real", "fake")


@dataclass(frozen=True)
class Provenance:
    """How many files of one class in one split came from an MP3 source."""

    total: int
    from_mp3: int

    @property
    def percent(self) -> float:
        return 100 * self.from_mp3 / self.total


def codec_provenance(path: Path = MANIFEST) -> dict[tuple[str, str], Provenance]:
    """MP3 provenance broken down by official split and class."""
    counts: dict[tuple[str, str], list[int]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["split_official"], "real" if row["label"] == "0" else "fake")
            tally = counts.setdefault(key, [0, 0])
            tally[0] += 1
            tally[1] += row["is_mp3"] in ("1", "True", "true")
    return {key: Provenance(*value) for key, value in counts.items()}


def split_sizes(path: Path = MANIFEST) -> dict[str, int]:
    provenance = codec_provenance(path)
    return {
        split: sum(provenance[(split, cls)].total for cls in CLASSES)
        for split in SPLITS
    }

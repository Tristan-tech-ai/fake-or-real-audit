"""Readers for the JSON files produced by the out-of-domain evaluations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SNR_LEVELS = (None, 20, 15, 10, 5, 0, -5)
MODERN_ERA = "2025-2026 komersial"


def _load(name: str):
    path = ROOT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def snr_runs() -> dict[tuple[str, int | None], list[dict]]:
    """Noise-sweep results keyed by (architecture, signal-to-noise ratio)."""
    grouped: dict[tuple[str, int | None], list[dict]] = defaultdict(list)
    for row in _load("snr_results.json") or []:
        grouped[(row["arch"], row["snr"])].append(row)
    return grouped


def snr_curve(metric: str = "acc_pm") -> dict[str, dict[int | None, float]]:
    """One metric averaged over seeds, per architecture and noise level."""
    curves: dict[str, dict[int | None, float]] = defaultdict(dict)
    for (arch, snr), rows in snr_runs().items():
        curves[arch][snr] = float(np.mean([row[metric] for row in rows]))
    return curves


def modern_tts_recall() -> dict[tuple[str, str], list[float]]:
    """Recall on 2025-2026 commercial TTS, keyed by (model, augmentation)."""
    from .runs import parse

    recall: dict[tuple[str, str], list[float]] = defaultdict(list)
    for tag, row in (_load("generations_results.json") or {}).items():
        parsed = parse(tag)
        if not parsed:
            continue
        modern = [
            system["recall"]
            for system in row["tts"].values()
            if system["era"] == MODERN_ERA
        ]
        if modern:
            recall[(parsed[0].model, parsed[0].aug)].append(float(np.mean(modern)) * 100)
    return recall


def public_baseline() -> dict | None:
    """A published detector evaluated on Fake-or-Real and on modern TTS."""
    return _load("sota_modern_results.json")
